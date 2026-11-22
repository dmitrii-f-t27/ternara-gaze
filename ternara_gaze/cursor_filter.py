"""Saccade-Rejecting Fixation-Cluster cursor controller + adaptive stillness detector + blink click machine.

Core ergonomic principle (matched to human eye physiology):
- Eyes constantly dart around (saccades / scanning). While the rolling gaze window has high
  spatial spread (`spread > fixation_spread_u16`), the cursor DOES NOT REACT AT ALL (`0 px`).
- ONLY when the user's gaze actually STOPS on a spot for ~300 ms (`spread <= fixation_spread_u16`)
  does the controller commit that calm cluster centroid as the new destination and smoothly
  glide the second cursor directly there, locking onto the region/button on arrival.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Literal, Optional

ClickEvent = Literal["none", "left_click", "right_click"]
MotionState = Literal["scanning", "gliding", "locked"]


@dataclass(frozen=True)
class FilterConfig:
    """All timing in milliseconds; coordinates in [0, 65535]."""

    fixation_window: int = 9          # ~300 ms rolling window at 30 FPS to confirm eyes truly stopped
    fixation_spread_u16: int = 4800   # max trimmed X+Y span (~7.3% combined) to count as stopped fixation
    retarget_deadzone_u16: int = 2100 # ~3.2% screen: ignore tiny shifts around already-locked spot
    arrival_radius_u16: int = 420     # snap neatly to goal once glide arrives within ~0.6%
    glide_alpha_q8: int = 58          # silky smooth glide (~22.6% per frame) once destination is committed
    max_step_u16: int = 4600          # cap single-frame glide velocity for butter-smooth motion
    blink_threshold: int = 24
    blink_threshold_l: Optional[int] = None
    blink_threshold_r: Optional[int] = None
    ear_threshold_l: Optional[int] = None
    ear_threshold_r: Optional[int] = None
    deliberate_blink_min_ms: int = 240
    deliberate_blink_max_ms: int = 750
    wink_min_ms: int = 210
    wink_max_ms: int = 650
    click_cooldown_ms: int = 550
    override_pause_ms: int = 1500


@dataclass(frozen=True)
class CursorDecision:
    x_u16: int
    y_u16: int
    click: ClickEvent
    frozen: bool
    paused_by_user: bool
    hold_progress_pct: int = 0
    locked: bool = False          # True when parked/locked on a spot (0 px movement)
    motion_state: MotionState = "locked"  # "scanning" | "gliding" | "locked"


class GazeCursorController:
    """Saccade-rejecting fixation-cluster controller: ignores darting eyes, glides only on calm stops."""

    def __init__(self, config: Optional[FilterConfig] = None) -> None:
        self.cfg = config or FilterConfig()
        self._has_pos = False
        self._x: int = 32768
        self._y: int = 32768
        self._goal_x: int = 32768
        self._goal_y: int = 32768
        self._locked: bool = True
        self._motion_state: MotionState = "locked"

        self._win_x: Deque[int] = deque(maxlen=max(3, self.cfg.fixation_window))
        self._win_y: Deque[int] = deque(maxlen=max(3, self.cfg.fixation_window))
        self._last_click_ms: int = -10_000
        self._pause_until_ms: int = -10_000

        self._both_closed_start_ms: Optional[int] = None
        self._left_wink_start_ms: Optional[int] = None
        self._right_wink_start_ms: Optional[int] = None

    def notify_physical_mouse_moved(self, now_ms: int) -> None:
        """Yield gaze control immediately when user touches physical mouse/trackpad."""
        self._pause_until_ms = now_ms + self.cfg.override_pause_ms

    def _window_stats(self) -> tuple[int, int, int]:
        """Return robust (centroid_x, centroid_y, trimmed_spread_u16) of rolling gaze window."""
        sx = sorted(self._win_x)
        sy = sorted(self._win_y)
        n = len(sx)
        cx = sx[n // 2]
        cy = sy[n // 2]
        if n >= 7:
            # Drop 1 lowest and 1 highest sample so a single webcam glitch never hides a true stop
            spread = (sx[-2] - sx[1]) + (sy[-2] - sy[1])
        else:
            spread = (sx[-1] - sx[0]) + (sy[-1] - sy[0])
        return cx, cy, spread

    def step(
        self,
        raw_x_u16: int,
        raw_y_u16: int,
        blink_left: int,
        blink_right: int,
        now_ms: int,
        ear_l: Optional[int] = None,
        ear_r: Optional[int] = None,
    ) -> CursorDecision:
        rx = max(0, min(65535, int(raw_x_u16)))
        ry = max(0, min(65535, int(raw_y_u16)))

        tl = self.cfg.blink_threshold if self.cfg.blink_threshold_l is None else self.cfg.blink_threshold_l
        tr = self.cfg.blink_threshold if self.cfg.blink_threshold_r is None else self.cfg.blink_threshold_r
        l_closed = int(blink_left) > tl or (
            ear_l is not None and self.cfg.ear_threshold_l is not None and int(ear_l) < self.cfg.ear_threshold_l
        )
        r_closed = int(blink_right) > tr or (
            ear_r is not None and self.cfg.ear_threshold_r is not None and int(ear_r) < self.cfg.ear_threshold_r
        )

        if now_ms < self._pause_until_ms:
            self._both_closed_start_ms = None
            self._left_wink_start_ms = None
            self._right_wink_start_ms = None
            return CursorDecision(
                x_u16=self._x,
                y_u16=self._y,
                click="none",
                frozen=True,
                paused_by_user=True,
                hold_progress_pct=0,
                locked=self._locked,
                motion_state=self._motion_state,
            )

        click: ClickEvent = "none"
        hold_pct = 0
        can_click = (now_ms - self._last_click_ms) >= self.cfg.click_cooldown_ms

        if l_closed and r_closed:
            if self._both_closed_start_ms is None:
                self._both_closed_start_ms = now_ms
            dur = now_ms - self._both_closed_start_ms
            if dur <= self.cfg.deliberate_blink_max_ms:
                hold_pct = min(100, int(100 * dur // max(1, self.cfg.deliberate_blink_min_ms)))
            self._left_wink_start_ms = None
            self._right_wink_start_ms = None
        else:
            if self._both_closed_start_ms is not None:
                dur = now_ms - self._both_closed_start_ms
                self._both_closed_start_ms = None
                if (
                    can_click
                    and self.cfg.deliberate_blink_min_ms <= dur <= self.cfg.deliberate_blink_max_ms
                ):
                    click = "left_click"
                    self._last_click_ms = now_ms

            if l_closed and not r_closed:
                if self._left_wink_start_ms is None:
                    self._left_wink_start_ms = now_ms
                dur = now_ms - self._left_wink_start_ms
                if dur <= self.cfg.wink_max_ms:
                    hold_pct = min(100, int(100 * dur // max(1, self.cfg.wink_min_ms)))
                self._right_wink_start_ms = None
            else:
                if self._left_wink_start_ms is not None:
                    dur = now_ms - self._left_wink_start_ms
                    self._left_wink_start_ms = None
                    if (
                        click == "none"
                        and can_click
                        and self.cfg.wink_min_ms <= dur <= self.cfg.wink_max_ms
                    ):
                        click = "left_click"
                        self._last_click_ms = now_ms

            if r_closed and not l_closed:
                if self._right_wink_start_ms is None:
                    self._right_wink_start_ms = now_ms
                dur = now_ms - self._right_wink_start_ms
                if dur <= self.cfg.wink_max_ms:
                    hold_pct = min(100, int(100 * dur // max(1, self.cfg.wink_min_ms)))
                self._left_wink_start_ms = None
            else:
                if self._right_wink_start_ms is not None:
                    dur = now_ms - self._right_wink_start_ms
                    self._right_wink_start_ms = None
                    if (
                        click == "none"
                        and can_click
                        and self.cfg.wink_min_ms <= dur <= self.cfg.wink_max_ms
                    ):
                        click = "right_click"
                        self._last_click_ms = now_ms

        frozen = l_closed or r_closed
        if not frozen:
            self._win_x.append(rx)
            self._win_y.append(ry)

            if not self._has_pos:
                self._x = rx
                self._y = ry
                self._goal_x = rx
                self._goal_y = ry
                self._has_pos = True
                self._locked = True
                self._motion_state = "locked"
            else:
                cx, cy, spread = self._window_stats()
                is_fixated = (len(self._win_x) >= self.cfg.fixation_window) and (
                    spread <= self.cfg.fixation_spread_u16
                )

                if is_fixated:
                    # Gaze truly STOPPED on (cx, cy)! Commit as new destination if far enough from current goal.
                    if (abs(cx - self._goal_x) + abs(cy - self._goal_y)) > self.cfg.retarget_deadzone_u16:
                        self._goal_x = cx
                        self._goal_y = cy
                        self._locked = False
                        self._motion_state = "gliding"
                else:
                    # Eyes are darting / scanning across the screen -> DO NOT update destination!
                    if self._locked:
                        self._motion_state = "scanning"

                # Smoothly glide toward committed goal (if not yet arrived); otherwise stay 100% motionless!
                dx = self._goal_x - self._x
                dy = self._goal_y - self._y
                rem = abs(dx) + abs(dy)
                if rem <= self.cfg.arrival_radius_u16:
                    self._x = self._goal_x
                    self._y = self._goal_y
                    self._locked = True
                    self._motion_state = "locked" if is_fixated else "scanning"
                else:
                    self._locked = False
                    self._motion_state = "gliding"
                    step_x = (dx * self.cfg.glide_alpha_q8) >> 8
                    step_y = (dy * self.cfg.glide_alpha_q8) >> 8
                    if step_x == 0 and dx != 0:
                        step_x = 1 if dx > 0 else -1
                    if step_y == 0 and dy != 0:
                        step_y = 1 if dy > 0 else -1
                    step_x = max(-self.cfg.max_step_u16, min(self.cfg.max_step_u16, step_x))
                    step_y = max(-self.cfg.max_step_u16, min(self.cfg.max_step_u16, step_y))
                    self._x = max(0, min(65535, self._x + step_x))
                    self._y = max(0, min(65535, self._y + step_y))

        return CursorDecision(
            x_u16=self._x,
            y_u16=self._y,
            click=click,
            frozen=frozen,
            paused_by_user=False,
            hold_progress_pct=hold_pct,
            locked=self._locked,
            motion_state=self._motion_state,
        )
