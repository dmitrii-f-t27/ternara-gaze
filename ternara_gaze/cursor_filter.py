"""Sustained-Intent Magnetic Region-Lock smoother + 7-sample median + safe blink/wink click machine.

Eliminates cursor wander completely:
- When LOCKED on a screen region, output stays 100% STATIC (0 px movement).
- Lock ONLY releases when sustained gaze stays outside `unlock_radius_u16` for
  `unlock_confirm_frames` consecutive frames (~165 ms), so 1-4 frame glitches never break lock.
- While GLIDING to a new zone, velocity-adaptive integer 1-Euro smoothing glides swiftly,
  then snaps back into LOCKED mode once gaze settles inside `settle_radius_u16` for
  `settle_confirm_frames` frames.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Literal, Optional

ClickEvent = Literal["none", "left_click", "right_click"]


def _median_int(buf: Deque[int]) -> int:
    s = sorted(buf)
    return s[len(s) // 2]


@dataclass(frozen=True)
class FilterConfig:
    """All timing in milliseconds; coordinates in [0, 65535]."""

    median_window: int = 7            # 7-sample integer median crushes webcam outliers
    alpha_min_q8: int = 24            # smooth glide speed near target
    alpha_max_q8: int = 128           # brisk across-screen saccade speed
    deadzone_u16: int = 1100          # fine micro-deadzone while gliding (~1.7% screen)
    unlock_radius_u16: int = 3400     # ~5.2% screen radius required to break a region lock
    unlock_confirm_frames: int = 5    # ~165 ms of sustained gaze outside radius to unlock
    settle_radius_u16: int = 1750     # ~2.7% screen radius to snap into locked region
    settle_confirm_frames: int = 6    # ~200 ms steady fixation to lock onto the region
    velocity_boost_shift: int = 7
    max_step_u16: int = 5200
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
    locked: bool = False  # True when magnetically pinned on a region (0 px tremor)


class GazeCursorController:
    """Stateful fixed-point Sustained-Intent Region-Lock smoother + safe blink/wink click machine."""

    def __init__(self, config: Optional[FilterConfig] = None) -> None:
        self.cfg = config or FilterConfig()
        self._has_pos = False
        self._x: int = 32768
        self._y: int = 32768
        self._locked: bool = True
        self._escape_streak: int = 0
        self._settle_streak: int = 0
        self._hist_x: Deque[int] = deque(maxlen=max(1, self.cfg.median_window))
        self._hist_y: Deque[int] = deque(maxlen=max(1, self.cfg.median_window))
        self._last_click_ms: int = -10_000
        self._pause_until_ms: int = -10_000

        self._both_closed_start_ms: Optional[int] = None
        self._left_wink_start_ms: Optional[int] = None
        self._right_wink_start_ms: Optional[int] = None

    def notify_physical_mouse_moved(self, now_ms: int) -> None:
        """Yield gaze control immediately when user touches physical mouse/trackpad."""
        self._pause_until_ms = now_ms + self.cfg.override_pause_ms

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
            self._hist_x.append(rx)
            self._hist_y.append(ry)
            mx = _median_int(self._hist_x)
            my = _median_int(self._hist_y)

            if not self._has_pos:
                self._x = mx
                self._y = my
                self._has_pos = True
                self._locked = True
                self._escape_streak = 0
                self._settle_streak = 0
            else:
                dx = mx - self._x
                dy = my - self._y
                dist = abs(dx) + abs(dy)

                if self._locked:
                    # Require sustained deliberate gaze shift outside unlock_radius_u16 to break lock
                    if dist > self.cfg.unlock_radius_u16:
                        self._escape_streak += 1
                        if self._escape_streak >= self.cfg.unlock_confirm_frames:
                            self._locked = False
                            self._escape_streak = 0
                            self._settle_streak = 0
                    else:
                        self._escape_streak = 0

                if not self._locked:
                    if dist > self.cfg.deadzone_u16:
                        boost = (dist - self.cfg.deadzone_u16) >> self.cfg.velocity_boost_shift
                        alpha = min(self.cfg.alpha_max_q8, self.cfg.alpha_min_q8 + boost)
                        step_x = max(-self.cfg.max_step_u16, min(self.cfg.max_step_u16, (dx * alpha) >> 8))
                        step_y = max(-self.cfg.max_step_u16, min(self.cfg.max_step_u16, (dy * alpha) >> 8))
                        self._x = max(0, min(65535, self._x + step_x))
                        self._y = max(0, min(65535, self._y + step_y))

                    # Once gaze settles inside settle_radius_u16, snap cleanly back to LOCKED!
                    rem_dist = abs(mx - self._x) + abs(my - self._y)
                    if rem_dist <= self.cfg.settle_radius_u16:
                        self._settle_streak += 1
                        if self._settle_streak >= self.cfg.settle_confirm_frames:
                            self._locked = True
                            self._settle_streak = 0
                            self._escape_streak = 0
                    else:
                        self._settle_streak = 0

        return CursorDecision(
            x_u16=self._x,
            y_u16=self._y,
            click=click,
            frozen=frozen,
            paused_by_user=False,
            hold_progress_pct=hold_pct,
            locked=self._locked,
        )
