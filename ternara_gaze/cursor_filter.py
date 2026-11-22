"""Fixed-point cursor smoothing (integer 1-Euro-like adaptive EMA) and safe blink/wink clicker.

Safety & Ergonomics design:
1. Smooth fixation + fast saccades: adaptive alpha increases smoothly when velocity rises,
   implemented strictly with integer arithmetic (Q8 alpha in [alpha_min, alpha_max]).
2. Reflex-blink rejection: normal human blinks last ~100-170 ms.
   - Left click requires a deliberate both-eye blink lasting [260 ms .. 750 ms] OR a clean
     left-eye wink lasting [220 ms .. 650 ms].
   - Right click triggers on a clean right-eye wink lasting [240 ms .. 650 ms].
   - Eyes held closed > 850 ms enter a rest/pause state (no spurious clicks!).
   - Cursor freezes while eyes are closing/closed so clicking doesn't drift off target.
3. Physical-mouse override: if the user touches the physical mouse/trackpad, gaze control
   automatically yields for `override_pause_ms` (default 1500 ms) so you NEVER fight the cursor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

ClickEvent = Literal["none", "left_click", "right_click"]


@dataclass(frozen=True)
class FilterConfig:
    alpha_min_q8: int = 38          # ~0.15 smooth fixation (out of 256)
    alpha_max_q8: int = 192         # ~0.75 fast saccade tracking
    velocity_boost_shift: int = 5   # higher -> smoother; lower -> snappier
    deadzone_u16: int = 220         # ~0.33% of screen width micro-jitter suppression
    blink_threshold: int = 12       # integer logit threshold for closed eye
    deliberate_blink_min_ms: int = 260
    deliberate_blink_max_ms: int = 750
    wink_min_ms: int = 220
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


class GazeCursorController:
    """Stateful fixed-point cursor filter + safe blink/wink click state machine."""

    def __init__(self, config: Optional[FilterConfig] = None) -> None:
        self.cfg = config or FilterConfig()
        self._has_pos = False
        self._x: int = 32768
        self._y: int = 32768
        self._last_click_ms: int = -10_000
        self._pause_until_ms: int = -10_000

        # Closure tracking
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
    ) -> CursorDecision:
        raw_x = max(0, min(65535, int(raw_x_u16)))
        raw_y = max(0, min(65535, int(raw_y_u16)))
        l_closed = int(blink_left) > self.cfg.blink_threshold
        r_closed = int(blink_right) > self.cfg.blink_threshold

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
            )

        click: ClickEvent = "none"
        can_click = (now_ms - self._last_click_ms) >= self.cfg.click_cooldown_ms

        # State machine for both-eye deliberate blink vs single-eye wink
        if l_closed and r_closed:
            if self._both_closed_start_ms is None:
                self._both_closed_start_ms = now_ms
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

        # Freeze cursor movement while either eye is closed so blink clicks stay pinned on target
        frozen = l_closed or r_closed
        if not self._has_pos:
            self._x = raw_x
            self._y = raw_y
            self._has_pos = True
        elif not frozen:
            dx = raw_x - self._x
            dy = raw_y - self._y
            dist = abs(dx) + abs(dy)
            if dist > self.cfg.deadzone_u16:
                boost = dist >> self.cfg.velocity_boost_shift
                alpha = min(self.cfg.alpha_max_q8, self.cfg.alpha_min_q8 + boost)
                self._x = max(0, min(65535, self._x + ((dx * alpha) >> 8)))
                self._y = max(0, min(65535, self._y + ((dy * alpha) >> 8)))

        return CursorDecision(
            x_u16=self._x,
            y_u16=self._y,
            click=click,
            frozen=frozen,
            paused_by_user=False,
        )
