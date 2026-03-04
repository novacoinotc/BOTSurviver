"""Pre-trade risk validation. Claude cannot override these rules."""

import logging
from config.settings import settings
from core.models import ActionType, TradeDecision

logger = logging.getLogger(__name__)


class RiskManager:
    """Validates every trade decision before execution. Non-negotiable rules."""

    def __init__(self, current_params: dict = None):
        self._params = current_params or {}
        self._fear_greed: int = 50  # neutral default

    def update_params(self, params: dict):
        self._params = params

    def update_fear_greed(self, value: int):
        self._fear_greed = value

    def _get(self, key: str, default):
        return self._params.get(key, default)

    def validate(
        self,
        decision: TradeDecision,
        balance: float,
        open_positions: int,
        has_position_for_pair: bool,
        circuit_breaker_active: bool,
        margin_ratio: float = 0.0,
        current_positions: list[dict] = None,
    ) -> tuple[bool, str]:
        """Validate a trade decision. Returns (is_valid, rejection_reason)."""

        # Circuit breaker overrides everything
        if circuit_breaker_active and decision.action in (ActionType.ENTER_LONG, ActionType.ENTER_SHORT):
            return False, "Circuit breaker is active - no new trades allowed"

        # HOLD and EXIT are always allowed
        if decision.action == ActionType.HOLD:
            return True, ""

        if decision.action == ActionType.EXIT:
            if not has_position_for_pair:
                return False, f"No position to exit for {decision.pair}"
            return True, ""

        if decision.action == ActionType.ADJUST:
            if not has_position_for_pair:
                return False, f"No position to adjust for {decision.pair}"
            return True, ""

        # Entry validation
        if decision.action in (ActionType.ENTER_LONG, ActionType.ENTER_SHORT):
            # Must not already have position on this pair
            if has_position_for_pair:
                return False, f"Already have position on {decision.pair}"

            # Max open positions
            max_pos = int(self._get("max_open_positions", settings.max_open_positions))
            if open_positions >= max_pos:
                return False, f"Max positions reached ({open_positions}/{max_pos})"

            # Correlation check: max 3 positions in the same direction
            if current_positions:
                dir_value = "LONG" if decision.action == ActionType.ENTER_LONG else "SHORT"
                same_dir = sum(1 for p in current_positions if p.get("direction") == dir_value)
                if same_dir >= 3:
                    return False, f"Max 3 {dir_value} positions (correlation risk)"

            # Confidence threshold
            min_score = self._get("min_score_to_enter", settings.min_score_to_enter)
            if decision.confidence < min_score:
                return False, f"Confidence {decision.confidence:.2f} below threshold {min_score}"

            # Fear-based confidence adjustment (graduated, not a wall)
            if self._fear_greed < 10 and decision.confidence < 0.72:
                return False, f"Extreme fear (F&G={self._fear_greed}): need confidence >= 0.72, got {decision.confidence:.2f}"
            elif self._fear_greed < 20 and decision.confidence < 0.65:
                return False, f"High fear (F&G={self._fear_greed}): need confidence >= 0.65, got {decision.confidence:.2f}"

            # Leverage check
            max_lev = settings.max_leverage
            if decision.leverage and decision.leverage > max_lev:
                return False, f"Leverage {decision.leverage}x exceeds max {max_lev}x"

            # Position size check
            pos_pct = decision.position_size_pct or settings.default_position_pct
            if pos_pct > settings.max_position_pct:
                return False, f"Position size {pos_pct:.3%} exceeds max {settings.max_position_pct:.3%}"

            # Stop loss MUST be set
            if not decision.stop_loss or decision.stop_loss <= 0:
                return False, "Stop loss is mandatory for every trade"

            # Take profit MUST be set
            if not decision.take_profit or decision.take_profit <= 0:
                return False, "Take profit is mandatory for every trade"

            # Minimum SL distance check (1H strategy uses 3% SL)
            leverage = decision.leverage or settings.default_leverage
            entry_approx = decision.entry_price or 0
            if entry_approx > 0 and decision.stop_loss > 0:
                sl_distance_pct = abs(entry_approx - decision.stop_loss) / entry_approx
                if sl_distance_pct < 0.008:  # 0.8% minimum SL distance (1H uses 3%)
                    return False, f"SL too tight: {sl_distance_pct:.2%} from entry (min 0.80%)"

                # Actual risk calculation based on SL distance
                actual_risk_pct = pos_pct * leverage * sl_distance_pct
                max_risk = settings.max_risk_per_trade_pct
                if actual_risk_pct > max_risk * 2:  # give some margin
                    return False, f"Risk too high: {actual_risk_pct:.2%} of capital at risk (max {max_risk:.2%})"

            # Margin ratio check - don't over-leverage
            if margin_ratio > 0.70:
                return False, f"Margin ratio too high ({margin_ratio:.0%}), reduce positions before adding more"

            # Check we have enough balance
            margin_needed = balance * pos_pct
            if margin_needed > balance * 0.95:  # keep 5% buffer
                return False, f"Margin {margin_needed:.2f} too high for balance {balance:.2f}"

            return True, ""

        return False, f"Unknown action type: {decision.action}"
