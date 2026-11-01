"""Robot analitico para preseleccion de activos."""

from .contracts import Params
from .analytics import log_returns, sample_stats, annualize, drawdown, descriptives, diagnostics
from .forecasting import (fit, cumulative_moments, terminal, path,
                          walk_forward, moments, fit_arima_garch)
from .data import load_fixture, fetch_prices, resample_prices, coverage, align
from .risk_rules import var, levels, terminal_probs, signal_gate
from .compare import summary, preselect
from .pipeline import analyze
from .contracts import CapmParams
from .capm import equal_weight_index, excess_returns, regress, analyze_asset, capm_table

__all__ = [
        "Params", "load_fixture", "fetch_prices", "resample_prices", "coverage", "align", 
        "log_returns", "sample_stats", "annualize", "drawdown", "descriptives", 
        "diagnostics", "fit", "cumulative_moments", "terminal", "path", "walk_forward",
        "moments", "fit_arima_garch",
        "var", "levels", "terminal_probs", "signal_gate", "summary", "preselect", "analyze",
        "CapmParams", "equal_weight_index", "excess_returns", "regress",
        "analyze_asset", "capm_table"]