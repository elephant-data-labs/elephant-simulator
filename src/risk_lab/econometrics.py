from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm, t


@dataclass(frozen=True)
class AR1Experiment:
    observations: int = 1_000
    replications: int = 1_000
    phi_y: float = 0.95
    phi_x: float = 0.95
    sigma_y: float = 1.0
    sigma_x: float = 1.0
    burn_in: int = 500
    hac_lags: int = 6
    seed: int = 42


def _simulate_ar1(
    rng: np.random.Generator,
    replications: int,
    observations: int,
    phi: float,
    sigma: float,
    burn_in: int,
) -> np.ndarray:
    total = observations + burn_in
    innovations = rng.normal(0.0, sigma, size=(replications, total))
    series = np.zeros_like(innovations)
    for index in range(1, total):
        series[:, index] = phi * series[:, index - 1] + innovations[:, index]
    return series[:, burn_in:]


def simulate_independent_ar1_regressions(
    settings: AR1Experiment,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate independent stationary AR(1) series and regress Y on 1 and X."""
    if settings.observations < 30:
        raise ValueError("Use al menos 30 observaciones por serie.")
    if settings.replications < 1:
        raise ValueError("Use al menos una réplica.")
    if not 0 <= settings.phi_y < 1 or not 0 <= settings.phi_x < 1:
        raise ValueError("Los coeficientes AR deben estar entre 0 y menos de 1.")
    if settings.sigma_y <= 0 or settings.sigma_x <= 0:
        raise ValueError("Las desviaciones de las innovaciones deben ser positivas.")
    if settings.burn_in < 0 or not 0 <= settings.hac_lags < settings.observations:
        raise ValueError("Revise el calentamiento y los rezagos HAC.")

    rng = np.random.default_rng(settings.seed)
    y = _simulate_ar1(
        rng, settings.replications, settings.observations,
        settings.phi_y, settings.sigma_y, settings.burn_in,
    )
    x = _simulate_ar1(
        rng, settings.replications, settings.observations,
        settings.phi_x, settings.sigma_x, settings.burn_in,
    )

    x_mean = x.mean(axis=1)
    y_mean = y.mean(axis=1)
    x_centered = x - x_mean[:, None]
    y_centered = y - y_mean[:, None]
    sum_x2 = np.sum(x_centered**2, axis=1)
    beta = np.sum(x_centered * y_centered, axis=1) / sum_x2
    alpha = y_mean - beta * x_mean
    residuals = y - alpha[:, None] - beta[:, None] * x

    degrees_freedom = settings.observations - 2
    sum_resid2 = np.sum(residuals**2, axis=1)
    residual_variance = sum_resid2 / degrees_freedom
    classical_beta_se = np.sqrt(residual_variance / sum_x2)
    classical_alpha_se = np.sqrt(
        residual_variance * (1 / settings.observations + x_mean**2 / sum_x2)
    )
    t_classical_beta = beta / classical_beta_se
    p_classical_beta = 2 * t.sf(np.abs(t_classical_beta), df=degrees_freedom)
    t_classical_alpha = alpha / classical_alpha_se
    p_classical_alpha = 2 * t.sf(np.abs(t_classical_alpha), df=degrees_freedom)

    # HAC/Newey-West covariance in the centered-regressor basis [1, X - mean(X)].
    scores = x_centered * residuals
    meat_alpha = np.sum(residuals**2, axis=1)
    meat_beta = np.sum(scores**2, axis=1)
    meat_cross = np.sum(residuals * scores, axis=1)
    for lag in range(1, settings.hac_lags + 1):
        bartlett_weight = 1 - lag / (settings.hac_lags + 1)
        alpha_covariance = np.sum(residuals[:, lag:] * residuals[:, :-lag], axis=1)
        beta_covariance = np.sum(scores[:, lag:] * scores[:, :-lag], axis=1)
        cross_covariance = np.sum(
            residuals[:, lag:] * scores[:, :-lag]
            + scores[:, lag:] * residuals[:, :-lag], axis=1
        )
        meat_alpha += 2 * bartlett_weight * alpha_covariance
        meat_beta += 2 * bartlett_weight * beta_covariance
        meat_cross += bartlett_weight * cross_covariance
    correction = settings.observations / degrees_freedom
    hac_alpha_se = np.sqrt(np.maximum(meat_alpha * correction, 0.0) / settings.observations**2)
    hac_beta_se = np.sqrt(np.maximum(meat_beta * correction, 0.0) / sum_x2**2)
    hac_t_alpha = alpha / hac_alpha_se
    hac_p_alpha = 2 * norm.sf(np.abs(hac_t_alpha))
    hac_t_beta = beta / hac_beta_se
    hac_p_beta = 2 * norm.sf(np.abs(hac_t_beta))

    sum_y_centered2 = np.sum(y_centered**2, axis=1)
    r_squared = 1 - sum_resid2 / sum_y_centered2
    adjusted_r_squared = 1 - (1 - r_squared) * (settings.observations - 1) / degrees_freedom
    log_likelihood = -settings.observations / 2 * (
        np.log(2 * np.pi) + 1 + np.log(sum_resid2 / settings.observations)
    )
    parameter_count = 3  # constante, pendiente y varianza residual
    aic = -2 * log_likelihood + 2 * parameter_count
    bic = -2 * log_likelihood + parameter_count * np.log(settings.observations)
    durbin_watson = np.sum(np.diff(residuals, axis=1)**2, axis=1) / sum_resid2

    results = pd.DataFrame({
        "Réplica": np.arange(1, settings.replications + 1),
        "Constante estimada": alpha,
        "Beta estimado": beta,
        "EE OLS constante": classical_alpha_se,
        "t OLS constante": t_classical_alpha,
        "p OLS constante": p_classical_alpha,
        "EE OLS beta": classical_beta_se,
        "t OLS usual": t_classical_beta,
        "p OLS usual": p_classical_beta,
        "EE HAC constante": hac_alpha_se,
        "t HAC constante": hac_t_alpha,
        "p HAC constante": hac_p_alpha,
        "EE HAC beta": hac_beta_se,
        "t HAC": hac_t_beta,
        "p HAC": hac_p_beta,
        "R cuadrado": r_squared,
        "R cuadrado ajustado": adjusted_r_squared,
        "Error estándar regresión": np.sqrt(residual_variance),
        "Suma residuos cuadrados": sum_resid2,
        "Log likelihood": log_likelihood,
        "AIC": aic,
        "BIC": bic,
        "Durbin-Watson": durbin_watson,
        "Media Y": y_mean,
        "Desv. estándar Y": y.std(axis=1, ddof=1),
        "Rechaza OLS": p_classical_beta < 0.05,
        "Rechaza HAC": hac_p_beta < 0.05,
    })
    first_run = pd.DataFrame({
        "Periodo": np.arange(1, settings.observations + 1),
        "Y": y[0],
        "X": x[0],
        "Residuo OLS": residuals[0],
    })
    return results, first_run
