from __future__ import annotations

import ast
import operator
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import sobol as sobol_sample
from scipy.stats import norm, rankdata


class ModelError(ValueError):
    """Error de validación en supuestos, fórmulas o correlaciones."""


@dataclass(frozen=True)
class Assumption:
    name: str
    distribution: str
    p1: float
    p2: float
    p3: float = 0.0


_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_COMPARE = {ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge, ast.Eq: operator.eq, ast.NotEq: operator.ne}
_FUNCTIONS = {
    "abs": np.abs,
    "sqrt": np.sqrt,
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "min": np.minimum,
    "max": np.maximum,
}


def formula_names(formula: str) -> set[str]:
    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError as exc:
        raise ModelError(f"La fórmula no se puede interpretar: {exc.msg}.") from exc
    names: set[str] = set()

    def inspect(node: ast.AST) -> None:
        if isinstance(node, ast.Expression):
            inspect(node.body)
        elif isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                raise ModelError("La fórmula solo admite números y variables.")
        elif isinstance(node, ast.Name):
            if node.id not in _FUNCTIONS:
                names.add(node.id)
        elif isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            inspect(node.left)
            inspect(node.right)
        elif isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            inspect(node.operand)
        elif isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _COMPARE:
            inspect(node.left)
            inspect(node.comparators[0])
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCTIONS:
            if node.keywords:
                raise ModelError("Las funciones de la fórmula no aceptan argumentos con nombre.")
            for arg in node.args:
                inspect(arg)
        else:
            raise ModelError("La fórmula contiene una operación no permitida.")

    inspect(tree)
    return names


def evaluate_formula(formula: str, values: Mapping[str, np.ndarray | float]) -> np.ndarray:
    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError as exc:
        raise ModelError(f"La fórmula no se puede interpretar: {exc.msg}.") from exc

    def evaluate(node: ast.AST):
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in _FUNCTIONS:
                return _FUNCTIONS[node.id]
            if node.id not in values:
                raise ModelError(f"La variable «{node.id}» no está definida en el modelo.")
            return values[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            try:
                return _BINARY[type(node.op)](evaluate(node.left), evaluate(node.right))
            except (FloatingPointError, ZeroDivisionError, OverflowError, ValueError) as exc:
                raise ModelError(f"La fórmula produjo un resultado inválido: {exc}.") from exc
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](evaluate(node.operand))
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _COMPARE:
            return _COMPARE[type(node.ops[0])](evaluate(node.left), evaluate(node.comparators[0]))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCTIONS:
            return _FUNCTIONS[node.func.id](*(evaluate(arg) for arg in node.args))
        raise ModelError("La fórmula contiene una operación no permitida.")

    with np.errstate(all="ignore"):
        result = np.asarray(evaluate(tree), dtype=float)
    if result.ndim == 0:
        first = next(iter(values.values()), None)
        size = len(first) if isinstance(first, (np.ndarray, list, tuple, pd.Series)) else 1
        result = np.full(size, float(result))
    if not np.isfinite(result).all():
        raise ModelError("La fórmula produjo valores no finitos. Revise divisiones, logaritmos y potencias.")
    return result


def _validate_assumption(item: Assumption) -> None:
    if not item.name.isidentifier() or item.name in _FUNCTIONS:
        raise ModelError(f"«{item.name}» debe ser un nombre de variable válido (por ejemplo, ventas o costo_1).")
    if item.distribution == "Normal" and item.p2 <= 0:
        raise ModelError(f"La desviación de «{item.name}» debe ser mayor que cero.")
    if item.distribution == "Uniforme" and item.p2 <= item.p1:
        raise ModelError(f"El máximo de «{item.name}» debe superar al mínimo.")
    if item.distribution in {"Triangular", "PERT"}:
        if item.p1 >= item.p3:
            raise ModelError(f"En «{item.name}», el máximo debe ser mayor que el mínimo.")
        if not item.p1 <= item.p2 <= item.p3:
            raise ModelError(f"En «{item.name}», se requiere mínimo ≤ valor más probable ≤ máximo.")
    if item.distribution == "Lognormal" and item.p2 <= 0:
        raise ModelError(f"La desviación logarítmica de «{item.name}» debe ser mayor que cero.")
    if item.distribution not in {"Normal", "Uniforme", "Triangular", "PERT", "Lognormal"}:
        raise ModelError(f"Distribución no reconocida: {item.distribution}.")


def _inverse_distribution(item: Assumption, u: np.ndarray) -> np.ndarray:
    p = np.clip(u, 1e-10, 1 - 1e-10)
    if item.distribution == "Normal":
        return norm.ppf(p, loc=item.p1, scale=item.p2)
    if item.distribution == "Uniforme":
        return item.p1 + (item.p2 - item.p1) * p
    if item.distribution == "Lognormal":
        return np.exp(norm.ppf(p, loc=item.p1, scale=item.p2))
    low, mode, high = item.p1, item.p2, item.p3
    if item.distribution == "Triangular":
        return low + (high - low) * np.where(
            p < (mode - low) / (high - low),
            np.sqrt(p * (mode - low) / (high - low)),
            1 - np.sqrt((1 - p) * (high - mode) / (high - low)),
        )
    # PERT: beta-PERT con el parámetro de forma clásico lambda = 4.
    alpha = 1 + 4 * (mode - low) / (high - low)
    beta = 1 + 4 * (high - mode) / (high - low)
    from scipy.stats import beta as beta_dist
    return low + (high - low) * beta_dist.ppf(p, alpha, beta)


def run_simulation(
    assumptions: list[Assumption],
    formula: str,
    iterations: int = 20_000,
    seed: int = 42,
    correlation: pd.DataFrame | None = None,
    sampling_method: str = "Monte Carlo",
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    if not assumptions:
        raise ModelError("Agregue al menos un supuesto incierto.")
    if iterations < 100:
        raise ModelError("Use al menos 100 iteraciones.")
    names = [item.name for item in assumptions]
    if len(set(names)) != len(names):
        raise ModelError("Cada variable debe tener un nombre único.")
    for item in assumptions:
        _validate_assumption(item)
    unknown = formula_names(formula) - set(names)
    if unknown:
        raise ModelError("Variables no definidas en la fórmula: " + ", ".join(sorted(unknown)))

    if sampling_method not in {"Monte Carlo", "Latin Hypercube"}:
        raise ModelError("Método de muestreo no reconocido.")
    rng = np.random.default_rng(seed)
    if correlation is None:
        matrix = np.eye(len(names))
    else:
        matrix = correlation.loc[names, names].to_numpy(dtype=float)
    if matrix.shape != (len(names), len(names)) or not np.isfinite(matrix).all():
        raise ModelError("La matriz de correlación tiene dimensiones o valores inválidos.")
    if not np.allclose(matrix, matrix.T, atol=1e-8) or not np.allclose(np.diag(matrix), 1.0, atol=1e-8):
        raise ModelError("La matriz debe ser simétrica y tener 1 en su diagonal.")
    if np.any(np.abs(matrix) > 1.0 + 1e-8):
        raise ModelError("Las correlaciones deben estar entre −1 y 1.")
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    if eigenvalues.min() < -1e-8:
        raise ModelError("La matriz no es semidefinida positiva; ajuste las correlaciones para que sean compatibles.")
    factor = eigenvectors @ np.diag(np.sqrt(np.clip(eigenvalues, 0, None)))
    if sampling_method == "Latin Hypercube":
        from scipy.stats import qmc

        base = qmc.LatinHypercube(d=len(names), seed=seed).random(iterations)
        latent = norm.ppf(np.clip(base, 1e-10, 1 - 1e-10)) @ factor.T
    else:
        latent = rng.standard_normal((iterations, len(names))) @ factor.T
    uniforms = norm.cdf(latent)
    draws = {item.name: _inverse_distribution(item, uniforms[:, i]) for i, item in enumerate(assumptions)}
    outcome = evaluate_formula(formula, draws)
    return draws, outcome


def filter_draws(
    draws: Mapping[str, np.ndarray], outcome: np.ndarray, valid: np.ndarray
) -> tuple[dict[str, np.ndarray], np.ndarray, int]:
    """Apply an iteration-level model constraint and report discarded draws."""
    mask = np.asarray(valid, dtype=bool)
    if mask.shape != outcome.shape:
        raise ModelError("La regla de validez debe producir una condición por iteración.")
    kept = int(mask.sum())
    if kept == 0:
        raise ModelError("Ninguna iteración cumple la regla del modelo; revise los supuestos.")
    return {name: np.asarray(values)[mask] for name, values in draws.items()}, outcome[mask], int(len(mask) - kept)


def convergence_summary(
    outcome: np.ndarray,
    seed: int = 42,
    replicates: int = 500,
    confidence: float = 0.95,
) -> tuple[float, pd.DataFrame]:
    """Return the standard error of the mean and bootstrap CIs for P5/P50/P95."""
    sample = np.asarray(outcome, dtype=float)
    if sample.size < 2:
        raise ModelError("Se necesitan al menos dos iteraciones válidas para medir convergencia.")
    quantiles = np.array([0.05, 0.50, 0.95])
    estimates = np.quantile(sample, quantiles)
    boot = np.empty((replicates, len(quantiles)), dtype=float)
    rng = np.random.default_rng(seed)
    batch_size = max(1, min(32, 2_000_000 // sample.size))
    for start in range(0, replicates, batch_size):
        count = min(batch_size, replicates - start)
        indices = rng.integers(0, sample.size, size=(count, sample.size))
        boot[start : start + count] = np.quantile(sample[indices], quantiles, axis=1).T
    alpha = (1 - confidence) / 2
    lows, highs = np.quantile(boot, [alpha, 1 - alpha], axis=0)
    labels = ["P5", "P50", "P95"]
    intervals = pd.DataFrame({
        "Percentil": labels,
        "Estimación": estimates,
        f"Límite {confidence:.0%} inferior": lows,
        f"Límite {confidence:.0%} superior": highs,
    })
    standard_error = float(np.std(sample, ddof=1) / np.sqrt(sample.size))
    return standard_error, intervals


def sobol_sensitivity(
    assumptions: list[Assumption], formula: str, seed: int = 42, base_sample: int = 512
) -> pd.DataFrame:
    """Estimate first/total-order Sobol indices for independent input distributions."""
    if not assumptions:
        raise ModelError("Agregue al menos un supuesto para calcular Sobol.")
    for item in assumptions:
        _validate_assumption(item)
    names = [item.name for item in assumptions]
    if len(set(names)) != len(names):
        raise ModelError("Cada variable debe tener un nombre único.")
    unknown = formula_names(formula) - set(names)
    if unknown:
        raise ModelError("Variables no definidas en la fórmula: " + ", ".join(sorted(unknown)))
    problem = {"num_vars": len(names), "names": names, "bounds": [[0.0, 1.0]] * len(names)}
    unit_samples = sobol_sample.sample(
        problem, N=base_sample, calc_second_order=False, seed=seed
    )
    draws = {
        item.name: _inverse_distribution(item, unit_samples[:, i])
        for i, item in enumerate(assumptions)
    }
    output = evaluate_formula(formula, draws)
    if np.var(output) <= np.finfo(float).eps:
        raise ModelError("El resultado no varía; no se pueden estimar índices de Sobol.")
    indices = sobol_analyze.analyze(
        problem, output, calc_second_order=False, num_resamples=100, seed=seed
    )
    return pd.DataFrame({
        "Variable": names,
        "Sobol primer orden": indices["S1"],
        "Sobol total": indices["ST"],
        "IC primer orden": indices["S1_conf"],
        "IC total": indices["ST_conf"],
    }).sort_values("Sobol total", ascending=False)


def sensitivity_table(draws: Mapping[str, np.ndarray], outcome: np.ndarray) -> pd.DataFrame:
    rows = []
    target_rank = rankdata(outcome)
    for name, sample in draws.items():
        rho = float(np.corrcoef(rankdata(sample), target_rank)[0, 1])
        if not np.isfinite(rho):
            rho = 0.0
        rows.append({"Variable": name, "Correlación de rangos": rho, "Impacto": abs(rho)})
    return pd.DataFrame(rows).sort_values("Impacto", ascending=True)


def scenario_table(
    assumptions: list[Assumption],
    formula: str,
    draws: Mapping[str, np.ndarray] | None = None,
    outcome: np.ndarray | None = None,
) -> pd.DataFrame:
    probabilities = {"Resultado P10": 0.10, "Resultado P50": 0.50, "Resultado P90": 0.90}
    rows = []
    for label, q in probabilities.items():
        if draws is not None and outcome is not None:
            target = float(np.quantile(outcome, q))
            index = int(np.argmin(np.abs(outcome - target)))
            values = {item.name: float(draws[item.name][index]) for item in assumptions}
            output = float(outcome[index])
        else:
            values = {item.name: float(_inverse_distribution(item, np.array([q]))[0]) for item in assumptions}
            output = float(evaluate_formula(formula, values)[0])
        rows.append({"Escenario": label, **values, "Resultado": output})
    return pd.DataFrame(rows)
