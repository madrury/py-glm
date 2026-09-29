# py-glm Development Principles

## Type annotations

- **Use domain-specific type aliases.** Don't annotate with bare `npt.NDArray` or
  `pd.DataFrame`; name what the array *means* in the model (e.g. `DesignMatrix`,
  `DesignFrame`, `ResponseVector`, `CoefficientVector`, `LinearPredictor`).
- **Use statistician's terminology.** This is a classical statistical learning
  library: prefer "design matrix", "response", "linear predictor", "coefficients"
  over ML terms like "features", "labels", "target", "weights" (for coefficients).
- **Plain aliases, not `NewType`.** Declare aliases with the `type` statement
  (`type DesignMatrix = npt.NDArray[np.float64]`). They document intent; we
  don't require explicit wrapping at API boundaries.

## Naming and notation

- **Greek letters for internal mathematical state.** Inside numerical code, use
  the textbook symbols as identifiers: `η = X @ β + offset`, `μ = family.inv_link(η)`.
- **ASCII for public APIs.** Public function/method names, parameters, and
  attributes (`fit(offset=...)`, `coef_`, `ExponentialFamily` methods) stay
  ASCII so users never have to type Greek characters.
- **Avoid Greek letters that look like Latin ones.** Prefer standard notation that
  isn't visually ambiguous, e.g. `η` for the linear predictor rather than `ν`
  (which reads as `v`).

## Checks

- Type check with `uv run mypy` (strict; configured in `pyproject.toml`).
- Run tests with `uv run pytest`.
