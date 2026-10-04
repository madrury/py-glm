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

## Code style

- **Format with black.** All Python is black-formatted; don't hand-align
  continuation lines.
- **Dataclasses for records.** Use a `dataclass` (frozen, where the record
  shouldn't change) for simple records of named fields, rather than a
  `NamedTuple`.
- **Absolute imports only.** Import within the package as `from glm.utils import
  ...`, never with relative imports (`from .utils import ...`).

## Output

- **No significance stars.** Never mark p-values with stars or other
  significance symbols, in any table, summary, or plot.

## Plotting

- **Axes in, axes out.** Plotting functions take a matplotlib `Axes` as their
  first argument, wherever possible, and always return it. They never create
  figures or axes, so the user keeps control of the layout.
- **No runtime matplotlib import.** Because axes are passed in, the library
  only needs matplotlib for type annotations: import `Axes` under
  `typing.TYPE_CHECKING`.

## Checks

- Type check with `uv run mypy` (strict; configured in `pyproject.toml`).
- Format with `uv run black glm tests doc`.
- Run tests with `uv run pytest`.
