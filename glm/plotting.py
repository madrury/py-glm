"""Plots of fit models.

Plotting functions take a matplotlib Axes as their first argument, draw on
it, and return it. They never create figures or axes, so the caller controls
the layout. The library only imports matplotlib for type checking.
"""

from typing import TYPE_CHECKING, get_args

import formulaic
import numpy as np

from glm.ridge import RidgePath
from glm.types import CoefficientScale, PathAxis, TermColumns

if TYPE_CHECKING:
    from matplotlib.axes import Axes

PATH_AXES: tuple[str, ...] = get_args(PathAxis.__value__)
COEFFICIENT_SCALES: tuple[str, ...] = get_args(CoefficientScale.__value__)


def plot_coefficient_paths(
    ax: "Axes",
    path: RidgePath,
    terms: str | None = None,
    *,
    groups: TermColumns | None = None,
    x: PathAxis = "log_alpha",
    scale: CoefficientScale = "standardized",
) -> "Axes":
    """Plot each coefficient's path as the penalty varies.

    Coefficients are drawn in groups, one color and legend entry per group:
    by default, the terms of the model, so that every level of a categorical
    variable shares a color. The intercept is never plotted.

    Parameters
    ----------

    ax: matplotlib Axes
        The axes to draw on.

    path: RidgePath
        A fit regularization path.

    terms: str, optional
        A formula selecting the terms to plot, e.g. "x1 + x2 + C(g)" plots the
        paths for x1, x2, and all the individual indicators composing C(g).
        A leading "~" is optional, and ignored, and a full formula, such as the
        model's own, selects its right hand side's terms. Formula syntax
        applies, so "x1 * C(g)" selects x1, C(g), and their interaction, and
        the order of factors in an interaction doesn't matter. By default,
        every term is plotted. For a model fit without a formula, terms are
        matched to the coefficient names, which must then be valid formula
        variable names; otherwise, use groups.

    groups: dict, optional
        An explicit grouping of coefficients to plot, mapping each group's
        name to its coefficients' names, in place of the model's terms.
        Cannot be combined with terms.

    x: "log_alpha" or "df"
        The horizontal axis: the penalty strength, on a log scale, or the
        effective degrees of freedom (see RidgePath.df_).

    scale: "standardized" or "original"
        The scale of the coefficients: standardized, as the penalty acts on
        them, which makes predictors comparable, or the original scale of the
        predictors.

    Returns
    -------

    ax: matplotlib Axes
        The axes drawn on.
    """
    if x not in PATH_AXES:
        raise ValueError(f"x must be one of {PATH_AXES}, got {x!r}.")
    if scale not in COEFFICIENT_SCALES:
        raise ValueError(f"scale must be one of {COEFFICIENT_SCALES}, got {scale!r}.")
    if terms is not None and groups is not None:
        raise ValueError("Pass either terms or groups, not both.")
    if (
        path.alphas_ is None
        or path.df_ is None
        or path.scales_ is None
        or path.term_columns_ is None
    ):
        raise ValueError("The path has not been fit.")

    # Color each group by its position among all the groups, not just the
    # selected ones, so that a term has the same color in every plot.
    if groups is None:
        palette = list(path.term_columns_)
        groups = path.term_columns_
        if terms is not None:
            groups = {name: groups[name] for name in _select_terms(path, terms)}
    else:
        palette = list(groups)

    table = path.coefficient_table()
    if scale == "standardized":
        table = table * path.scales_
    horizontal = path.alphas_ if x == "log_alpha" else path.df_

    for name, columns in groups.items():
        color = f"C{palette.index(name) % 10}"
        for j, column in enumerate(columns):
            ax.plot(
                horizontal,
                table[column].to_numpy(),
                color=color,
                label=name if j == 0 else None,
            )
    ax.axhline(0.0, color="grey", linewidth=0.5)
    if x == "log_alpha":
        ax.set_xscale("log")
        ax.set_xlabel("alpha")
    else:
        ax.set_xlabel("effective degrees of freedom")
    ax.set_ylabel(f"{scale} coefficient")
    ax.legend()
    return ax


def _select_terms(path: RidgePath, terms: str) -> list[str]:
    """The names of the path's terms selected by a formula, in the order the
    model lists them."""
    assert path.term_columns_ is not None and path.models_ is not None
    # from_spec is formulaic's typed constructor: calling the abstract Formula
    # directly also works, but only through a metaclass type checkers can't
    # see. A formula with a left hand side, such as the model's own, parses
    # as a StructuredFormula; its right hand side holds the terms.
    parsed = formulaic.Formula.from_spec(terms)
    requested = (
        parsed.rhs if isinstance(parsed, formulaic.StructuredFormula) else parsed
    )
    spec = path.models_[0].X_spec
    selected: set[str] = set()
    unknown: list[str] = []
    for term in requested:
        if str(term) == "1":
            continue  # The intercept is never plotted.
        name: str | None
        if spec is None:
            # Without a formula, the terms are the coefficients' names.
            name = str(term)
        else:
            # Match formulaic's terms, rather than their names, so that the
            # order of factors in an interaction doesn't matter.
            name = next((str(t) for t in spec.term_slices if t == term), None)
        if name is not None and name in path.term_columns_:
            selected.add(name)
        else:
            unknown.append(str(term))
    if unknown:
        raise ValueError(
            f"Terms {unknown} are not in the model, whose terms are "
            f"{list(path.term_columns_)}."
        )
    return [name for name in path.term_columns_ if name in selected]
