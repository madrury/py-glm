from collections.abc import Iterator

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes

from glm.families import Poisson
from glm.plotting import plot_coefficient_paths
from glm.ridge import RidgePath

N = 500
N_ALPHAS = 20


@pytest.fixture
def df() -> pd.DataFrame:
    rng = np.random.default_rng(154)
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=N),
            "x2": rng.normal(size=N),
            "g": rng.choice(["a", "b", "c"], N),
        }
    )
    g_effect = df["g"].map({"a": 0.0, "b": 0.5, "c": -0.3})
    df["y"] = rng.poisson(np.exp(0.3 + 0.4 * df["x1"] - 0.2 * df["x2"] + g_effect))
    return df


@pytest.fixture
def path(df: pd.DataFrame) -> RidgePath:
    return RidgePath(Poisson(), n_alphas=N_ALPHAS).fit(
        df, formula="y ~ x1 + x2 + C(g) + x1:C(g)"
    )


@pytest.fixture
def ax() -> Iterator[Axes]:
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)


def path_lines(ax: Axes) -> list:
    """The coefficient path lines, leaving out the line at zero."""
    return [line for line in ax.get_lines() if len(line.get_xdata()) == N_ALPHAS]


def legend_labels(ax: Axes) -> list[str]:
    legend = ax.get_legend()
    assert legend is not None
    return [text.get_text() for text in legend.get_texts()]


def colors_by_label(ax: Axes) -> dict[str, set[str]]:
    """The colors of each legend entry's lines."""
    colors: dict[str, set[str]] = {}
    label = ""
    for line in path_lines(ax):
        if not line.get_label().startswith("_"):
            label = line.get_label()
        colors.setdefault(label, set()).add(line.get_color())
    return colors


def test_returns_axes_and_creates_no_figures(path: RidgePath, ax: Axes) -> None:
    figures = plt.get_fignums()
    assert plot_coefficient_paths(ax, path) is ax
    assert plt.get_fignums() == figures


def test_plots_every_penalized_coefficient_by_term(path: RidgePath, ax: Axes) -> None:
    plot_coefficient_paths(ax, path)
    # x1, x2, three levels of C(g), and three of x1:C(g); no intercept.
    assert len(path_lines(ax)) == 8
    assert legend_labels(ax) == ["x1", "x2", "C(g)", "x1:C(g)"]
    colors = colors_by_label(ax)
    # Each term's lines share one color, and terms have different colors.
    assert all(len(term_colors) == 1 for term_colors in colors.values())
    assert len(set.union(*colors.values())) == 4


def test_select_terms_with_a_formula(path: RidgePath, ax: Axes) -> None:
    plot_coefficient_paths(ax, path, "~ C(g) + x1")
    # Legend entries follow the model's order, not the formula's.
    assert legend_labels(ax) == ["x1", "C(g)"]
    assert len(path_lines(ax)) == 4


def test_formula_selection_expands_and_ignores_factor_order(
    path: RidgePath, ax: Axes
) -> None:
    plot_coefficient_paths(ax, path, "C(g):x1")
    assert legend_labels(ax) == ["x1:C(g)"]
    fig, other = plt.subplots()
    plot_coefficient_paths(other, path, "x1 * C(g)")
    assert legend_labels(other) == ["x1", "C(g)", "x1:C(g)"]
    plt.close(fig)


def test_term_colors_are_stable_across_plots(path: RidgePath, ax: Axes) -> None:
    plot_coefficient_paths(ax, path)
    fig, selected = plt.subplots()
    plot_coefficient_paths(selected, path, "C(g)")
    assert colors_by_label(selected)["C(g)"] == colors_by_label(ax)["C(g)"]
    plt.close(fig)


def test_unknown_terms_raise(path: RidgePath, ax: Axes) -> None:
    with pytest.raises(ValueError, match=r"Terms \['x3'\] are not in the model"):
        plot_coefficient_paths(ax, path, "x1 + x3")


def test_groups(path: RidgePath, ax: Axes) -> None:
    plot_coefficient_paths(ax, path, groups={"slopes": ["x1", "x2"]})
    assert legend_labels(ax) == ["slopes"]
    assert len(path_lines(ax)) == 2


def test_terms_and_groups_together_raise(path: RidgePath, ax: Axes) -> None:
    with pytest.raises(ValueError, match="either terms or groups"):
        plot_coefficient_paths(ax, path, "x1", groups={"slopes": ["x1"]})


def test_horizontal_axes(path: RidgePath, ax: Axes) -> None:
    plot_coefficient_paths(ax, path, "x1")
    np.testing.assert_array_equal(path_lines(ax)[0].get_xdata(), path.alphas_)
    assert ax.get_xscale() == "log"
    fig, by_df = plt.subplots()
    plot_coefficient_paths(by_df, path, "x1", x="df")
    np.testing.assert_array_equal(path_lines(by_df)[0].get_xdata(), path.df_)
    assert by_df.get_xscale() == "linear"
    plt.close(fig)


def test_coefficient_scales(df: pd.DataFrame, ax: Axes) -> None:
    rescaled = df.assign(x1=1000 * df["x1"])
    path = RidgePath(Poisson(), n_alphas=N_ALPHAS).fit(rescaled, formula="y ~ x1")
    assert path.scales_ is not None
    coefficients = path.coefficient_table()["x1"].to_numpy()
    plot_coefficient_paths(ax, path)
    np.testing.assert_allclose(
        path_lines(ax)[0].get_ydata(), coefficients * path.scales_[1]
    )
    fig, original = plt.subplots()
    plot_coefficient_paths(original, path, scale="original")
    np.testing.assert_allclose(path_lines(original)[0].get_ydata(), coefficients)
    plt.close(fig)


def test_array_path_selects_by_name(df: pd.DataFrame, ax: Axes) -> None:
    X = np.column_stack([np.ones(N), df["x1"], df["x2"]])
    path = RidgePath(Poisson(), n_alphas=N_ALPHAS).fit(
        X, df["y"].to_numpy().astype(np.float64), X_names=["Intercept", "a", "b"]
    )
    plot_coefficient_paths(ax, path, "b")
    assert legend_labels(ax) == ["b"]


@pytest.mark.parametrize(
    "kwargs, message",
    [({"x": "alpha"}, "x must be one of"), ({"scale": "raw"}, "scale must be one of")],
)
def test_invalid_options_raise(
    path: RidgePath, ax: Axes, kwargs: dict[str, str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        plot_coefficient_paths(ax, path, **kwargs)  # type: ignore[arg-type]


def test_unfit_path_raises(ax: Axes) -> None:
    with pytest.raises(ValueError, match="has not been fit"):
        plot_coefficient_paths(ax, RidgePath(Poisson()))


def test_select_terms_with_the_models_formula(path: RidgePath, ax: Axes) -> None:
    assert path.models_ is not None and path.models_[0].formula is not None
    plot_coefficient_paths(ax, path, path.models_[0].formula)
    assert legend_labels(ax) == ["x1", "x2", "C(g)", "x1:C(g)"]
