from pathlib import Path
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
from scipy import stats

def normalize_colors(c):
    """Normalize color entries to lowercase string list."""
    if c is None:
        return []
    if isinstance(c, str):
        return [c.strip().lower()]
    try:
        return [str(x).strip().lower() for x in c if str(x).strip()]
    except Exception:
        return [str(c).strip().lower()]


def _variant_sort_key(p: Path):
    """Sort variants in FG/BG order"""
    name = p.name
    m_fg = re.match(r"FG_(\d{3})\.png$", name)
    if m_fg:
        return (0, int(m_fg.group(1)))
    m_bg = re.match(r"BG_(\d{3})\.png$", name)
    if m_bg:
        return (1, int(m_bg.group(1)))
    return (9, name)


def collect_variants_for(image_path: str, target_color: str, out_root: Path) -> list[Path]:
    """
    Collect all FG_*.png and BG_*.png variant images for a given base image and color.
    """
    stem = Path(image_path).stem
    color_dir = out_root / f"{stem}_{target_color}"
    if not color_dir.exists():
        return []

    fg = list(color_dir.glob("FG_*.png"))
    bg = list(color_dir.glob("BG_*.png"))
    paths = fg + bg
    return sorted(paths, key=_variant_sort_key)


def variant_label(p: Path):
    """
    Create readable labels from variant filenames:
    FG_030_seq.png  -> "FG 30% (seq)"
    BG_050_ind.png  -> "BG 50% (ind)"
    base image      -> "white"
    """

    name = p.name
    stem = p.stem  # e.g. "FG_030_seq"

    # Patterns: FG_###_mode, BG_###_mode
    m = re.match(r"(FG|BG)_(\d{3})_(ind|seq)$", stem)
    if m:
        region = m.group(1)
        pct = int(m.group(2))
        mode = m.group(3)
        return f"{region} {pct}% ({mode})"
    # fallback for base image (non-variant)
    return "white"


def show_variants_grid(
    image_path: str,
    target_color: str,
    out_root: Path,
    df_predictions: pd.DataFrame | None = None,
    question: str = "",    # empty, "this" or "most"
    thumb_w: int = 256,
    row_h: float = 3.0,
    fontsize: int = 14,
    color_mode : str | None = None,
    pct_range: list[int] | None = None
):
    """
    Display a 2-row grid of FG/BG color variants for a given image.

    If df_predictions is provided, also show predicted colors (pred_color_this / pred_color_most)
    beneath each variant image.

    Parameters
    ----------
    image_path : str or Path
        Base image path (used to find variant subfolder).
    target_color : str
        Target recolor name.
    out_root : Path
        Root directory containing the generated variants.
    df_predictions : pd.DataFrame, optional
        DataFrame with predicted colors. Must include columns:
          - image_variant
          - correct_answer
          - pred_color_this / pred_color_most
    question : str
        "this" or "most" - determines which prediction column to use.
    thumb_w : int
        Thumbnail width per image.
    row_h : float
        Row height scaling factor.
    fontsize : int
        Font size for titles and predictions.
    color_mode : str
        "independent" or "sequential" - indicates which variant type to display.
    pct_range: list
        Custom percentage ranges to display if provided.
    """

    # Load variant paths
    paths = collect_variants_for(image_path, target_color, out_root)
    if not paths:
        print(f"No variants found for {image_path}")
        return

    if color_mode is None:
        mode_suffix = ""
    elif color_mode == "independent":
        mode_suffix = "ind"
    elif color_mode == "sequential":
        mode_suffix = "seq"

    fg_paths = [p for p in paths if p.name.startswith("FG_") and p.name.endswith(f"{mode_suffix}.png")]
    bg_paths = [p for p in paths if p.name.startswith("BG_") and p.name.endswith(f"{mode_suffix}.png")]

    def parse_pct(p):
        return int(p.name.split("_")[1])

    if pct_range is not None:
        fg_paths = [p for p in fg_paths if parse_pct(p) in pct_range]
        bg_paths = [p for p in bg_paths if parse_pct(p) in pct_range]

    fg_paths = sorted(fg_paths, key=parse_pct)
    bg_paths = sorted(bg_paths, key=parse_pct)[1:]

    cols = max(len(fg_paths), len(bg_paths))
    rows = 2

    # Figure sizing
    fig_w = cols * (thumb_w / 80)
    fig_h = rows * row_h
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))

    # Ensure consistent axis shape
    if rows == 1:
        axes = [axes]
    if cols == 1:
        axes = [[axes[0]], [axes[1]]]

    # Helper to get model prediction (if df_predictions is provided)
    def get_prediction(label):
        if df_predictions is None:
            return None

        pred_col = "pred_color_this" if question.lower() == "this" else "pred_color_most"
        match = df_predictions[
            (df_predictions["image_variant"].str.lower() == label.lower()) &
            (df_predictions["correct_answer"].str.lower() == target_color.lower())
        ]

        if not match.empty and pred_col in match.columns:
            return str(match.iloc[0][pred_col])
        return None

    def _draw_row(ax_row, paths_row, title_prefix, start_col=0):
        for c in range(cols):
            ax = ax_row[c]
            ax.axis("off")
            idx = c - start_col
            if 0 <= idx < len(paths_row):
                p = paths_row[idx]
                im = Image.open(p).convert("RGB")
                ax.imshow(im)

                label = variant_label(p)
                pred = get_prediction(label)
                label = label.split("(")[0]

                ax.set_title(label, fontsize=fontsize, pad=4)
                if pred:
                    ax.text(
                        0.5, -0.15, pred,
                        transform=ax.transAxes,
                        ha="center", va="top",
                        fontsize=fontsize,
                        fontweight="bold"
                    )

    _draw_row(axes[0], fg_paths, "FG", start_col=0)
    _draw_row(axes[1], bg_paths, "BG", start_col=1)
    
    if not question == "":
        question = f", question: {question}"

    fig.suptitle(
        f"{Path(image_path).name} - target: {target_color}, color_mode: {color_mode}{question}",
        fontsize=fontsize + 4,
        fontweight="bold"
    )

    plt.subplots_adjust(top=0.85, bottom=0.1, hspace=0.4, wspace=0.05)
    plt.show()


def plot_vlm_prolific(
    df: pd.DataFrame,
    show_accuracy: bool = True,
    show_probability: bool = True,
    ci: bool = False,
):
    """
    Plot VLM accuracy / probability vs recoloring percentage
    using merged Prolific-style stimulus metadata.

    Required columns:
      - variant_region (FG / BG)
      - percent_colored (int)
      - pred_color_this
      - target_color
      - prob_correct_this (optional; GPT-safe)
    """

    df = df.copy()

    # Normalize text columns
    for col in ["pred_color_this", "target_color"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower()

    # Accuracy computation (unified)
    df["acc_this"] = (
        df["pred_color_this"] == df["target_color"]
    ).astype(float)

    # Aggregation
    grouped = df.groupby(["variant_region", "percent_colored"])

    def summarize(col):
        mean = grouped[col].mean()
        std = grouped[col].std()
        n = grouped[col].count()

        if ci:
            err = stats.t.ppf(0.975, n - 1) * (std / np.sqrt(n))
        else:
            err = std

        return mean, err

    metrics = []

    if show_accuracy:
        metrics.append("acc_this")

    if show_probability and "prob_correct_this" in df.columns:
        metrics.append("prob_correct_this")
    else:
        show_probability = False  # GPT-safe fallback

    summary = {}

    for col in metrics:
        mean, err = summarize(col)
        summary[f"{col}_mean"] = mean
        summary[f"{col}_err"] = err

    summary_df = pd.DataFrame(summary).reset_index()

    # Plot
    fig, ax = plt.subplots(figsize=(9, 6))

    colors = {
        ("acc_this", "FG"): "#1f77b4",
        ("acc_this", "BG"): "#ff7f0e",
        ("prob_correct_this", "FG"): "#2ca02c",
        ("prob_correct_this", "BG"): "#d62728",
    }

    for region in ["FG", "BG"]:
        sub = summary_df[summary_df["variant_region"] == region]
        if sub.empty:
            continue

        for col in metrics:
            mean_col = f"{col}_mean"
            err_col = f"{col}_err"

            ax.errorbar(
                sub["percent_colored"],
                sub[mean_col],
                yerr=sub[err_col],
                fmt="o-" if "prob" in col else "o--",
                color=colors[(col, region)],
                label=f"{col.replace('_this','')} {region}",
                capsize=3 if ci else 0,
            )

    ax.set_xlabel("Colored pixel percentage (%)")
    ax.set_ylabel("Accuracy / P(correct)" if show_probability else "Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("VLM performance vs recoloring fraction (sequential)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

    plt.tight_layout()
    plt.show()



def plot_vlm_performance(
    df,
    show_accuracy=True,
    show_probability=True,
    ci=False,
    color_mode="seq",
    counterfact=False,
    pct_range: list[int] | None = None
):

    # Detect available columns
    has_this = "pred_color_this" in df.columns
    has_most = "pred_color_most" in df.columns

    if not has_this and not has_most:
        raise ValueError("DataFrame has neither 'this' nor 'most' prediction columns.")

    # Parse variant strings
    def parse_variant(variant):
        m = re.match(r"(FG|BG)\s*(\d+)%(?:\s*\((seq|ind)\))?", str(variant))
        if not m:
            return None, None, None
        return m.group(1), int(m.group(2)), m.group(3)

    df[["region", "pct", "mode"]] = df["image_variant"].apply(
        lambda v: pd.Series(parse_variant(v))
    )
    df = df.dropna(subset=["region", "pct"])

    # Filter by percentage range
    if pct_range is not None:
        df = df[df["pct"].isin(set(pct_range))]

    # Filter by mode (seq/ind/all)
    if color_mode not in ["all", "both"]:
        df = df[df["mode"] == color_mode]

    # Accuracy columns
    if has_this:
        df["acc_this"] = np.nan

        target_col = "incorrect_answer" if counterfact else "correct_answer" 

        # FG: must match target color
        fg_mask = df["region"] == "FG"
        df.loc[fg_mask, "acc_this"] = (
            df.loc[fg_mask, "pred_color_this"].str.lower()
            == df.loc[fg_mask, target_col].str.lower()
        ).astype(float)

        # BG: must be white
        bg_mask = df["region"] == "BG"
        df.loc[bg_mask, "acc_this"] = (
            df.loc[bg_mask, "pred_color_this"].str.lower() == "white"
        ).astype(float)


    if has_most:
        df["acc_most"] = np.nan

        # FG: must match target color
        fg_mask = df["region"] == "FG"
        df.loc[fg_mask, "acc_most"] = (
            df.loc[fg_mask, "pred_color_most"].str.lower()
            == df.loc[fg_mask, "correct_answer"].str.lower()
        ).astype(float)

        # BG: must be white
        bg_mask = df["region"] == "BG"
        df.loc[bg_mask, "acc_most"] = (
            df.loc[bg_mask, "pred_color_most"].str.lower()
            == df.loc[bg_mask, "correct_answer"].str.lower()
        ).astype(float)


    grouped = df.groupby(["region", "pct"])

    def summarize(col):
        mean = grouped[col].mean()
        std = grouped[col].std()
        n = grouped[col].count()
        if ci:
            ci_val = stats.t.ppf(0.975, n - 1) * (std / np.sqrt(n))
            return mean, ci_val
        else:
            return mean, std

    # Collect metrics
    data = {}
    metrics = []
    if show_accuracy:
        if has_this: metrics.append("acc_this")
        if has_most: metrics.append("acc_most")
    if show_probability:
        if has_this and "prob_correct_this" in df.columns:
            metrics.append("prob_correct_this")
        if has_most and "prob_correct_most" in df.columns:
            metrics.append("prob_correct_most")

    for metric in metrics:
        mean, err = summarize(metric)
        data[f"{metric}_mean"] = mean
        data[f"{metric}_err"] = err

    summary = pd.DataFrame(data).reset_index()

    
    fig, ax = plt.subplots(figsize=(9, 6))

    colors = {}
    if has_this:
        colors["this_FG"] = "#1f77b4"
        colors["this_BG"] = "#ff7f0e"
    if has_most:
        colors["most_FG"] = "#2ca02c"
        colors["most_BG"] = "#d62728"

    def plot_metric(region, acc_col, prob_col, label_prefix):
        if region not in ["FG", "BG"]:
            return
        if f"{acc_col}_mean" not in summary.columns and f"{prob_col}_mean" not in summary.columns:
            return

        sub = summary[summary["region"] == region]
        if sub.empty:
            return

        c = colors.get(f"{label_prefix}_{region}", "black")

        # Accuracy (dashed)
        if show_accuracy and f"{acc_col}_mean" in sub.columns:
            ax.errorbar(
                sub["pct"], sub[f"{acc_col}_mean"],
                yerr=sub.get(f"{acc_col}_err", None),
                fmt="o--", color=c, capsize=3 if ci else 0,
                label=f"{label_prefix}_{region} (acc)"
            )

        # Probability (solid)
        if show_probability and f"{prob_col}_mean" in sub.columns:
            ax.errorbar(
                sub["pct"], sub[f"{prob_col}_mean"],
                yerr=sub.get(f"{prob_col}_err", None),
                fmt="o-", color=c, capsize=3 if ci else 0,
                label=f"{label_prefix}_{region} (P)"
            )

  
    if has_this:
        for region in ["FG", "BG"]:
            plot_metric(region, "acc_this", "prob_correct_this", "this")

    if has_most:
        for region in ["FG", "BG"]:
            plot_metric(region, "acc_most", "prob_correct_most", "most")

    # Titles, labels, legend
    ax.set_xlabel("Colored pixel percentage (%)", fontsize=12)

    if show_accuracy and show_probability:
        ylabel = "Accuracy / P(correct)"
    elif show_accuracy:
        ylabel = "Accuracy"
    else:
        ylabel = "P(correct)"

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"VLM performance vs. recoloring fraction ({color_mode})", fontsize=14)

    if pct_range is not None:
        ax.set_xticks(sorted(set(pct_range)))

    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

    plt.tight_layout()
    plt.show()


def summarize_response_frequencies(
    df: pd.DataFrame,
    *,
    variant_region: str,
    response_col: str,
    target_col: str,
    percent_colored_col: str,
    variant_region_col: str = "variant_region",
    ci: float = 0.95,
):
    df_region = df[df[variant_region_col] == variant_region].copy()

    if df_region.empty:
        raise ValueError(
            f"No data found for {variant_region_col} == '{variant_region}'"
        )

    df_region["is_white"] = df_region[response_col] == "white"
    df_region["is_target"] = (
        df_region[response_col] == df_region[target_col]
    )
    if variant_region == "FG":
        # FG: target color vs white
        df_region["is_target"] = (
            df_region[response_col] == df_region[target_col]
        )

    elif variant_region == "BG":
        # BG: any non-white response is an error
        df_region["is_target"] = ~df_region["is_white"]

    grouped = df_region.groupby(percent_colored_col)

    summary = grouped.agg(
        p_white=("is_white", "mean"),
        p_target=("is_target", "mean"),
        n=("is_white", "count"),
        std_white=("is_white", "std"),
        std_target=("is_target", "std"),
    ).reset_index()

    alpha = 1 - ci
    tval = stats.t.ppf(1 - alpha / 2, summary["n"] - 1)

    summary["ci_white"] = tval * summary["std_white"] / np.sqrt(summary["n"])
    summary["ci_target"] = tval * summary["std_target"] / np.sqrt(summary["n"])

    return summary.sort_values(percent_colored_col)


def plot_fg_response_frequencies_line(
    df: pd.DataFrame,
    *,
    title: str,
    variant_region: str = "FG",
    response_col: str = "response_label",
    target_col: str = "target_color",
    percent_colored_col: str = "percent_colored",
    variant_region_col: str = "variant_region",
    ci: float = 0.95,
    figsize=(9, 6),
):
    summary = summarize_response_frequencies(
        df,
        variant_region=variant_region,
        response_col=response_col,
        target_col=target_col,
        percent_colored_col=percent_colored_col,
        variant_region_col=variant_region_col,
        ci=ci,
    )
    summary = summary[summary["percent_colored"] > 0]

    fig, ax = plt.subplots(figsize=figsize)

    ax.errorbar(
        summary[percent_colored_col],
        summary["p_white"],
        yerr=summary["ci_white"],
        fmt="o--",
        label="White response",
        color="#ff7f0e",
        capsize=3,
    )

    ax.errorbar(
        summary[percent_colored_col],
        summary["p_target"],
        yerr=summary["ci_target"],
        fmt="o-",
        label="Target-color response",
        color="#1f77b4",
        capsize=3,
    )

    ax.set_xlabel("Colored pixel percentage (%)")
    ax.set_ylabel("Response frequency")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    plt.tight_layout()
    plt.show()

    #return summary



def plot_fg_response_frequencies_bar(
    df: pd.DataFrame,
    *,
    title: str,
    variant_region: str = "FG",
    response_col: str = "response_label",
    target_col: str = "target_color",
    percent_colored_col: str = "percent_colored",
    variant_region_col: str = "variant_region",
    ci: float = 0.95,
    figsize=(10, 6),
    bar_width: float = 0.35,
):
    summary = summarize_response_frequencies(
        df,
        variant_region=variant_region,
        response_col=response_col,
        target_col=target_col,
        percent_colored_col=percent_colored_col,
        variant_region_col=variant_region_col,
        ci=ci,
    )

    summary = summary[summary["percent_colored"] > 0]
    x = np.arange(len(summary))

    fig, ax = plt.subplots(figsize=figsize)

    ax.bar(
        x - bar_width / 2,
        summary["p_white"],
        bar_width,
        yerr=summary["ci_white"],
        label="White response",
        color="#ff7f0e",
        capsize=3,
    )

    ax.bar(
        x + bar_width / 2,
        summary["p_target"],
        bar_width,
        yerr=summary["ci_target"],
        label="Target-color response",
        color="#1f77b4",
        capsize=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(summary[percent_colored_col])
    ax.set_xlabel("Colored pixel percentage (%)")
    ax.set_ylabel("Response frequency")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()

    #return summary


def plot_bg_response_frequencies_line(
    df: pd.DataFrame,
    *,
    title: str,
    response_col: str,
    target_col: str = "target_color",
    percent_colored_col: str = "percent_colored",
    variant_region: str = "BG",
    variant_region_col: str = "variant_region",
    ci: float = 0.95,
    figsize=(9, 6),
):
    summary = summarize_response_frequencies(
        df,
        variant_region=variant_region,
        response_col=response_col,
        target_col=target_col,
        percent_colored_col=percent_colored_col,
        variant_region_col=variant_region_col,
        ci=ci,
    )

    fig, ax = plt.subplots(figsize=figsize)

    ax.errorbar(
        summary[percent_colored_col],
        summary["p_white"],
        yerr=summary["ci_white"],
        fmt="o-",
        label="White response (correct)",
        color="#2ca02c",
        capsize=3,
    )

    ax.errorbar(
        summary[percent_colored_col],
        summary["p_target"],
        yerr=summary["ci_target"],
        fmt="o--",
        label="Any color response (error)",
        color="#d62728",
        capsize=3,
    )

    ax.set_xlabel("Colored pixel percentage (%)")
    ax.set_ylabel("Response frequency")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    plt.tight_layout()
    plt.show()

    #return summary

def plot_bg_response_frequencies_bar(
    df: pd.DataFrame,
    *,
    title: str,
    response_col: str,
    target_col: str = "target_color",
    percent_colored_col: str = "percent_colored",
    variant_region: str = "BG",
    variant_region_col: str = "variant_region",
    ci: float = 0.95,
    figsize=(10, 6),
    bar_width: float = 0.35,
):
    summary = summarize_response_frequencies(
        df,
        variant_region=variant_region,
        response_col=response_col,
        target_col=target_col,
        percent_colored_col=percent_colored_col,
        variant_region_col=variant_region_col,
        ci=ci,
    )

    x = np.arange(len(summary))

    fig, ax = plt.subplots(figsize=figsize)

    ax.bar(
        x - bar_width / 2,
        summary["p_white"],
        bar_width,
        yerr=summary["ci_white"],
        label="White response (correct)",
        color="#2ca02c",
        capsize=3,
    )

    ax.bar(
        x + bar_width / 2,
        summary["p_target"],
        bar_width,
        yerr=summary["ci_target"],
        label="Any color response (error)",
        color="#d62728",
        capsize=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(summary[percent_colored_col])
    ax.set_xlabel("Colored pixel percentage (%)")
    ax.set_ylabel("Response frequency")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()

    #return summary


def summarize_certainty(
    df: pd.DataFrame,
    *,
    value_col: str,
    variant_region_col: str,
    percent_colored_col: str,
    ci: float = 0.95,
):
    required = {value_col, variant_region_col, percent_colored_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for certainty summary: {missing}")

    df_valid = df.dropna(subset=[value_col])
    if df_valid.empty:
        return None  # ← explicitly signal "nothing to plot"

    grouped = df_valid.groupby([variant_region_col, percent_colored_col])

    summary = grouped[value_col].agg(
        mean="mean",
        std="std",
        n="count",
    ).reset_index()

    if summary.empty:
        return None

    alpha = 1 - ci
    tval = stats.t.ppf(1 - alpha / 2, summary["n"] - 1)
    summary["ci"] = tval * (summary["std"] / np.sqrt(summary["n"]))
    summary["certainty_type"] = value_col

    return summary


    #return summary


def plot_certainty(
    df: pd.DataFrame,
    *,
    certainty_cols: list[str],
    variant_region_col: str = "variant_region",
    percent_colored_col: str = "percent_colored",
    title: str = "Certainty vs recoloring fraction",
    y_label: str = "Mean certainty",
    y_lim: tuple[float, float] = (0, 10.5),
    ci: float = 0.95,
    colors: dict | None = None,
    linestyles: dict | None = None,
    labels: dict | None = None,
):
    """
    Plot certainty (with CI) vs percent colored, split by FG/BG and certainty type.
    """

    if colors is None:
        colors = {
            "FG": "#1f77b4",
            "BG": "#ff7f0e",
        }

    if linestyles is None:
        linestyles = {
            certainty_cols[0]: "--",
            certainty_cols[-1]: "-",
        }

    summaries = []
    for col in certainty_cols:
        summaries.append(
            summarize_certainty(
                df,
                value_col=col,
                variant_region_col=variant_region_col,
                percent_colored_col=percent_colored_col,
                ci=ci,
            )
        )

    cert_summary = pd.concat(summaries, ignore_index=True)

    fig, ax = plt.subplots(figsize=(9, 6))

    for region in cert_summary[variant_region_col].unique():
        for cert_type in certainty_cols:
            sub = cert_summary[
                (cert_summary[variant_region_col] == region)
                & (cert_summary["certainty_type"] == cert_type)
            ]

            if sub.empty:
                continue

            label = (
                labels[(region, cert_type)]
                if labels and (region, cert_type) in labels
                else f"{region} – {cert_type.replace('_', ' ')}"
            )

            ax.errorbar(
                sub[percent_colored_col],
                sub["mean"],
                yerr=sub["ci"],
                fmt="o",
                linestyle=linestyles.get(cert_type, "-"),
                color=colors.get(region, "black"),
                capsize=3,
                label=label,
            )

    ax.set_xlabel("Colored pixel percentage (%)")
    ax.set_ylabel(y_label)
    ax.set_ylim(*y_lim)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    plt.tight_layout()
    plt.show()


def summarize_model_confidence(
    df: pd.DataFrame,
    *,
    prob_col: str = "prob_correct_this",
    variant_region_col: str = "variant_region",
    percent_colored_col: str = "percent_colored",
    ci: float = 0.95,
):
    required = {prob_col, variant_region_col, percent_colored_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for model confidence summary: {missing}")

    df_valid = df.dropna(subset=[prob_col])
    if df_valid.empty:
        raise ValueError("No non-NaN model confidence values to plot.")

    grouped = df_valid.groupby([variant_region_col, percent_colored_col])

    summary = grouped[prob_col].agg(
        mean="mean",
        std="std",
        n="count",
    ).reset_index()

    alpha = 1 - ci
    tval = stats.t.ppf(1 - alpha / 2, summary["n"] - 1)

    summary["ci"] = tval * (summary["std"] / np.sqrt(summary["n"]))

    return summary.sort_values(percent_colored_col)


def plot_model_confidence(
    df: pd.DataFrame,
    *,
    prob_col: str = "prob_correct",
    variant_region_col: str = "variant_region",
    percent_colored_col: str = "percent_colored",
    title: str,
    y_label: str = "Mean model confidence (P(correct))",
    y_lim: tuple[float, float] = (0.0, 1.05),
    ci: float = 0.95,
    colors: dict | None = None,
):
    """
    Plot VLM softmax confidence vs recoloring fraction (FG vs BG).
    """

    if colors is None:
        colors = {
            "FG": "#1f77b4",
            "BG": "#ff7f0e",
        }

    summary = summarize_model_confidence(
        df,
        prob_col=prob_col,
        variant_region_col=variant_region_col,
        percent_colored_col=percent_colored_col,
        ci=ci,
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    for region in summary[variant_region_col].unique():
        sub = summary[summary[variant_region_col] == region]

        ax.errorbar(
            sub[percent_colored_col],
            sub["mean"],
            yerr=sub["ci"],
            fmt="o-",
            capsize=3,
            color=colors.get(region, "black"),
            label=f"{region} trials",
        )

    ax.set_xlabel("Colored pixel percentage (%)")
    ax.set_ylabel(y_label)
    ax.set_ylim(*y_lim)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_model_logprobs(
    df: pd.DataFrame,
    *,
    logprob_correct_col: str = "logprob_correct_token",
    logprob_pred_col: str = "logprob_pred_token",
    variant_region_col: str = "variant_region",
    percent_colored_col: str = "percent_colored",
    title: str,
    ci: float = 0.95,
    colors: dict | None = None,
):
    """
    Plot mean log-probabilities vs recoloring fraction (FG vs BG).

    Plots:
    - logprob_correct_token (solid lines)
    - logprob_pred_token (dashed lines)
    """

    if colors is None:
        colors = {
            "FG": "#1f77b4",
            "BG": "#ff7f0e",
        }

    df = df.copy()
    df = df[df[logprob_correct_col].notna()]

    fig, ax = plt.subplots(figsize=(9, 6))

    for region in df[variant_region_col].unique():
        sub = df[df[variant_region_col] == region]

        # --- Correct token logprob ---
        summary_correct = summarize_model_confidence(
            sub,
            prob_col=logprob_correct_col,
            variant_region_col=variant_region_col,
            percent_colored_col=percent_colored_col,
            ci=ci,
        )

        ax.errorbar(
            summary_correct[percent_colored_col],
            summary_correct["mean"],
            yerr=summary_correct["ci"],
            fmt="o-",
            capsize=3,
            color=colors.get(region, "black"),
            label=f"{region} - {logprob_correct_col}",
        )

        # --- Predicted token logprob ---
        summary_pred = summarize_model_confidence(
            sub,
            prob_col=logprob_pred_col,
            variant_region_col=variant_region_col,
            percent_colored_col=percent_colored_col,
            ci=ci,
        )

        ax.errorbar(
            summary_pred[percent_colored_col],
            summary_pred["mean"],
            yerr=summary_pred["ci"],
            fmt="o--",
            capsize=3,
            color=colors.get(region, "black"),
            label=f"{region} - {logprob_pred_col}",
        )

    ax.set_xlabel("Colored pixel percentage (%)")
    ax.set_ylabel("Mean log P(token)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    plt.tight_layout()
    plt.show()


