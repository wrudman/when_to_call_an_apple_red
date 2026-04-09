import re
from pathlib import Path
import pandas as pd


# Variant label helper
def variant_label(p: Path) -> str:
    """
    Create readable labels from variant filenames:
    FG_030_seq.png  -> "FG 30% (seq)"
    BG_050_ind.png  -> "BG 50% (ind)"
    base image      -> "white"
    """
    stem = p.stem
    m = re.match(r"(FG|BG)_(\d{3})_(ind|seq)$", stem)
    if m:
        region, pct, mode = m.groups()
        return f"{region} {int(pct)}% ({mode})"
    return "white"


# Folder-name parser
# Example:
# acoustic_guitar_1_ab533cc0_resized_purple
OBJECT_FOLDER_RE = re.compile(
    r"^(?P<object>.+?)_(?P<instance>\d+)_(?P<hash>[a-f0-9]+)_resized_(?P<color>[a-z]+)$"
)

SHAPE_FOLDER_RE = re.compile(
    r"^(?P<object>[a-z]+)_v\d+_(?P<color>[a-z]+)$"
)

def parse_folder_name(name: str, stimulus_type: str) -> dict:
    if stimulus_type == "shape":
        m = SHAPE_FOLDER_RE.match(name)
    elif stimulus_type in {"correct_prior", "counterfact"}:
        m = OBJECT_FOLDER_RE.match(name)
    else:
        raise ValueError("stimulus_type must be 'object' or 'shape'")

    if not m:
        raise ValueError(
            f"Unrecognized {stimulus_type} folder name: {name}"
        )

    return {
        "object": m.group("object"),
        "manipulation_color": m.group("color"),
    }



# Table builder
VARIANT_RE = re.compile(r"(FG|BG)_(\d{3})_(ind|seq)\.png")

def build_stimulus_table(
    dataset_root: Path,
    stimulus_type: str,
    data_root: Path,
) -> pd.DataFrame:
    """
    Build a stimulus table from a dataset directory.

    Args:
        dataset_root: Path to directory containing object/shape folders
        stimulus_type: "correct_prior", "counterfact", or "shape"

    Returns:
        pd.DataFrame with one row per stimulus image
    """

    if stimulus_type not in {"correct_prior", "counterfact", "shape"}:
        raise ValueError("stimulus_type must be 'correct_prior', 'counterfact', or 'shape'")
    rows = []

    dataset_root = Path(dataset_root)

    for obj_dir in sorted(dataset_root.iterdir()):
        if not obj_dir.is_dir():
            continue

        meta = parse_folder_name(obj_dir.name, stimulus_type)

        for img in sorted(obj_dir.glob("*.png")):
            m = VARIANT_RE.match(img.name)

            region, pct, mode = m.groups()
            percent_colored = int(pct)
            variant_region = region

            manipulation_color = meta["manipulation_color"]

            if percent_colored == 0 or variant_region == "BG":
                target_color = "white"
            else:
                target_color = manipulation_color

            rows.append({
                "object": meta["object"].replace("_", " "),
                "stimulus_type": stimulus_type,

                # experimental condition
                "manipulation_color": manipulation_color,

                # what the participant should report
                "target_color": target_color,

                # manipulation metadata
                "variant_region": variant_region,
                "percent_colored": percent_colored,
                "mode": mode,
                "variant_label": variant_label(img),

                # path for jsPsych
                "image_path": str(img.relative_to(data_root)),
            })

    return pd.DataFrame(rows)
