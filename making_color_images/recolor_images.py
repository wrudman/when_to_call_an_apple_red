"""
recolor_image.py
----------------
Module for recoloring segmented outline images into FG/BG color variants.
Supports pixel-wise and patch-wise recoloring.

Usage:
    from recolor_image import generate_variants

    paths = generate_variants(row, target_color="red", out_dir=Path("data/colored"), use_patches=True)
"""

from pathlib import Path
from PIL import Image
import numpy as np
import colorsys
import gc
import re
from tqdm import tqdm
import cv2


def color_remap(pixel, target_color: str, blend: float = 0.75):
    """
    Recolor a single RGB pixel toward the target color.

    Parameters
    ----------
    pixel : tuple(int, int, int)
        Original (R, G, B) pixel values in 0–255.
    target_color : str
        Target color name ("red", "green", etc.).
    blend : float
        Blend strength between 0 (original) and 1 (fully recolored).

    Returns
    -------
    (r, g, b) : tuple(int)
        The recolored pixel.
    """
    target_color = target_color.lower()
    r0, g0, b0 = pixel
    h, s, v = colorsys.rgb_to_hsv(r0/255.0, g0/255.0, b0/255.0)

    # Protect outlines and very dark pixels
    if v < 0.12 or (v < 0.25 and s < 0.25):
        return pixel

    color_hue_map = {
        "red":    0.0,
        "brown":  0.05,
        "pink":   0.9,
        "orange": 0.07,
        "yellow": 0.15,
        "gold":   0.12,
        "green":  0.33,
        "blue":   0.55,
        "purple": 0.78,
        "black":  0.0,
        "grey":   0.0,
        "silver": 0.0,
        "white":  0.0,
    }

    neutral_targets = {"grey", "silver", "black"}

    # Neutral colors
    if target_color in neutral_targets:
        if target_color == "grey":
            new_v = min(max(v, 0.55), 0.75)
            new_s = 0.0
            per_color_blend = 0.85
        elif target_color == "silver":
            new_v = min(max(v, 0.65), 0.9)
            new_s = 0.0
            per_color_blend = 0.85
        elif target_color == "black":
            new_v = min(max(v * 0.2, 0.05), 0.19)
            new_s = 0.0
            per_color_blend = 1.0

        r1, g1, b1 = colorsys.hsv_to_rgb(0.0, new_s, new_v)
        r1, g1, b1 = int(r1 * 255), int(g1 * 255), int(b1 * 255)

        w = per_color_blend
        r = int((1 - w) * r0 + w * r1)
        g = int((1 - w) * g0 + w * g1)
        b = int((1 - w) * b0 + w * b1)
        return r, g, b

    # Colorful targets
    target_hue = color_hue_map.get(target_color, 0.0)
    new_h, new_s, new_v = target_hue, 0.75, min(max(v, 0.8), 0.95)

    # Per-color tweaks
    tweaks = {
        "red":    (0.85, 0.9),
        "brown":  (0.7,  0.55),
        "pink":   (0.5,  1.0),
        "orange": (0.85, 0.95),
        "yellow": (0.65, 1.0),
        "gold":   (0.6,  0.85),
        "green":  (0.75, 0.7),
        "blue":   (0.9,  0.7),
        "purple": (0.7,  0.85),
    }
    if target_color in tweaks:
        new_s, new_v = tweaks[target_color]

    r1, g1, b1 = colorsys.hsv_to_rgb(new_h, new_s, new_v)
    r1, g1, b1 = int(r1 * 255), int(g1 * 255), int(b1 * 255)

    r = int((1 - blend) * r0 + blend * r1)
    g = int((1 - blend) * g0 + blend * g1)
    b = int((1 - blend) * b0 + blend * b1)

    return r, g, b


def recolor_region(
    arr_rgb,
    idx_flat,
    target_total_pct,
    target_color,
    H,
    W,
    colored_mask,
    rng,
    use_patches=False,
    patch_size=16,
):
    """
    Recoloring with:
      - proportional patch contribution
      - random but reproducible patch order (seed)
      - independent & sequential modes
    """

    flat = arr_rgb.reshape(-1, 3)

    total_pixels = len(idx_flat)
    target_total = int(round(target_total_pct / 100.0 * total_pixels))

    already = colored_mask[idx_flat].sum()
    need = target_total - already
    if need <= 0:
        return arr_rgb

    # PIXEL-WISE MODE
    if not use_patches:
        available = idx_flat[~colored_mask[idx_flat]]
        if len(available) == 0:
            return arr_rgb

        chosen = rng.choice(
            available,
            size=min(need, len(available)),
            replace=False,
        )

        for fi in chosen:
            r, g, b = flat[fi]
            flat[fi] = color_remap((int(r), int(g), int(b)), target_color)
            colored_mask[fi] = True

        return arr_rgb

  
    # PATCH-WISE MODE (FIXED)
    rows = idx_flat // W
    cols = idx_flat % W
    patch_rows = rows // patch_size
    patch_cols = cols // patch_size

    patch_ids = np.stack([patch_rows, patch_cols], axis=1)
    patch_ids_unique, inverse = np.unique(
        patch_ids, axis=0, return_inverse=True
    )

    # Precompute patch_index 
    patch_to_pixels = [[] for _ in range(len(patch_ids_unique))]
    for flat_i, p_i in zip(idx_flat, inverse):
        patch_to_pixels[p_i].append(flat_i)

    # Random but reproducible patch order
    order = np.arange(len(patch_ids_unique))
    rng.shuffle(order)

    recolored = 0

    for patch_index in order:

        if recolored >= need:
            break

        patch_pixels = patch_to_pixels[patch_index]

        # Exclude already-colored pixels
        patch_pixels = [
            fi for fi in patch_pixels if not colored_mask[fi]
        ]

        if not patch_pixels:
            continue

        # Recolor ALL FG pixels in this patch
        for fi in patch_pixels:
            r, g, b = flat[fi]
            flat[fi] = color_remap(
                (int(r), int(g), int(b)), target_color
            )
            colored_mask[fi] = True
            recolored += 1

        # Stop after finishing this full patch
        if recolored >= need:
            break

    return arr_rgb


def generate_variants(
    row,
    target_color,
    out_dir: Path,
    rng,
    use_patches=False,
    patch_size=16,
    step_size=10,
    mode="independent",
    pct_schedule=None,
):
    """
    Generate FG/BG recolored variants for one image + one target color.
    """

    # Check mode parameter
    valid_modes = {"independent", "sequential"}
    if isinstance(mode, str):
        mode = mode.lower().strip()
    if mode not in valid_modes:
        raise ValueError(
            f"Invalid mode '{mode}'. Must be one of: {sorted(valid_modes)}"
        )

    img = Image.open(row["image_path"]).convert("RGB")
    W, H = img.size
    base = np.array(img, dtype=np.uint8)

    m = Image.open(row["cv_mask_path"]).convert("L").resize((W, H), Image.BILINEAR)
    mask = (np.array(m, dtype=np.uint8) > 127)

    gray = np.mean(base, axis=2) / 255.0
    outline = gray < 0.12

    # Exclude outline pixels from recolorable sets
    fg_mask_clean = mask & ~outline
    bg_mask_clean = (~mask) & ~outline

    idx_all = np.arange(H * W)
    idx_fg = idx_all[fg_mask_clean.flatten()]
    idx_bg = idx_all[bg_mask_clean.flatten()]

    stem = Path(row["image_path"]).stem
    color_dir = out_dir / f"{stem}_{target_color}"
    color_dir.mkdir(parents=True, exist_ok=True)

    fg_paths, bg_paths = [], []

    # set schedule
    if pct_schedule is None:
        pct_list = list(range(0, 101, step_size))
    else:
        pct_list = pct_schedule

    # Foreground recoloring
    if mode == "sequential":
        arr_seq = base.copy()
        colored_fg = np.zeros(H * W, dtype=bool)
    else:
        colored_fg = None  # not used
        
    for pct in pct_list:

        if mode == "independent":
            arr = base.copy()
            colored_fg = np.zeros(H * W, dtype=bool)
        else:
            arr = arr_seq  # sequential accumulation

        recolor_region(
            arr,
            idx_fg,
            pct,
            target_color,
            H,
            W,
            colored_fg,
            use_patches=use_patches,
            patch_size=patch_size,
            rng=rng
        )

        if mode == "sequential":
            arr_seq = arr

        suffix = "ind" if mode == "independent" else "seq"
        out_path = color_dir / f"FG_{pct:03d}_{suffix}.png"
        Image.fromarray(arr).save(out_path)
        fg_paths.append(out_path)

        # SANITY CHECK
        actual_colored = colored_fg[idx_fg].sum()
        actual_pct = 100 * actual_colored / len(idx_fg)
        #print(f"[FG] target={pct:3d}% → actual={actual_pct:6.2f}%   ({actual_colored}/{len(idx_fg)})")


    # Background recoloring
    if mode == "sequential":
        arr_seq = base.copy()
        colored_bg = np.zeros(H * W, dtype=bool)
    else:
        colored_bg = None

    for pct in pct_list:

        if mode == "independent":
            arr = base.copy()
            colored_bg = np.zeros(H * W, dtype=bool)
        else:
            arr = arr_seq

        recolor_region(
            arr,
            idx_bg,
            pct,
            target_color,
            H,
            W,
            colored_bg,
            use_patches=use_patches,
            patch_size=patch_size,
            rng=rng
        )

        if mode == "sequential":
            arr_seq = arr
        
        out_path = color_dir / f"BG_{pct:03d}_{suffix}.png"
        Image.fromarray(arr).save(out_path)
        bg_paths.append(out_path)

        # SANITY CHECK
        actual_colored = colored_bg[idx_bg].sum()
        actual_pct = 100 * actual_colored / len(idx_bg)
        #print(f"[BG] target={pct:3d}% → actual={actual_pct:6.2f}%   ({actual_colored}/{len(idx_bg)})")


    gc.collect()
    return [str(p) for p in (fg_paths + bg_paths)]


def pad_to_square_pil(img: Image.Image, fill=(255, 255, 255)):
    """Pad a PIL image to a square canvas (centered), no distortion."""
    w, h = img.size
    s = max(w, h)
    canvas = Image.new("RGB", (s, s), fill)
    offset = ((s - w) // 2, (s - h) // 2)
    canvas.paste(img, offset)
    return canvas


def pad_to_square_mask(mask_np: np.ndarray):
    """Pad a binary mask (numpy array) to a square, centered."""
    h, w = mask_np.shape
    s = max(h, w)
    canvas = np.zeros((s, s), dtype=np.uint8)
    off_y = (s - h) // 2
    off_x = (s - w) // 2
    canvas[off_y:off_y+h, off_x:off_x+w] = mask_np
    return canvas



def resize_image_and_mask(
    image_path: str,
    mask_path: str,
    target_size: int = 512,
):
    """
    Resize an image + mask to a square, without distortion.
    Pad image/mask to square
    Resize image (LANCZOS) and mask (NEAREST)
    Returns numpy arrays.
    """
    img = Image.open(image_path).convert("RGB")

    mask = Image.open(mask_path).convert("L")
    mask_np = (np.array(mask) > 0).astype(np.uint8)

    img_sq = pad_to_square_pil(img)
    mask_sq = pad_to_square_mask(mask_np)
    img_resized = img_sq.resize((target_size, target_size), Image.LANCZOS)

    mask_resized = Image.fromarray(mask_sq * 255).resize(
        (target_size, target_size),
        Image.NEAREST
    )
    mask_resized_np = (np.array(mask_resized) > 0)

    return np.array(img_resized), mask_resized_np



def resize_all_images_and_masks(
    df,
    img_out_folder: Path,
    mask_out_folder: Path,
    target_size: int = 512,
    mask_column: str = "cv_mask_path",
    img_column: str = "image_path",
):
    """
    Resize all images + masks in a dataframe.
    Saves resized outputs and updates df with new paths.
    """

    img_out_folder.mkdir(parents=True, exist_ok=True)
    mask_out_folder.mkdir(parents=True, exist_ok=True)

    missing = []
    processed = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Resizing all"):
        img_path = row[img_column]
        mask_path = row[mask_column]

        # Check paths
        if not (Path(img_path).exists() and Path(mask_path).exists()):
            missing.append((img_path, mask_path))
            continue

        # Resize
        img_resized_np, mask_resized_np = resize_image_and_mask(
            img_path,
            mask_path,
            target_size=target_size,
        )

        # Output file paths
        stem = Path(img_path).stem

        img_out = img_out_folder / f"{stem}_resized.png"
        mask_out = mask_out_folder / f"{stem}_mask_resized.png"

        Image.fromarray(img_resized_np).save(str(img_out))
        cv2.imwrite(str(mask_out), (mask_resized_np.astype(np.uint8) * 255))

        df.at[idx, img_column] = str(img_out)
        df.at[idx, mask_column] = str(mask_out)

        processed += 1

    print(f"Resized {processed} images.")
    if missing:
        print("WARNING: skipped because paths missing:")
        for bad in missing:
            print(" -", bad)

    return df
