#!/usr/bin/env python
"""Build a small balanced real-vs-AI fine-tuning dataset for the
"real-photo false positive" fix.

REAL class: genuine photographs bundled with scientific Python (scikit-image,
scikit-learn) and OpenCV's sample data (curated to exclude identifiable
people).  AI class: synthetic images produced for this project, expanded with
deterministic, realistic augmentation (flip / rotation / JPEG / blur /
brightness-contrast) to balance the class counts.

This is a *quick adaptation set*, not a replacement for full CIFAKE+Defactify
training (which requires the public datasets and a GPU). It is used to retrain
the classification head so that out-of-distribution real photographs stop
being labelled AI-generated.

Outputs:
    data/processed/finetune_train.csv
    data/processed/finetune_val.csv     (held-out, never used in training)
"""

from __future__ import annotations

import io
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

REPO = Path(__file__).resolve().parent.parent
REAL_DIR = REPO / "data" / "real_photos"
AI_DIRS = [REPO / "data" / "ai_synthetic", REPO / "demo" / "ai_generated"]
OUT_DIR = REPO / "data" / "processed"
SEED = 42

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

# Genuine photographs in OpenCV's samples/data (curated by hand to exclude
# identifiable people, e.g. lena.jpg / messi5.jpg are deliberately omitted).
OPENCV_REAL = [
    'aero1.jpg','aero3.jpg','aloeL.jpg','aloeR.jpg','apple.jpg','baboon.jpg',
    'basketball1.png','basketball2.png','blox.jpg','board.jpg','box.png',
    'box_in_scene.png','building.jpg','butterfly.jpg','chicky_512.png',
    'ela_original.jpg','fruits.jpg','gauge-1.jpg','graf1.png','graf3.png',
    'HappyFish.jpg','home.jpg','left01.jpg','left02.jpg','left03.jpg','left04.jpg',
    'left05.jpg','left06.jpg','left07.jpg','left08.jpg','left09.jpg','left11.jpg',
    'left12.jpg','left13.jpg','left14.jpg','leuvenA.jpg','leuvenB.jpg',
    'licenseplate_motion.jpg','orange.jpg','pic1.png','pic2.png','pic3.png',
    'pic4.png','pic5.png','pic6.png','right01.jpg','right02.jpg','right03.jpg',
    'right04.jpg','right05.jpg','right06.jpg','right07.jpg','right08.jpg',
    'right09.jpg','right11.jpg','right12.jpg','right13.jpg','right14.jpg',
    'rubberwhale1.png','rubberwhale2.png','squirrel_cls.jpg','starry_night.jpg',
    'stuff.jpg','text_defocus.jpg','text_motion.jpg',
]


def _download_opencv_samples() -> None:
    """Populate data/real_photos/opencv from OpenCV's GitHub repo (sparse
    checkout of samples/data only). Skipped if the directory already exists."""
    dst = REAL_DIR / "opencv"
    if dst.exists() and any(dst.iterdir()):
        return
    dst.mkdir(parents=True, exist_ok=True)
    tmp = REPO / "data" / ".opencv-samples-tmp"
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
             "https://github.com/opencv/opencv.git", str(tmp)],
            check=True, capture_output=True,
        )
        subprocess.run(["git", "-C", str(tmp), "sparse-checkout", "set", "samples/data"],
                       check=True, capture_output=True)
        for name in OPENCV_REAL:
            src = tmp / "samples" / "data" / name
            if src.exists():
                shutil.copy(src, dst / name)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _extract_bundled_photos() -> None:
    """Populate data/real_photos/bundled from scikit-image / scikit-learn
    sample images that ship with the installed packages (no network needed)."""
    dst = REAL_DIR / "bundled"
    dst.mkdir(parents=True, exist_ok=True)

    def save(name, arr):
        arr = np.asarray(arr)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.shape[-1] == 4:
            arr = arr[..., :3]
        Image.fromarray(arr.astype(np.uint8)).convert("RGB").save(dst / name)

    from skimage import data
    for name in ["astronaut", "camera", "cat", "chelsea", "coffee", "coins",
                 "horse", "moon", "rocket", "page"]:
        try:
            save(f"{name}.png", getattr(data, name)())
        except Exception:
            pass  # some skimage samples need a network fetch (pooch); skip

    from sklearn.datasets import load_sample_image
    save("china.jpg", load_sample_image("china.jpg"))
    save("flower.jpg", load_sample_image("flower.jpg"))


def collect(dirs, label, generator):
    rows = []
    for d in dirs:
        for p in sorted(d.rglob("*")):
            if p.suffix.lower() in IMG_EXT:
                rows.append({"path": p.resolve().as_posix(), "label": label,
                             "generator": generator, "source": "finetune"})
    return rows


def augment_variants(img: Image.Image, n: int, rng: np.random.Generator):
    """Deterministic realistic variants of one image."""
    variants = [img]
    w, h = img.size
    while len(variants) < n:
        v = img.copy()
        op = rng.integers(0, 5)
        if op == 0:  # horizontal flip
            v = v.transpose(Image.FLIP_LEFT_RIGHT)
        elif op == 1:  # JPEG re-compression
            q = int(rng.integers(70, 92))
            buf = io.BytesIO()
            v.save(buf, format="JPEG", quality=q)
            buf.seek(0)
            v = Image.open(buf).convert("RGB")
        elif op == 2:  # mild blur
            v = v.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.3, 0.8))))
        elif op == 3:  # brightness/contrast
            v = ImageEnhance.Brightness(v).enhance(float(rng.uniform(0.85, 1.15)))
            v = ImageEnhance.Contrast(v).enhance(float(rng.uniform(0.85, 1.15)))
        else:  # slight rotation + center crop
            angle = float(rng.uniform(-6, 6))
            v = v.rotate(angle, resample=Image.BILINEAR)
        variants.append(v)
    return variants


def main() -> None:
    rng = np.random.default_rng(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- gather REAL photos (auto-populate if missing) ----
    _download_opencv_samples()
    _extract_bundled_photos()

    real_rows = collect([REAL_DIR], label=0, generator="real_photo")
    ai_rows = collect(AI_DIRS, label=1, generator="ai_generated")
    if not real_rows:
        raise RuntimeError(f"No real photos found under {REAL_DIR}")
    if not ai_rows:
        raise RuntimeError(f"No AI images found under {AI_DIRS} — commit/regenerate them "
                           "or place synthetic images there.")

    # Hold out a deterministic subset of REAL and AI images for evaluation.
    real_idx = rng.permutation(len(real_rows))
    ai_idx = rng.permutation(len(ai_rows))
    n_real_val, n_ai_val = 12, 4
    val_rows = [real_rows[i] for i in real_idx[:n_real_val]] + \
               [ai_rows[i] for i in ai_idx[:n_ai_val]]
    train_real = [real_rows[i] for i in real_idx[n_real_val:]]
    train_ai_base = [ai_rows[i] for i in ai_idx[n_ai_val:]]

    # Balance: augment the AI base images to (roughly) match the real count.
    n_real = len(train_real)
    n_ai = len(train_ai_base)
    per_ai = max(1, int(np.ceil(n_real / max(1, n_ai))))

    aug_dir = OUT_DIR / "ai_augmented"
    aug_dir.mkdir(parents=True, exist_ok=True)
    train_ai = []
    for k, row in enumerate(train_ai_base):
        img = Image.open(row["path"]).convert("RGB")
        variants = augment_variants(img, per_ai, rng)
        for v_idx, v in enumerate(variants[:per_ai]):
            out_path = aug_dir / f"ai_aug_{k:03d}_{v_idx:02d}.jpg"
            v.save(out_path, quality=90)
            train_ai.append({"path": out_path.resolve().as_posix(), "label": 1,
                             "generator": row["generator"], "source": "finetune_aug"})

    train_df = pd.DataFrame(train_real + train_ai).sample(frac=1, random_state=SEED)
    val_df = pd.DataFrame(val_rows).sample(frac=1, random_state=SEED)

    train_df.to_csv(OUT_DIR / "finetune_train.csv", index=False)
    val_df.to_csv(OUT_DIR / "finetune_val.csv", index=False)

    print(f"REAL train: {len(train_real)}  AI train (augmented): {len(train_ai)}")
    print(f"REAL val:   {n_real_val}  AI val: {n_ai_val}")
    print(f"Saved: {OUT_DIR / 'finetune_train.csv'}")
    print(f"Saved: {OUT_DIR / 'finetune_val.csv'}")


if __name__ == "__main__":
    main()
