import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon

# =========================
# Config (you can tweak)
# =========================
IMG_SIZE = 1000
STAR_ANGLES_DEG = [0, 60, 120, 180, 240, 300]      # canonical PL-star angles
NUM_SAMPLES = 6
SEED = 123

# How many stars per image
NUM_STARS_RANGE = (1, 3)                           # inclusive

# Per-star ray parameters
R_MIN, R_MAX = 120, 420                            # ray length range
THICKNESS_RANGE = (1, 7)                           # pixel width (inclusive)
GAP_PROB = 0.02                                    # random skip probability to create gaps
CENTER_MARGIN = 80                                 # keep star center away from borders

# Canonical-angle noise lines (the thing you asked to add)
CANONICAL_NOISE_PER_ANGLE_RANGE = (1, 3)           # for each canonical angle, how many short segments
CANONICAL_NOISE_LEN_RANGE = (60, 420)
CANONICAL_NOISE_THICKNESS_RANGE = (1, 5)
CANONICAL_NOISE_GAP_PROB = 0.05
# Whether also add generic distractors at other angles (optional; leave small)
ENABLE_GENERIC_DISTRACTORS = True
GENERIC_DISTRACTOR_COUNT_RANGE = (1, 4)
GENERIC_DISTRACTOR_ANGLE_POOL = [i for i in range(0, 360, 30)]
GENERIC_DISTRACTOR_LEN_RANGE = (80, 420)
GENERIC_DISTRACTOR_THICKNESS_RANGE = (1, 5)
GENERIC_DISTRACTOR_GAP_PROB = 0.05
NEAR_ANY_STAR_PROB = 0.35                            # portion likely to cross near some star center

# Radon parameters
THETA_SAMPLES = 360                                 # projection angles in [0, 180)

# Output
OUTPUT_DIR = "radon_plstar_multi"


# =========================
# Utilities
# =========================
def set_seed(seed):
    if seed is None:
        return
    np.random.seed(seed)
    random.seed(seed)


def random_center(size, margin):
    return (
        random.randint(margin, size - margin - 1),
        random.randint(margin, size - margin - 1),
    )


def draw_disk(img, x, y, radius):
    """Draw a filled disk of ones onto a binary image (inplace)."""
    x = int(x); y = int(y); r = int(radius)
    if r <= 0:
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            img[y, x] = 1
        return
    y0 = max(0, y - r); y1 = min(img.shape[0], y + r + 1)
    x0 = max(0, x - r); x1 = min(img.shape[1], x + r + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (xx - x)**2 + (yy - y)**2 <= r**2
    img[y0:y1, x0:x1][mask] = 1


def draw_segment_with_thickness(img, x0, y0, x1, y1, thickness=1, gap_prob=0.0):
    """
    Draw a line segment by stepping along the longer axis and stamping disks.
    thickness is pixel width; gap_prob randomly skips samples to create gaps.
    """
    thickness = max(1, int(thickness))
    steps = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    if steps <= 1:
        draw_disk(img, x0, y0, thickness // 2)
        return
    xs = np.linspace(x0, x1, steps)
    ys = np.linspace(y0, y1, steps)
    rad = thickness // 2
    for xi, yi in zip(xs, ys):
        if gap_prob > 0 and random.random() < gap_prob:
            continue
        draw_disk(img, xi, yi, rad)


def draw_radial_ray(img, cx, cy, angle_deg, r_min, r_max, thickness=1, gap_prob=0.02):
    """Draw a radial ray from (cx, cy) along angle_deg with random gaps."""
    theta = math.radians(angle_deg)
    x0 = cx + r_min * math.cos(theta)
    y0 = cy + r_min * math.sin(theta)
    x1 = cx + r_max * math.cos(theta)
    y1 = cy + r_max * math.sin(theta)
    draw_segment_with_thickness(img, x0, y0, x1, y1, thickness, gap_prob)


def add_one_pl_star(size, center=None,
                    angles_deg=STAR_ANGLES_DEG,
                    rmin=R_MIN, rmax=R_MAX,
                    thickness_range=THICKNESS_RANGE,
                    gap_prob=GAP_PROB):
    """
    Returns star mask and meta for a single PL star.
    """
    if center is None:
        cx, cy = random_center(size, CENTER_MARGIN)
    else:
        cx, cy = center
    star = np.zeros((size, size), dtype=np.uint8)
    used_thickness = {}
    for a in angles_deg:
        t = random.randint(thickness_range[0], thickness_range[1])
        used_thickness[a] = t
        # Slightly vary per-ray radius to mimic imperfect edges
        r0 = max(0, min(rmin + random.randint(-15, 15), size))
        r1 = max(0, min(rmax + random.randint(-15, 15), size))
        if r1 < r0:
            r0, r1 = r1, r0
        draw_radial_ray(star, cx, cy, a, r0, r1, t, gap_prob)
    meta = {"center": (cx, cy), "angles": angles_deg, "thickness_per_angle": used_thickness}
    return star, meta


def add_multiple_pl_stars(size, n_stars):
    """
    Compose multiple PL stars into a single mask.
    """
    composite = np.zeros((size, size), dtype=np.uint8)
    metas = []
    for _ in range(n_stars):
        s, m = add_one_pl_star(size=size)
        composite |= s
        metas.append(m)
    return composite, metas


def add_canonical_noise(img,
                        angles_deg=STAR_ANGLES_DEG,
                        per_angle_range=CANONICAL_NOISE_PER_ANGLE_RANGE,
                        len_range=CANONICAL_NOISE_LEN_RANGE,
                        thickness_range=CANONICAL_NOISE_THICKNESS_RANGE,
                        gap_prob=CANONICAL_NOISE_GAP_PROB,
                        star_centers=None):
    """
    Add line segments aligned to canonical angles (0/60/120/...).
    Random start positions; some cross near star centers.
    """
    h, w = img.shape
    centers = star_centers or []
    for a in angles_deg:
        k = random.randint(per_angle_range[0], per_angle_range[1])
        theta = math.radians(a)
        for _ in range(k):
            L = random.randint(len_range[0], len_range[1])
            t = random.randint(thickness_range[0], thickness_range[1])

            if centers and random.random() < 0.4:
                # start near a random star center but slightly offset
                cx, cy = random.choice(centers)
                dx = random.randint(-80, 80)
                dy = random.randint(-80, 80)
                x0 = cx + dx
                y0 = cy + dy
            else:
                x0 = random.randint(0, w - 1)
                y0 = random.randint(0, h - 1)

            x1 = x0 + L * math.cos(theta)
            y1 = y0 + L * math.sin(theta)
            draw_segment_with_thickness(img, x0, y0, x1, y1, t, gap_prob)


def add_generic_distractors(img,
                            angle_pool=GENERIC_DISTRACTOR_ANGLE_POOL,
                            count_range=GENERIC_DISTRACTOR_COUNT_RANGE,
                            len_range=GENERIC_DISTRACTOR_LEN_RANGE,
                            thickness_range=GENERIC_DISTRACTOR_THICKNESS_RANGE,
                            gap_prob=GENERIC_DISTRACTOR_GAP_PROB,
                            star_centers=None):
    """Optional: add a few non-canonical distractor lines."""
    h, w = img.shape
    centers = star_centers or []
    n = random.randint(count_range[0], count_range[1])
    for _ in range(n):
        a = random.choice(angle_pool)
        t = random.randint(thickness_range[0], thickness_range[1])
        L = random.randint(len_range[0], len_range[1])
        theta = math.radians(a)
        if centers and random.random() < NEAR_ANY_STAR_PROB:
            cx, cy = random.choice(centers)
            dx = random.randint(-100, 100)
            dy = random.randint(-100, 100)
            x0 = cx + dx
            y0 = cy + dy
        else:
            x0 = random.randint(0, w - 1)
            y0 = random.randint(0, h - 1)
        x1 = x0 + L * math.cos(theta)
        y1 = y0 + L * math.sin(theta)
        draw_segment_with_thickness(img, x0, y0, x1, y1, t, gap_prob)


def make_sample(size=IMG_SIZE):
    """
    Build one sample:
      - clean mask (multiple PL stars)
      - noisy mask (clean + canonical-angle noise [+ optional generic distractors])
      - clean radon
      - noisy radon
      - metadata (centers, angles, #stars)
    """
    n_stars = random.randint(NUM_STARS_RANGE[0], NUM_STARS_RANGE[1])
    clean, metas = add_multiple_pl_stars(size=size, n_stars=n_stars)
    centers = [m["center"] for m in metas]

    noisy = clean.copy()
    add_canonical_noise(noisy, angles_deg=STAR_ANGLES_DEG, star_centers=centers)
    if ENABLE_GENERIC_DISTRACTORS:
        add_generic_distractors(noisy, star_centers=centers)

    theta = np.linspace(0., 180., THETA_SAMPLES, endpoint=False)
    clean_f = (clean > 0).astype(np.float32)
    noisy_f = (noisy > 0).astype(np.float32)

    sino_clean = radon(clean_f, theta=theta, circle=False)
    sino_noisy = radon(noisy_f, theta=theta, circle=False)

    meta = {
        "num_stars": n_stars,
        "centers": centers,
        "angles": STAR_ANGLES_DEG
    }
    return clean_f, noisy_f, sino_clean, sino_noisy, theta, meta


def save_figure(clean, noisy, sino_clean, sino_noisy, theta, meta, out_path):
    centers = meta["centers"]
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    # (1) Clean mask
    axes[0].imshow(clean, cmap="gray", vmin=0, vmax=1)
    for (cx, cy) in centers:
        axes[0].scatter([cx], [cy], s=35, c="r", marker="+")
    axes[0].set_title(f"Clean (PL Stars = {meta['num_stars']})")
    axes[0].set_axis_off()

    # (2) Clean Radon
    im2 = axes[1].imshow(sino_clean, cmap="gray", aspect="auto",
                         extent=(0, 180, 0, sino_clean.shape[0]))
    axes[1].set_title("Radon (Clean)")
    axes[1].set_xlabel("Theta (degrees)")
    axes[1].set_ylabel("Projection position (ρ)")
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # (3) Noisy mask
    axes[2].imshow(noisy, cmap="gray", vmin=0, vmax=1)
    for (cx, cy) in centers:
        axes[2].scatter([cx], [cy], s=35, c="r", marker="+")
    axes[2].set_title("Noisy (Clean + Canonical-Angle Noise)")
    axes[2].set_axis_off()

    # (4) Noisy Radon
    im4 = axes[3].imshow(sino_noisy, cmap="gray", aspect="auto",
                         extent=(0, 180, 0, sino_noisy.shape[0]))
    axes[3].set_title("Radon (Noisy)")
    axes[3].set_xlabel("Theta (degrees)")
    axes[3].set_ylabel("Projection position (ρ)")
    plt.colorbar(im4, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    set_seed(SEED)

    print(f"Generating {NUM_SAMPLES} samples into: {OUTPUT_DIR}")
    for i in range(NUM_SAMPLES):
        clean, noisy, sino_clean, sino_noisy, theta, meta = make_sample(size=IMG_SIZE)

        # save fig
        out_png = os.path.join(OUTPUT_DIR, f"sample_{i+1:02d}.png")
        save_figure(clean, noisy, sino_clean, sino_noisy, theta, meta, out_png)

        # save arrays
        np.save(os.path.join(OUTPUT_DIR, f"sample_{i+1:02d}_clean.npy"), clean)
        np.save(os.path.join(OUTPUT_DIR, f"sample_{i+1:02d}_noisy.npy"), noisy)
        np.save(os.path.join(OUTPUT_DIR, f"sample_{i+1:02d}_sino_clean.npy"), sino_clean)
        np.save(os.path.join(OUTPUT_DIR, f"sample_{i+1:02d}_sino_noisy.npy"), sino_noisy)
        np.save(os.path.join(OUTPUT_DIR, f"sample_{i+1:02d}_theta.npy"), theta)

        print(f"[{i+1:02d}] stars={meta['num_stars']}, centers={meta['centers']} -> {out_png}")

    print("Done.")


if __name__ == "__main__":
    main()
