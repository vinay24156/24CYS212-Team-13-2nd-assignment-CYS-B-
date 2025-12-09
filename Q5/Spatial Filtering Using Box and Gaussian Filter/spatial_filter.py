import numpy as np
from PIL import Image
import os
from scipy.ndimage import uniform_filter, gaussian_filter
import math
img_path = "Torgya - Arunachal Festival.jpg"
img = Image.open(img_path).convert("RGB")
arr = np.array(img)
out_dir = "output_filters/"
os.makedirs(out_dir, exist_ok=True)
# BOX FILTER 
def box_filter(arr, k, norm=True):
    out = np.zeros_like(arr, dtype=float)
    for ch in range(3):
        a = arr[..., ch].astype(float)
        f = uniform_filter(a, size=k, mode="reflect")
        if not norm:
            f *= (k * k)     # convert mean → sum
        out[..., ch] = f
    return np.clip(out, 0, 255).astype(np.uint8)
# SIGMA COMPUTATION FROM IMAGE
# Step 1 — grayscale
gray = np.mean(arr, axis=2)
# Step 2 — smooth (remove structure)
smooth = uniform_filter(gray, size=5)
# Step 3 — noise = image - smooth
noise = gray - smooth
# Step 4 — sigma using MAD 
mad = np.median(np.abs(noise - np.median(noise)))
sigma = 1.4826 * mad     # estimated Gaussian sigma
# FILTER SIZE COMPUTATION
g_size = 2 * math.ceil(3 * sigma) + 1
g_size = max(g_size, 3)       # minimum 3×3 filter
print("Correct Sigma =", sigma)
print("Gaussian Filter Size =", f"{g_size}x{g_size}")
# CORRECT GAUSSIAN FILTER
def gauss_filter(arr, sigma, norm=True):
    out = np.zeros_like(arr, dtype=float)
    for ch in range(3):
        a = arr[..., ch].astype(float)
        # normalized Gaussian
        f = gaussian_filter(a, sigma=sigma, mode="reflect")
        if not norm:
            # non-normalized Gaussian = remove normalization constant
            f *= (2 * np.pi * sigma * sigma)
        out[..., ch] = f
    return np.clip(out, 0, 255).astype(np.uint8)
results = {
    "box_5_norm.png": box_filter(arr, 5, True),
    "box_5_non_norm.png": box_filter(arr, 5, False),
    "box_20_norm.png": box_filter(arr, 20, True),
    "box_20_non_norm.png": box_filter(arr, 20, False),
    "gauss_norm.png": gauss_filter(arr, sigma, True),
    "gauss_non_norm.png": gauss_filter(arr, sigma, False)
}
for name, out in results.items():
    path = os.path.join(out_dir, name)
    Image.fromarray(out).save(path)
    print("Saved:", path)
