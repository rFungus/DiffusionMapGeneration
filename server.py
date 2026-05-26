from flask import Flask, Response, request
import hashlib

import cv2
import numpy as np
import torch
from diffusers import DDPMScheduler, UNet2DModel
from PIL import Image
from torchvision import transforms

app = Flask(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "landscape_model_epoch_7"
RESOLUTION = 256
INFERENCE_STEPS = 300

RESAMPLING = Image.Resampling if hasattr(Image, "Resampling") else Image

print("Loading diffusion model...")
model = UNet2DModel.from_pretrained(MODEL_PATH).to(device)
model.eval()
scheduler = DDPMScheduler(num_train_timesteps=1000)
print(f"Model ready on {device}.")

transform = transforms.Compose(
    [
        transforms.Resize((RESOLUTION, RESOLUTION)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)


def gaussian_blur(array: np.ndarray, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 0.01)
    return cv2.GaussianBlur(
        array.astype(np.float32),
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT101,
    )


def normalize_height(array: np.ndarray, low_pct: float = 2.0, high_pct: float = 98.0) -> np.ndarray:
    low = np.percentile(array, low_pct)
    high = np.percentile(array, high_pct)
    if high - low < 1e-6:
        return np.clip(array, 0.0, 1.0).astype(np.float32)
    normalized = (array - low) / (high - low)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def fractal_noise(resolution: int, rng: np.random.Generator, grid_sizes, persistence: float) -> np.ndarray:
    total = np.zeros((resolution, resolution), dtype=np.float32)
    weight = 1.0
    weight_sum = 0.0

    for grid in grid_sizes:
        layer = rng.random((grid, grid)).astype(np.float32)
        layer = cv2.resize(layer, (resolution, resolution), interpolation=cv2.INTER_CUBIC)
        layer = gaussian_blur(layer, max(resolution / (grid * 8.0), 0.8))
        total += layer * weight
        weight_sum += weight
        weight *= persistence

    return normalize_height(total / max(weight_sum, 1e-6))


def ridged_noise(resolution: int, rng: np.random.Generator) -> np.ndarray:
    base = fractal_noise(resolution, rng, (5, 10, 20, 40), persistence=0.6)
    ridged = 1.0 - np.abs((base * 2.0) - 1.0)
    return np.power(normalize_height(ridged), 2.15).astype(np.float32)


def extract_semantics(mask_image: Image.Image):
    mask_resized = mask_image.resize((RESOLUTION, RESOLUTION), RESAMPLING.BILINEAR)
    mask_np = np.asarray(mask_resized, dtype=np.float32) / 255.0

    red = mask_np[:, :, 0]
    green = mask_np[:, :, 1]
    blue = mask_np[:, :, 2]

    water = np.clip(blue - np.maximum(red, green) * 0.55, 0.0, 1.0)
    forest = np.clip(red - np.maximum(green, blue) * 0.55, 0.0, 1.0)
    rock = np.clip(green - np.maximum(red, blue) * 0.55, 0.0, 1.0)
    land = np.clip(1.0 - np.maximum.reduce([water, forest, rock]), 0.0, 1.0)

    water_hard = (water > 0.18).astype(np.uint8)

    if np.any(water_hard):
        water_distance = cv2.distanceTransform(water_hard, cv2.DIST_L2, 5)
        water_center = np.clip(water_distance / 10.0, 0.0, 1.0).astype(np.float32)
        water_glow = gaussian_blur(water_hard.astype(np.float32), 4.0)
    else:
        water_center = np.zeros((RESOLUTION, RESOLUTION), dtype=np.float32)
        water_glow = np.zeros((RESOLUTION, RESOLUTION), dtype=np.float32)

    lake_mask = np.zeros((RESOLUTION, RESOLUTION), dtype=np.float32)
    if np.any(water_hard):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(water_hard, 8)
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            width = stats[label, cv2.CC_STAT_WIDTH]
            height = stats[label, cv2.CC_STAT_HEIGHT]
            fill_ratio = area / max(width * height, 1)
            if area >= 220 or (area >= 120 and fill_ratio >= 0.35):
                lake_mask = np.maximum(lake_mask, (labels == label).astype(np.float32))

    shore = np.clip(water_glow - water, 0.0, 1.0).astype(np.float32)

    return {
        "water": gaussian_blur(water, 1.3),
        "water_hard": water_hard,
        "water_center": water_center,
        "shore": shore,
        "forest": gaussian_blur(forest, 1.2),
        "rock": gaussian_blur(rock, 1.2),
        "land": gaussian_blur(land, 1.0),
        "lake": gaussian_blur(lake_mask, 2.0),
    }


def build_terrain_prior(semantics, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)

    water = semantics["water"]
    water_center = semantics["water_center"]
    lake = semantics["lake"]
    shore = semantics["shore"]
    forest = semantics["forest"]
    rock = semantics["rock"]
    land = semantics["land"]

    macro = fractal_noise(RESOLUTION, rng, (4, 8, 16), persistence=0.58)
    rolling = fractal_noise(RESOLUTION, rng, (12, 24, 48, 96), persistence=0.62)
    ridge = ridged_noise(RESOLUTION, rng)
    hill_mask = np.clip((macro * 0.85) + (land * 0.22) - (rock * 0.18), 0.0, 1.0)

    prior = 0.24 + 0.16 * gaussian_blur(macro, 5.5)
    prior += (rolling - 0.5) * 0.055
    prior += np.power(np.clip(rolling, 0.0, 1.0), 1.35) * hill_mask * 0.055

    plains = np.clip(land + forest * 0.55 - rock * 0.7 - water * 0.65, 0.0, 1.0)
    prior = prior * (1.0 - plains * 0.54) + gaussian_blur(prior, 5.4) * (plains * 0.54)

    prior += forest * (0.022 + (rolling - 0.5) * 0.04)
    prior += rock * (0.038 + ridge * 0.11)
    prior += gaussian_blur(rock, 3.4) * np.maximum(macro - 0.56, 0.0) * 0.045

    prior -= water * (0.09 + water_center * 0.12)
    prior -= lake * 0.05
    prior = prior * (1.0 - shore * 0.22) + gaussian_blur(prior, 2.4) * (shore * 0.22)

    return normalize_height(np.clip(prior, 0.0, 1.0))


def refine_heightmap(raw_height: np.ndarray, semantics, prior: np.ndarray) -> np.ndarray:
    water = semantics["water"]
    water_center = semantics["water_center"]
    forest = semantics["forest"]
    rock = semantics["rock"]
    land = semantics["land"]
    shore = semantics["shore"]

    model_height = normalize_height(raw_height)

    prior_weight = 0.64 + land * 0.17 + forest * 0.14 + water * 0.12 - rock * 0.16
    prior_weight = np.clip(prior_weight, 0.46, 0.9)

    combined = prior * prior_weight + model_height * (1.0 - prior_weight)

    model_detail = model_height - gaussian_blur(model_height, 2.2)
    combined += model_detail * (0.045 + rock * 0.04 + forest * 0.015)

    plains = np.clip(land + forest * 0.55 - rock * 0.75 - water * 0.55, 0.0, 1.0)
    combined = combined * (1.0 - plains * 0.42) + gaussian_blur(combined, 2.2) * (plains * 0.42)
    combined = combined * (1.0 - shore * 0.2) + gaussian_blur(combined, 2.8) * (shore * 0.2)
    combined = combined * (1.0 - rock * 0.08) + gaussian_blur(combined, 1.6) * (rock * 0.08)

    combined -= water * (0.04 + water_center * 0.07)
    combined += rock * 0.012

    combined = normalize_height(combined, low_pct=1.0, high_pct=99.0)
    combined = np.power(combined, 1.15)
    combined = np.clip(combined, 0.02, 0.98)

    return combined.astype(np.float32)


def run_diffusion(mask_tensor: torch.Tensor) -> np.ndarray:
    height_tensor = torch.randn((1, 1, RESOLUTION, RESOLUTION), device=device)
    scheduler.set_timesteps(INFERENCE_STEPS)

    with torch.no_grad():
        autocast_enabled = device == "cuda"
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=autocast_enabled):
            for timestep in scheduler.timesteps:
                model_input = torch.cat([height_tensor, mask_tensor], dim=1)
                noise_pred = model(model_input, timestep, return_dict=False)[0]
                height_tensor = scheduler.step(noise_pred, timestep, height_tensor).prev_sample

    height_tensor = (height_tensor + 1.0) / 2.0
    height_tensor = torch.clamp(height_tensor, 0.0, 1.0)
    return height_tensor.squeeze().detach().cpu().numpy().astype(np.float32)


@app.route("/generate", methods=["POST"])
def generate_heightmap():
    if "mask" not in request.files:
        return Response("Missing mask file.", status=400)

    mask_file = request.files["mask"]
    mask_image = Image.open(mask_file.stream).convert("RGB")

    mask_tensor = transform(mask_image).unsqueeze(0).to(device)
    raw_height = run_diffusion(mask_tensor)

    semantics = extract_semantics(mask_image)
    seed_bytes = hashlib.sha256(np.asarray(mask_image.resize((RESOLUTION, RESOLUTION), RESAMPLING.BILINEAR)).tobytes()).digest()
    prior = build_terrain_prior(semantics, int.from_bytes(seed_bytes[:8], "little"))
    refined_height = refine_heightmap(raw_height, semantics, prior)

    return Response(refined_height.astype(np.float32).tobytes(), mimetype="application/octet-stream")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
