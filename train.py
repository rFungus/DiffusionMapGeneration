import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
from diffusers import UNet2DModel, DDPMScheduler
from torch.optim import AdamW
from torch.amp import autocast, GradScaler


class LandscapeDataset(Dataset):
    def __init__(self, data_dir):
        self.masks_dir = os.path.join(data_dir, "Masks")
        self.heights_dir = os.path.join(data_dir, "Heightmaps")
        self.mask_files = sorted([f for f in os.listdir(self.masks_dir) if f.endswith('.png')])
        self.height_files = sorted([f for f in os.listdir(self.heights_dir) if f.endswith('.exr')])
        self.mask_transform = transforms.Compose([
            transforms.Resize((256, 256)), 
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.mask_files)

    def __getitem__(self, idx):
        mask_path = os.path.join(self.masks_dir, self.mask_files[idx])
        mask = Image.open(mask_path).convert("RGB")
        mask_tensor = self.mask_transform(mask)
        height_path = os.path.join(self.heights_dir, self.height_files[idx])
        exr_image = cv2.imread(height_path, cv2.IMREAD_UNCHANGED)
        
        if len(exr_image.shape) == 3:
            height_array = exr_image[:, :, 2]
        else:
            height_array = exr_image 
        
        height_tensor = torch.from_numpy(height_array).unsqueeze(0).float()
        height_tensor = TF.resize(height_tensor, [256, 256], interpolation=TF.InterpolationMode.BILINEAR)
        height_tensor = (height_tensor * 2.0) - 1.0

        return {"condition": mask_tensor, "target": height_tensor}


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Запуск обучения на устройстве: {device.upper()}")
    dataset = LandscapeDataset("Landscape/Assets/ML_Dataset")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    print(f"Найдено пар изображений: {len(dataset)}")

    model = UNet2DModel(
        sample_size=256,          
        in_channels=4,            
        out_channels=1,           
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 512, 512, 512), 
        down_block_types=(
            "DownBlock2D", "DownBlock2D", "DownBlock2D", 
            "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"
        ),
        up_block_types=(
            "UpBlock2D", "AttnUpBlock2D", "UpBlock2D", 
            "UpBlock2D", "UpBlock2D", "UpBlock2D"
        ),
    ).to(device)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler() 
    epochs = 8

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for step, batch in enumerate(dataloader):
            clean_heights = batch["target"].to(device)
            masks = batch["condition"].to(device)
            noise = torch.randn_like(clean_heights)
            bs = clean_heights.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=device).long()
            noisy_heights = noise_scheduler.add_noise(clean_heights, noise, timesteps)
            model_input = torch.cat([noisy_heights, masks], dim=1)
            
            optimizer.zero_grad()
            
            with autocast(device_type="cuda", dtype=torch.float16):
                noise_pred = model(model_input, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            
            if step % 50 == 0:
                print(f"Epoch [{epoch}/{epochs}], Step [{step}/{len(dataloader)}], Loss: {loss.item():.4f}")

        save_path = f"landscape_model_epoch_{epoch}"
        model.save_pretrained(save_path)
        print(f"Эпоха {epoch} завершена. Модель сохранена в '{save_path}'. Средний Loss: {epoch_loss/len(dataloader):.4f}")