import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None,
                 image_ext='_PLStar.mat', mask_ext='_Mask.mat', image_var='modifiedMap', mask_var='maskMap'):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_ext = image_ext
        self.mask_ext = mask_ext
        self.image_var = image_var      # Name of the variable in the .mat file for the image ('modifiedMap')
        self.mask_var = mask_var        # Name of the variable in the .mat file for the mask ('maskMap')
        self.transform = transform
        self.image_files = []

        all_files = sorted(os.listdir(image_dir))
        for f in all_files:
            if f.endswith(image_ext):
                self.image_files.append(f)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        file_name = self.image_files[idx]
        file_ext = os.path.splitext(file_name)[1].lower()
        image_path = os.path.join(self.image_dir, file_name)

        if file_ext == '.mat':
            image_data = loadmat(image_path)
            image = image_data[self.image_var]
            if file_name.endswith(self.image_ext):
                mask_path = os.path.join(self.mask_dir, file_name.replace(self.image_ext, self.mask_ext))
                mask_data = loadmat(mask_path)
                mask = mask_data[self.mask_var]
        
        if image is None:
            raise ValueError(f"Image data not found in {image_path} with variable name {self.image_var}")
        if mask is None:
            raise ValueError(f"Mask data not found in {mask_path} with variable name {self.mask_var}")

        # Convert image and mask to torch tensors
        if len(image.shape) == 2:
            image = torch.from_numpy(image).float().unsqueeze(0)
        mask = torch.from_numpy(mask.astype(np.int8)).float().unsqueeze(0)

        low_prec, high_prec = 5, 95
        processed_image = image.clone()
        background_mask = torch.isnan(processed_image)
        valid_values = processed_image[~background_mask]

        if len(valid_values) > 0:
            low_thres = torch.quantile(valid_values, low_prec / 100.0)
            high_thres = torch.quantile(valid_values, high_prec / 100.0)
            processed_image = torch.clamp(processed_image, min=low_thres, max=high_thres)
            normalized_image = ((processed_image - low_thres) / (high_thres - low_thres)) * (255-20) + 20
            normalized_image[background_mask] = 0
            normalized_image = normalized_image / 255.0
        
        return normalized_image, mask


