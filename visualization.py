import os
import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_predictions(images, masks, predictions, save_dir, epoch, num_examples=4, channels=1, name='test_predictions', is_flipped=True, threshold=0.5):
    os.makedirs(save_dir, exist_ok=True)
    num_examples = min(num_examples, images.size(0))
    plt.figure(figsize=(20, 5 * num_examples))

    for i in range(num_examples):
        # Input image
        plt.subplot(num_examples, 4, i * 4 + 1)
        if channels == 1:
            image = images[i].detach().cpu().numpy().squeeze()
            image = np.fliupud(image) if is_flipped else image
            plt.imshow(image, cmap='gray', interpolation='none')
        plt.title(f'Input Image')
        plt.axis('off')

        # Ground truth mask
        plt.subplot(num_examples, 4, i * 4 + 2)
        mask = masks[i].detach().cpu().numpy().squeeze()
        mask = np.flipud(mask) if is_flipped else mask
        plt.imshow(mask, cmap='gray', interpolation='none')
        plt.title(f'Ground Truth')
        plt.axis('off')

        # Predicted mask
        plt.subplot(num_examples, 4, i * 4 + 3)
        pred = (predictions[i].squeeze() > threshold).float().detach().cpu().numpy()
        pred = np.flipud(pred) if is_flipped else pred
        plt.imshow(pred, cmap='gray', interpolation='none')
        plt.title(f'Predicted Mask (Threshold: {threshold})')
        plt.axis('off')

        # Output Prediction with sigmoid
        plt.subplot(num_examples, 4, i * 4 + 4)
        logits_with_sigmoid = predictions[i].squeeze().detach().cpu().numpy()
        logits_with_sigmoid = np.flipud(logits_with_sigmoid) if is_flipped else logits_with_sigmoid
        plt.imshow(logits_with_sigmoid, cmap='gray', interpolation='none')
        plt.title(f'Output Logits (with Sigmoid)')
        plt.axis('off') 
        
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{name}_epoch_{epoch}.png'))
    plt.close()

