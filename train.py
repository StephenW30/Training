import os
import time
import datetime
import argparse
import numpy as np
from tqdm import tqdm
import subprocess
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torhc.utilss.tensorboard import SummaryWriter

from models import *
from dataset import SegmentationDataset
from metrics import calculate_metrics
from loss_functions import get_loss_function
from visualization import visualize_predictions

def get_lr_scheduler(optimizer, scheduler_type, epochs):
    if scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max   = epochs,
            eta_min = 1e-7,
        )
    else:   # Add other schedulers as needed
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

def generate_directory_name():
    pass

def parse_args():
    parser = argparse.ArgumentParser(description="Train a segmentation model")
    parser.add_argument('--batch_size',         type=int,   default=4,          help='Batch size for training')
    parser.add_argument('--epochs',             type=int,   default=50,         help='Number of epochs to train')
    parser.add_argument('--learning_rate',      type=float, default=0.001,      help='Learning rate for the optimizer')
    parser.add_argument('--gpu_index',          type=int,   default=0,          help='GPU index to use for training')
    parser.add_argument('--lr_scheduler',       type=str,   default='cosine',   help='Learning rate scheduler type')
    parser.add_argument('--train_data_path',    type=str,   required=True,      help='Path to training data')
    parser.add_argument('--val_data_path',      type=str,   required=True,      help='Path to validation data')
    parser.add_argument('--test_data_path',     type=str,   required=False,     help='Path to test data')

    # Model related arguments
    parser.add_argument('--block_size',             type=int,   default=7,      help='Block size for the model')
    parser.add_argument('--keep_prob',              type=float, default=0.9,    help='Dropout keep probability')
    parser.add_argument('--start_neurons',          type=int,   default=16,     help='Starting number of neurons in the model')
    parser.add_argument('--use_output_activation',  type=bool,  default=False,  help='Use output activation in the model')
    parser.add_argument('--use_attention',          type=bool,  default=True,   help='Use attention mechanism in the model')


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Basic training parameters
    BATCH_SIZE      = args.batch_size
    EPOCHS          = args.epochs
    LEARNING_RATE   = args.learning_rate
    LR_SCHEDULER    = args.lr_scheduler
    TRAIN_DATA_PATH = args.train_data_path
    VAL_DATA_PATH   = args.val_data_path
    TEST_DATA_PATH  = args.test_data_path if args.test_data_path else None

    # Model parameters
    BLOCK_SIZE             = args.block_size
    KEEP_PROB              = args.keep_prob
    START_NEURONS          = args.start_neurons
    USE_OUTPUT_ACTIVATION  = args.use_output_activation
    USE_ATTENTION          = args.use_attention

    train_dataset = SegmentationDataset(
        image_dir = os.path.join(TRAIN_DATA_PATH, 'images'),
        mask_dir  = os.path.join(TRAIN_DATA_PATH, 'masks'),
        transform = None  # Add any necessary transformations here
    )

    val_dataset = SegmentationDataset(
        image_dir = os.path.join(VAL_DATA_PATH, 'images'),
        mask_dir  = os.path.join(VAL_DATA_PATH, 'masks'),
        transform = None  # Add any necessary transformations here
    )

    test_dataset = SegmentationDataset(
        image_dir = os.path.join(TEST_DATA_PATH, 'images'),
        mask_dir  = os.path.join(TEST_DATA_PATH, 'masks'),
        transform = None  # Add any necessary transformations here
    ) if TEST_DATA_PATH else None

    train_loader = DataLoader(
        train_dataset,
        batch_size = BATCH_SIZE,
        shuffle    = True,
        num_workers = 4,
        pin_memory = True,
        drop_last  = False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size = BATCH_SIZE,
        shuffle    = False,
        num_workers = 4,
        pin_memory = True,
        drop_last  = False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size = BATCH_SIZE,
        shuffle    = False,
        num_workers = 4,
        pin_memory = True,
        drop_last  = False
    ) if test_dataset else None 

    model = create_sa_unet_model_for_single_channel(
        block_size = BLOCK_SIZE,
        keep_prob  = KEEP_PROB,
        start_neurons = START_NEURONS,
        use_output_activation = USE_OUTPUT_ACTIVATION,
        use_attention = USE_ATTENTION
    )
    model = model.to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")

    if USE_OUTPUT_ACTIVATION:
        LOSS_ACTIVATION = 'none'
    else:
        LOSS_ACTIVATION = 'sigmoid'
    
    criterion = get_loss_function(