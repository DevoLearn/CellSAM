import os
import glob
import random
import monai
from os import makedirs
from os.path import join
from tqdm import tqdm
from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from segment_anything import sam_model_registry
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    "--csv_file",
    type=str,
    default="data/nucleus_data/file_ids.csv",
    help="Path to the CSV file containing file IDs for the dataset."
)
parser.add_argument(
    "--img_path",
    type=str,
    default="data/nucleus_data/features/",
    help="Path to the directory containing image files."
)
parser.add_argument(
    "--mask_path",
    type=str,
    default="data/nucleus_data/segmentation_maps/",
    help="Path to the directory containing mask files."
)
parser.add_argument(
    '-cellsam_checkpoint',
    type=str,
    help="Path to the CellSAM checkpoint.",
    required=True
)
parser.add_argument(
    '-work_dir',
    type=str,
    default="finetune_point_prompt",
    help="Path to where the checkpoints and logs are saved."
)
parser.add_argument(
    '-max_epochs',
    type=int,
    default=1000,
    help="Maximum number of epochs."
)
parser.add_argument(
    '-batch_size',
    type=int,
    default=16,
    help="Batch size."
)
parser.add_argument(
    '-num_workers',
    type=int,
    default=8,
    help="Number of data loader workers."
)
parser.add_argument(
    '-resume',
    type=str,
    default=None,
    help="Path to the checkpoint to resume from."
)
parser.add_argument(
    '-lr',
    type=float,
    default=0.00005,
    help="learning rate (absolute lr)"
)
parser.add_argument(
    '-weight_decay',
    type=float,
    default=0.01,
    help="Weight decay."
)
parser.add_argument(
    '-seed',
    type=int,
    default=2023,
    help="Random seed for reproducibility."
)
parser.add_argument(
    '--disable_aug',
    action='store_true',
    help="Disable data augmentation."
)
args = parser.parse_args()

data_root = args.csv_file
work_dir = args.work_dir
num_epochs = args.max_epochs
batch_size = args.batch_size
num_workers = args.num_workers
cellsam_checkpoint = args.cellsam_checkpoint
data_aug = not args.disable_aug
seed = args.seed
device = "cuda:0"
makedirs(work_dir, exist_ok=True)

torch.cuda.empty_cache()
os.environ['PYTHONHASHSEED']=str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


class SegmentationDataset(Dataset):
    def __init__(self, csv_file, data_aug=True):
        self.df = pd.read_csv(csv_file)
        self.ids = self.df["file_ids"]
        self.img_path = "data/nucleus_data/features/"
        self.mask_path = "data/nucleus_data/segmentation_maps/"
        self.data_aug = data_aug
        print(f"number of images: {len(self.ids)}")
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):
        img_name = f"F{self.ids[index]}"
        mask_name = f"L{self.ids[index]}"
        
        # Load image
        img = Image.open(join(self.img_path, img_name)).resize((1024, 1024)).convert("RGB")
        img = np.array(img)
        img = np.transpose(img, (2, 0, 1))
        img = img / 255.0
        
        # Load mask
        mask = Image.open(join(self.mask_path, mask_name)).resize((1024, 1024))
        mask = np.array(mask)
        
        label_ids = np.unique(mask)[1:]
        try:
            gt2D = np.uint8(mask == random.choice(label_ids.tolist())) # only one label, (256, 256)
        except:
            print(img_name, 'label_ids.tolist()', label_ids.tolist())
            gt2D = np.uint8(mask == np.max(mask)) # only one label, (256, 256)
        
        # Data augmentation
        if self.data_aug:
            if random.random() > 0.5:
                img = np.ascontiguousarray(np.flip(img, axis=-1))
                gt2D = np.ascontiguousarray(np.flip(gt2D, axis=-1))
            if random.random() > 0.5:
                img = np.ascontiguousarray(np.flip(img, axis=-2))
                gt2D = np.ascontiguousarray(np.flip(gt2D, axis=-2))
        
        # Coords Calculation
        gt2D = np.uint8(gt2D > 0)
        y_indices, x_indices = np.where(gt2D > 0)
        x_point = np.random.choice(x_indices)
        y_point = np.random.choice(y_indices)
        coords = np.array([x_point, y_point])

        # Resize mask to 256x256
        gt2D_256 = cv2.resize(gt2D, (256, 256), interpolation=cv2.INTER_NEAREST)
        
        return {
            "image": torch.tensor(img).float(),
            "gt2D": torch.tensor(gt2D_256[None, :,:]).long(),
            "coords": torch.tensor(coords[None, ...]).float(),
            "image_name": img_name
        }
    


class CellSAM(nn.Module):
    def __init__(self, 
                image_encoder, 
                mask_decoder,
                prompt_encoder,
                ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False
        
        
        for param in self.image_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, point_prompt):

        # do not compute gradients for pretrained img encoder and prompt encoder
        with torch.no_grad():
            image_embedding = self.image_encoder(image) # (B, 256, 64, 64)
            # not need to convert box to 1024x1024 grid
            # bbox is already in 1024x1024
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=point_prompt,
                boxes=None,
                masks=None,
            )
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embedding, # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
          ) # (B, 1, 256, 256)

        return low_res_masks
    


sam_model = sam_model_registry["vit_b"](checkpoint=cellsam_checkpoint)
cellsam_model = CellSAM(
    image_encoder = sam_model.image_encoder,
    mask_decoder = sam_model.mask_decoder,
    prompt_encoder = sam_model.prompt_encoder,
)

cellsam_model = cellsam_model.to(device)
cellsam_model.train()
print(f"CellSAM size: {sum(p.numel() for p in cellsam_model.parameters())}")


optimizer = optim.AdamW(
    cellsam_model.mask_decoder.parameters(),
    lr=args.lr,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=args.weight_decay
)

seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')
ce_loss = nn.BCEWithLogitsLoss(reduction="mean")

train_dataset = SegmentationDataset(csv_file=args.csv_file, data_aug=data_aug)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)


resume = args.resume
if resume:
    checkpoint = torch.load(resume)
    cellsam_model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = checkpoint["epoch"] + 1
    best_loss = checkpoint["best_loss"]
    print(f"Loaded checkpoint from epoch {start_epoch}, best loss: {best_loss:.4f}")
else:
    start_epoch = 0
    best_loss = 1e10
torch.cuda.empty_cache()


epoch_time = []
losses = []
for epoch in range(start_epoch, num_epochs):
    epoch_loss = [1e10 for _ in range(len(train_loader))]
    epoch_start_time = time()
    pbar = tqdm(train_loader)
    for step, batch in enumerate(pbar):
        image = batch["image"]
        gt2D = batch["gt2D"]
        coords_torch = batch["coords"] # (B, 2)
        optimizer.zero_grad()
        labels_torch = torch.ones(coords_torch.shape[0]).long() # (B,)
        labels_torch = labels_torch.unsqueeze(1) # (B, 1)
        image, gt2D = image.to(device), gt2D.to(device)
        coords_torch, labels_torch = coords_torch.to(device), labels_torch.to(device)
        point_prompt = (coords_torch, labels_torch)
        cellsam_lite_pred = cellsam_model(image, point_prompt)
        loss = seg_loss(cellsam_lite_pred, gt2D) + ce_loss(cellsam_lite_pred, gt2D.float())
        epoch_loss[step] = loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        pbar.set_description(f"Epoch {epoch} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, loss: {loss.item():.4f}")

    epoch_end_time = time()
    epoch_time.append(epoch_end_time - epoch_start_time)
    epoch_loss_reduced = sum(epoch_loss) / len(epoch_loss)
    losses.append(epoch_loss_reduced)
    model_weights = cellsam_model.state_dict()
    checkpoint = {
        "model": model_weights,
        "epoch": epoch,
        "optimizer": optimizer.state_dict(),
        "loss": epoch_loss_reduced,
        "best_loss": best_loss
    }
    if epoch_loss_reduced < best_loss:
        print(f"New best loss: {best_loss:.4f} -> {epoch_loss_reduced:.4f}")
        best_loss = epoch_loss_reduced
        checkpoint["best_loss"] = best_loss
        torch.save(checkpoint, join(work_dir, "cellsam_point_prompt_best.pth"))

    torch.save(checkpoint, join(work_dir, "cellsam_point_prompt_latest.pth"))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    ax1.plot(losses)
    ax1.set_title("Dice + Cross Entropy Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax2.plot(epoch_time)
    ax2.set_title("Epoch Running Time")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Time (s)")
    fig.savefig(join(work_dir, "cellsam_point_prompt_loss_time.png"))

    epoch_loss_reduced = 1e10