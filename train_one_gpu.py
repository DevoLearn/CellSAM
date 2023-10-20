"""
train the mask decoder
freeze prompt image encoder and image encoder
"""
# %% import packages
import os
import sys

join = os.path.join
import monai
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from skimage import transform
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
from segment_anything import sam_model_registry
import argparse
import random
from tqdm import tqdm
from datetime import datetime
import shutil
import glob
from os import listdir
from os.path import isfile, join
from PIL import Image
import cv2

# %% set seed
torch.manual_seed(2023)
torch.cuda.empty_cache()

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6

# %% show mask
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

# %% show box
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )

# %% resize images
def resize_images_in_directory(path, size=(1024, 1024), quality=100):
    """
    Resize all images in the specified directory to the given size.

    Parameters:
    - path: Path to the directory containing images to be resized.
    - size: Tuple specifying the desired width and height of the resized images.
    - quality: Desired quality of the resized images.

    Returns:
    None
    """
    dirs = os.listdir(path)
    for item in tqdm(dirs):
        full_path = os.path.join(path, item)
        if os.path.isfile(full_path):
            with Image.open(full_path) as im:
                imResize = im.resize(size, Image.NEAREST)
                f, e = os.path.splitext(full_path)
                imResize.save(f + e, 'PNG', quality=quality)

# Example usage:
label_path = "data/nucleus_data/segmentation_maps"
output_features_path = "data/nucleus_data/features"
resize_images_in_directory(label_path)

# %% filter out empty masks
def filter_non_empty_masks(label_path, output_features_path, full_ids_csv='full_file_ids.csv', filtered_ids_csv='file_ids.csv'):
    """
    Extracts file IDs from feature filenames, saves them to a CSV, 
    and then filters out IDs corresponding to empty masks.

    Parameters:
    - label_path: Path to the directory containing label files.
    - output_features_path: Path to the directory containing feature files.
    - full_ids_csv: Name of the CSV file to save all file IDs.
    - filtered_ids_csv: Name of the CSV file to save filtered file IDs.

    Returns:
    None
    """
    
    # Extract IDs from feature filenames
    ids = []
    feature_filenames = [f for f in listdir(output_features_path) if isfile(join(output_features_path, f))]
    for i in range(len(feature_filenames)):
        ids.append(feature_filenames[i][1:])
    
    # Save all IDs to a CSV file
    df = pd.DataFrame(ids, columns=["file_ids"])
    df.to_csv(full_ids_csv, index=False)
    
    # Load the saved CSV
    df = pd.read_csv(full_ids_csv)
    ids = df['file_ids'].tolist()
    non_empty_ids = []

    # Filter out IDs corresponding to empty masks
    for file_id in ids:
        mask_path = os.path.join(label_path, 'L' + file_id)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if cv2.countNonZero(mask) > 0:
            non_empty_ids.append(file_id)

    # Save the filtered IDs to another CSV file
    df_non_empty = pd.DataFrame(non_empty_ids, columns=["file_ids"])
    df_non_empty.sort_values(by='file_ids', inplace=True)  # Sort the DataFrame by 'file_ids'
    df_non_empty.to_csv(filtered_ids_csv, index=False)

# Example usage:
# label_path = "/path_to_labels"
# output_features_path = "/path_to_features"
# filter_non_empty_masks(label_path, output_features_path)



# %% dataset
class SegmentationDataset(Dataset):
    def __init__(self, csv_file, bbox_shift=20):
        self.df = pd.read_csv(csv_file)
        self.ids = self.df["file_ids"]
        self.img_path = "data/features/"
        self.mask_path = "data/segmentation_maps/"
        self.bbox_shift = bbox_shift
        print(f"number of images: {len(self.ids)}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        # Load image and mask using the ID from the CSV
        img_name = f"F{self.ids[index]}"
        mask_name = f"L{self.ids[index]}"

        # Load and resize image to 1024x1024, then convert to RGB
        img = Image.open(join(self.img_path, img_name)).resize((1024, 1024)).convert("RGB")
        img = np.array(img)  # Convert image to numpy array

        img = img / 255.0

        # Load and resize mask to 1024x1024
        mask = Image.open(join(self.mask_path, mask_name)).resize((1024, 1024))
        mask = np.array(mask)  # Convert mask to numpy array

        # Convert the shape to (3, H, W) for image and (1, H, W) for mask
        img = np.transpose(img, (2, 0, 1))
        mask = np.expand_dims(mask, axis=0)  # Add an extra dimension for the channel

        label_ids = np.unique(mask)[1:]
        mask_binary = np.uint8(mask == random.choice(label_ids.tolist()))[1]  # only one label, (1024, 1024)


        y_indices, x_indices = np.where(mask_binary > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = mask_binary.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        bboxes = np.array([x_min, y_min, x_max, y_max])

        return (
            torch.tensor(img).float(),
            torch.tensor(mask_binary[None, :, :]).long(),
            torch.tensor(bboxes).float(),
            img_name,
        )


# %% sanity check
tr_dataset = SegmentationDataset(csv_file='file_ids.csv',)
tr_dataloader = DataLoader(tr_dataset, batch_size=4, shuffle=True)

for step, (image, mask_binary, bboxes, img_name) in enumerate(tr_dataloader):
    print(image.shape, mask_binary.shape, bboxes.shape)
    # show the example
    _, axs = plt.subplots(1, 2, figsize=(25, 25))
    idx = random.randint(0, image.size(0) - 1)  # Update this line to get a valid index
    axs[0].imshow(image[idx].cpu().permute(1, 2, 0).numpy())
    show_mask(mask_binary[idx].cpu().numpy()[0], axs[0])  # Passing the 2D mask to show_mask
    show_box(bboxes[idx].numpy(), axs[0])
    axs[0].axis("off")
    # set title
    axs[0].set_title(img_name[idx])
    idx = random.randint(0, image.size(0) - 1)  # Update this line to get a valid index
    axs[1].imshow(image[idx].cpu().permute(1, 2, 0).numpy())
    show_mask(mask_binary[idx].cpu().numpy()[0], axs[1])  # Passing the 2D mask to show_mask
    show_box(bboxes[idx].numpy(), axs[1])
    axs[1].axis("off")
    # set title
    axs[1].set_title(img_name[idx])
    # plt.show()
    plt.subplots_adjust(wspace=0.01, hspace=0)
    plt.savefig("./data_sanitycheck.png", bbox_inches="tight", dpi=300)
    plt.close()
    break


# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "--csv_file",
    type=str,
    default="data/nucleus_data/file_ids.csv",
    help="Path to the CSV file containing file IDs for the dataset."
)
parser.add_argument(
    "--bbox_shift",
    type=int,
    default=20,
    help="Bounding box shift value for data augmentation."
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
parser.add_argument("-task_name", type=str, default="CellSAM-ViT-B")
parser.add_argument("-model_type", type=str, default="vit_b")
parser.add_argument(
    "-checkpoint", type=str, default="work_dir/CellSAM/sam_vit_b_01ec64.pth"
)
# parser.add_argument('-device', type=str, default='cuda:0')
parser.add_argument(
    "--load_pretrain", type=bool, default=True, help="use wandb to monitor training"
)
parser.add_argument("-pretrain_model_path", type=str, default="")
parser.add_argument("-work_dir", type=str, default="./work_dir")
# train
parser.add_argument("-num_epochs", type=int, default=1000)
parser.add_argument("-batch_size", type=int, default=2)
parser.add_argument("-num_workers", type=int, default=0)
# Optimizer parameters
parser.add_argument(
    "-weight_decay", type=float, default=0.01, help="weight decay (default: 0.01)"
)
parser.add_argument(
    "-lr", type=float, default=0.0001, metavar="LR", help="learning rate (absolute lr)"
)
parser.add_argument(
    "-use_wandb", type=bool, default=False, help="use wandb to monitor training"
)
parser.add_argument("-use_amp", action="store_true", default=False, help="use amp")
parser.add_argument(
    "--resume", type=str, default="", help="Resuming training from checkpoint"
)
parser.add_argument("--device", type=str, default="cuda:0")
args = parser.parse_args()


# %% set up wandb
if args.use_wandb:
    import wandb

    wandb.login()
    wandb.init(
        project=args.task_name,
        config={
            "lr": args.lr,
            "batch_size": args.batch_size,
            "data_path": args.tr_npy_path,
            "model_type": args.model_type,
        },
    )


# %% set up model
run_id = datetime.now().strftime("%Y%m%d-%H%M")
model_save_path = join(args.work_dir, args.task_name + "-" + run_id)
device = torch.device(args.device)

# %% model
class CellSAM(nn.Module):
    def __init__(
        self,
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

    def forward(self, image, box):
        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]  # (B, 1, 4)

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )


        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks
    

# %% main
def main():
    os.makedirs(model_save_path, exist_ok=True)
    shutil.copyfile(
        __file__, join(model_save_path, run_id + "_" + os.path.basename(__file__))
    )

    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    cellsam_model = CellSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)
    cellsam_model.train()

    print(
        "Number of total parameters: ",
        sum(p.numel() for p in cellsam_model.parameters()),
    )  # 93735472
    print(
        "Number of trainable parameters: ",
        sum(p.numel() for p in cellsam_model.parameters() if p.requires_grad),
    )  # 4058340

    img_mask_encdec_params =  cellsam_model.mask_decoder.parameters()
    
    optimizer = torch.optim.AdamW(
        img_mask_encdec_params, lr=args.lr, weight_decay=args.weight_decay
    )
    print(
        "Number of image encoder and mask decoder parameters: ",
        sum(p.numel() for p in img_mask_encdec_params if p.requires_grad),
    )  # 93729252
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    # cross entropy loss
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    # %% train
    num_epochs = args.num_epochs
    iter_num = 0
    losses = []
    best_loss = 1e10
    train_dataset = SegmentationDataset(args.csv_file, bbox_shift=args.bbox_shift)

    print("Number of training samples: ", len(train_dataset))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            ## Map model to be loaded to specified single GPU
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint["epoch"] + 1
            cellsam_model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        for step, (image, gt2D, boxes, _) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            boxes_np = boxes.detach().cpu().numpy()
            image, gt2D = image.to(device), gt2D.to(device)
            if args.use_amp:
                ## AMP
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    cellsam_pred = cellsam_model(image, boxes_np)
                    loss = seg_loss(cellsam_pred, gt2D) + ce_loss(
                        cellsam_pred, gt2D.float()
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                cellsam_pred = cellsam_model(image, boxes_np)
                loss = seg_loss(cellsam_pred, gt2D) + ce_loss(cellsam_pred, gt2D.float())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            iter_num += 1

        epoch_loss /= step
        losses.append(epoch_loss)
        if args.use_wandb:
            wandb.log({"epoch_loss": epoch_loss})
        print(
            f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}'
        )
        ## save the latest model
        checkpoint = {
            "model": cellsam_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(checkpoint, join(model_save_path, "cellsam_model_latest.pth"))
        ## save the best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            checkpoint = {
                "model": cellsam_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, join(model_save_path, "cellsam_model_best.pth"))

        # %% plot loss
        plt.plot(losses)
        plt.title("Dice + Cross Entropy Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(join(model_save_path, args.task_name + "train_loss.png"))
        plt.close()


if __name__ == "__main__":
    main()
