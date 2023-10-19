# -*- coding: utf-8 -*-
import torch

# %% convert cellsam model checkpoint to sam checkpoint format for convenient inference
sam_ckpt_path = ""
cellsam_ckpt_path = ""
save_path = ""
multi_gpu_ckpt = True  # set as True if the model is trained with multi-gpu

sam_ckpt = torch.load(sam_ckpt_path)
cellsam_ckpt = torch.load(cellsam_ckpt_path)
sam_keys = sam_ckpt.keys()
for key in sam_keys:
    if not multi_gpu_ckpt:
        sam_ckpt[key] = cellsam_ckpt["model"][key]
    else:
        sam_ckpt[key] = cellsam_ckpt["model"]["module." + key]

torch.save(sam_ckpt, save_path)
