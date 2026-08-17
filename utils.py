from google.colab import drive
import os
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import timm
from timm.data import resolve_model_data_config, create_transform, resolve_data_config
from timm.data.transforms_factory import create_transform
from tqdm.auto import tqdm
from tqdm import tqdm
import copy
import numpy as np
from sklearn.metrics import average_precision_score
import tarfile
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path
import zipfile
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re

def walk_through_dir(dir_path):
  """
  Walks through dir_path returning its contents.
  Args:
    dir_path (str or pathlib.Path): target directory

  Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
  """
  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")
    
def display_images_by_label(dataset, class_name, num_images=5):
    if class_name not in dataset.class_to_idx:
        print(f"Class '{class_name}' not found in the dataset.")
        return

    # Get the numeric index for the class
    target_idx = dataset.class_to_idx[class_name]

    # Find all indices in the dataset that match the target label
    matching_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == target_idx]

    num_to_display = min(num_images, len(matching_indices))
    selected_indices = random.sample(matching_indices, num_to_display)

    # plot the images
    # plt.figure(figsize=(15, 15))
    # for i, idx in enumerate(selected_indices):
    #     img, label = dataset[idx]
    #     plt.subplot(1, num_to_display, i + 1)
    #     plt.imshow(img)
    #     plt.title(f"{class_name}")
    #     plt.axis('off')
    # # plt.tight_layout()
    # plt.show()

    num_to_display = len(selected_indices)
    num_cols = 4
    num_rows = int(np.ceil(num_to_display / num_cols))

    plt.figure(figsize=(4 * num_cols, 4 * num_rows))

    for i, idx in enumerate(selected_indices):
        img, label = dataset[idx]

        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(img)
        plt.title(f"{class_name}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
    
def parse_coin_filename(path):
    path = Path(path)
    stem = path.stem

    type_match = re.search(r"cn_type_(\d+)", stem, flags=re.IGNORECASE)
    coin_match = re.search(r"cn_coin_(\d+)", stem, flags=re.IGNORECASE)
    side_match = re.search(r"_(rev|obv|reverse|obverse)(?:_|$)", stem, flags=re.IGNORECASE)

    type_id = type_match.group(1) if type_match else None
    coin_id = coin_match.group(1) if coin_match else None

    side = side_match.group(1).lower() if side_match else None
    if side == "reverse":
        side = "rev"
    elif side == "obverse":
        side = "obv"

    image_id = f"{coin_id}_{side}" if coin_id and side else stem

    return {
        "filename": path.name,
        "image_id": image_id,
        "coin_id": coin_id,
        "type_id": type_id,
        "side": side,
    }
    
class CoinMotifDataset(Dataset):
    def __init__(self, frame, motif_cols, transform=None):
        self.frame = frame.reset_index(drop=True)
        self.motif_cols = motif_cols
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]

        img = Image.open(row["image_path"]).convert("RGB")

        if self.transform:
            img = self.transform(img)

        y = torch.tensor(row[self.motif_cols].values.astype("float32"))

        return img, y
        
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = model.to(device)

def count_trainable(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")

def macro_ap(y_true, y_prob, motif_cols):
    scores = []
    for j, motif in enumerate(motif_cols):
        if y_true[:, j].sum() > 0:
            scores.append(average_precision_score(y_true[:, j], y_prob[:, j]))
    return float(np.mean(scores)) if scores else np.nan

def train_one_epoch(model, loader, optimizer, criterion, device, epoch, phase):
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc=f"{phase} epoch {epoch}", leave=False)

    for imgs, y in pbar:
        imgs = imgs.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(imgs)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader.dataset)

@torch.no_grad()
def predict(model, loader, device, desc="Validate"):
    model.eval()

    probs_all = []
    y_all = []

    pbar = tqdm(loader, desc=desc, leave=False)

    for imgs, y in pbar:
        imgs = imgs.to(device, non_blocking=True)

        logits = model(imgs)
        probs = torch.sigmoid(logits).cpu()

        probs_all.append(probs)
        y_all.append(y.cpu())

    return torch.cat(probs_all).numpy(), torch.cat(y_all).numpy()

def run_training_phase(model, train_loader, val_loader, optimizer, criterion, device,
                       num_epochs, phase, best_score=-np.inf, best_state=None):
    for epoch in tqdm(range(1, num_epochs + 1), desc=phase):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            phase=phase,
        )

        val_probs, val_y = predict(
            model=model,
            loader=val_loader,
            device=device,
            desc=f"{phase} validation {epoch}",
        )

        val_ap = macro_ap(val_y, val_probs, sample_motifs)

        print(f"{phase} epoch {epoch}: train_loss={train_loss:.4f}, val_macro_AP={val_ap:.4f}")

        if val_ap > best_score:
            best_score = val_ap
            best_state = copy.deepcopy(model.state_dict())
            print(f"  New best val_macro_AP: {best_score:.4f}")

    return best_score, best_state

# def parse_page(start):

#     url = BASE_URL.format(start)

#     r = session.get(url, timeout=30)
#     r.raise_for_status()

#     soup = BeautifulSoup(r.text, "html.parser")

#     page_records = []

#     for coin in soup.select("div.result-doc"):

#         # records ID

#         h4 = coin.find("h4")
#         if h4 is None:
#             continue

#         a = h4.find("a")
#         if a is None:
#             continue

#         record_id = a["href"].split("/")[-1]

#         # gets image

#         image_url = None

#         thumb = coin.select_one("a.thumbImage")

#         if thumb is not None:
#             image_url = thumb.get("href")

#         # gets metadata

#         metadata = {}

#         for dt, dd in zip(
#             coin.find_all("dt"),
#             coin.find_all("dd")
#         ):
#             metadata[
#                 dt.get_text(strip=True)
#             ] = dd.get_text(" ", strip=True)

#         page_records.append({

#             "RecordId": record_id,

#             "ImageURL": image_url,

#             "Obverse": metadata.get("Obverse"),

#             "Reverse": metadata.get("Reverse"),

#             "Date": metadata.get("Date"),

#             "Denomination": metadata.get("Denomination"),

#             "Weight": metadata.get("Weight (in g)")
#         })

#     return page_records

# # downloads image
# def download_image(url, filename):

#     if url is None:
#         return False

#     try:

#         r = session.get(url, timeout=30)

#         if r.status_code != 200:
#             return False

#         with open(filename, "wb") as f:
#             f.write(r.content)

#         return True

#     except Exception:

#         return False
        
def open_numismatics_data(csv_path: Path, images_path: Path) -> pd.DataFrame:
  df = pd.read_csv(csv_path)
  df.rename(columns={"Image": "filename"}, inplace=True)
  # drops weights column, which is all na.
  df.dropna(axis='columns', how='all', inplace=True)
  # drops any rows with no image.
  df.dropna(axis='rows', subset="filename", inplace=True)

  df["image_path"] = images_path
  df["image_path"] = df["image_path"] / df["filename"]

  return df

def check_for_motifs(df: pd.DataFrame, motifs: list[str], column_to_check: str = "Reverse") -> pd.DataFrame:
  if column_to_check in df.columns:
    # adds all columns at once instead of one by one to avoid errors
    motif_columns = pd.DataFrame(columns=motifs, dtype=str)
    result = pd.concat([df, motif_columns], axis='columns')
    for motif in motifs:
      result[motif] = result[column_to_check].str.contains("(?i)" + motif).astype(int)
    return result
  else:
    raise KeyError(f"DataFrame {df} has no column {column_to_check}.")

@torch.no_grad()
def show_top_predictions(model, dataset, motif_cols, device, n=12, top_k=3):
    model.eval()

    indices = np.random.choice(len(dataset), size=min(n, len(dataset)), replace=False)

    num_cols = 4
    num_rows = int(np.ceil(len(indices) / num_cols))

    plt.figure(figsize=(4.5 * num_cols, 4.8 * num_rows))

    for plot_i, idx in enumerate(indices):
        img_tensor, y_true = dataset[idx]

        x = img_tensor.unsqueeze(0).to(device)
        logits = model(x)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        y_true = y_true.numpy()

        true_labels = [motif_cols[j] for j, v in enumerate(y_true) if v == 1]

        top_idxs = np.argsort(probs)[::-1][:top_k]
        top_preds = [f"{motif_cols[j]}:{probs[j]:.2f}" for j in top_idxs]

        img = img_tensor.detach().cpu().permute(1, 2, 0).numpy()

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)

        plt.subplot(num_rows, num_cols, plot_i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(
            "True: " + (", ".join(true_labels) if true_labels else "none") +
            "\nTop: " + ", ".join(top_preds),
            fontsize=9,
        )

    # plt.tight_layout()
    plt.show()