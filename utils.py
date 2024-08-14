import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image


def create_df(image_path):
    """Creates Dataframe with id of images without extensions

    Args:
        image_path (_type_): path to images folder

    Returns:
        pandas.DataFrame: Dataframe with id of images without extensions
    """
    name = []
    mask = []
    for dirname, _, filenames in os.walk(
        image_path
    ):  # given a directory iterates over the files
        for filename in filenames:
            f = filename.split(".")[0]
            name.append(f)

    return (
        pd.DataFrame({"id": name}, index=np.arange(0, len(name)))
        .sort_values("id")
        .reset_index(drop=True)
    )


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def check_accuracy(loader, model, num_classes=5, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            preds = model(x)
            preds = F.softmax(preds, dim=1)  # Применяем softmax по каналу классов
            preds = torch.argmax(
                preds, dim=1
            )  # Выбираем класс с наибольшей вероятностью

            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)

            # Вычисляем Dice score для каждого класса и усредняем
            dice = 0
            for cls in range(num_classes):
                pred_mask = (preds == cls).float()
                true_mask = (y == cls).float()
                dice += (2 * (pred_mask * true_mask).sum()) / (
                    (pred_mask + true_mask).sum() + 1e-8
                )
            dice_score += dice / num_classes

    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()


def save_images_in_grid(images, grid_size, image_size, save_path):
    # Create a new image with the size of the grid
    grid_img = Image.new(
        "RGB", (grid_size[1] * image_size[1], grid_size[0] * image_size[0])
    )

    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            idx = i * grid_size[1] + j
            if idx < len(images):
                img = images[idx]
                grid_img.paste(img, (j * image_size[1], i * image_size[0]))

    grid_img.save(save_path)


def save_predictions_as_imgs(
    loader,
    model,
    img_width,
    img_height,
    folder="saved_images/",
    device="cuda",
    max_rows=4,
    columns=3,
):
    color_palette = [
        155,
        38,
        182,  # Class 0: obstacles
        14,
        135,
        204,  # Class 1: water
        124,
        252,
        0,  # Class 2: nature
        255,
        20,
        147,  # Class 3: moving
        169,
        169,
        169,  # Class 4: landable
    ]

    model.eval()
    os.makedirs(folder, exist_ok=True)  # Ensure folder exists

    idx = 0
    img_count = 0
    total_images = max_rows * columns
    fig, axes = plt.subplots(max_rows, columns, figsize=(15, 5 * max_rows))
    axes = axes.flatten()

    for x, y in loader:
        x = x.to(device=device)
        with torch.no_grad():
            preds = model(x)  # Get model predictions
            preds = torch.argmax(preds, dim=1)  # Convert to class indices

        preds = preds.cpu().numpy()  # Move to CPU and convert to numpy array
        y = y.cpu().numpy()  # Move to CPU and convert to numpy array

        for i in range(x.shape[0]):  # Iterate over batch
            orig_img = x[i].cpu().numpy().transpose(1, 2, 0)
            orig_img = (orig_img - orig_img.min()) / (
                orig_img.max() - orig_img.min()
            )  # Normalize

            pred_img = preds[i]
            pred_img = Image.fromarray(pred_img.astype(np.uint8))
            pred_img.putpalette(color_palette)
            pred_img = np.array(pred_img)

            mask_img = y[i]
            mask_img = Image.fromarray(mask_img.astype(np.uint8))
            mask_img.putpalette(color_palette)
            mask_img = np.array(mask_img)

            if img_count >= total_images:
                plt.tight_layout()
                plt.savefig(f"{folder}/grid_{idx}.png")
                plt.close(fig)
                idx += 1
                fig, axes = plt.subplots(max_rows, columns, figsize=(15, 5 * max_rows))
                axes = axes.flatten()
                img_count = 0

            # Ensure we don't exceed the bounds of axes
            if img_count < len(axes):
                axes[img_count].imshow(orig_img)
                axes[img_count].set_title("Original")
                axes[img_count].axis("off")

                axes[img_count + 1].imshow(mask_img)
                axes[img_count + 1].set_title("Original Mask")
                axes[img_count + 1].axis("off")

                axes[img_count + 2].imshow(pred_img)
                axes[img_count + 2].set_title("Predicted Mask")
                axes[img_count + 2].axis("off")

                img_count += columns  # Move to next row set

    if img_count > 0:
        plt.tight_layout()
        plt.savefig(f"{folder}/grid_{idx}.png")
        plt.close(fig)

    model.train()
