from __future__ import annotations
import itertools
from typing import Sequence
from torch import Tensor
from matplotlib.colors import ListedColormap

#!/usr/bin/env python
import io
import os
from pathlib import Path

from PIL import Image, ImageTk
import PySimpleGUI as sg
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import seaborn as sns

__all__ = ["visualize_clusters", "visualize_merging"]


def visualize_clusters(
    x: npt.NDArray[np.floating] | Tensor,
    *,
    labels: npt.NDArray[np.number] | Tensor,
    title: str | None = None,
    legend: bool = True,
) -> plt.Figure:
    if x.shape[1] != 2:
        raise ValueError("Cluster-visualization can only be performed for 2-dimensional inputs.")
    if isinstance(x, Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(labels, Tensor):
        labels_ls = labels.detach().cpu().long().tolist()
    else:
        labels_ls = labels.astype("uint").tolist()

    classes = np.unique(labels)
    num_classes = len(classes)
    fig, ax = plt.subplots(dpi=100, figsize=(6, 6))
    cmap = ListedColormap(sns.color_palette("bright", num_classes).as_hex())  # type: ignore
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=labels_ls, cmap=cmap)
    ax.set_xticks([])
    ax.set_yticks([])
    sns.despine(left=True, bottom=True, right=True)


    if legend:
        def _flip(items: Sequence, ncol: int):
            return itertools.chain(*[items[i::ncol] for i in range(ncol)])

        plt.legend(
            handles=_flip(sc.legend_elements()[0], 5),
            labels=_flip(classes.tolist(), 5),
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.3, -0.03),
            ncol=5,
        )

    if title is not None:
        ax.set_title(title)

    return fig


def visualize_merging(image_dir: Path | str) -> None:
    if isinstance(image_dir, str):
        image_dir = Path(image_dir)
    # We assume the image directory follows a particular structure
    gt_image_path = image_dir / "ground_truth.png"
    pred_image_dir = image_dir / "predicted"

    pred_image_paths: list[Path] = []
    for ext in ("jpg", "jpeg", "png"):
        # Glob images from child folders recusrively, excluding hidden files
        pred_image_paths.extend(pred_image_dir.glob(f"**/[!.]*.{ext}"))

    num_files = len(pred_image_paths)  # number of iamges found
    del pred_image_paths

    if num_files == 0:
        sg.popup("No files in image directory.")
        raise SystemExit()

    def get_img_data(f, maxsize=(1200, 850), first=False):
        """Generate image data using PIL"""
        img = Image.open(f)
        img.thumbnail(maxsize)
        if first:  # tkinter is inactive the first time
            bio = io.BytesIO()
            img.save(bio, format="PNG")
            del img
            return bio.getvalue()
        return ImageTk.PhotoImage(img)

    # make these 2 elements outside the layout as we want to "update" them later
    # initialize to the first file in the list
    image_path = pred_image_dir / "0.png"
    image_elem = sg.Image(data=get_img_data(image_path, first=True))
    gt_image_elem = sg.Image(data=get_img_data(gt_image_path, first=True))

    sg.theme("Material2")
    # define layout, show and read the form
    col = [[image_elem, gt_image_elem]]
    col_files = [
        [
            sg.Text(u"\u03C4", key="_V2_", font=("Helvetica", 20), justification="center"),
            sg.Slider(
                range=(0, num_files - 1),
                default_value=0,
                size=(20, 10),
                orientation="v",
                key="-SLIDER-THRESHOLD-",
                enable_events=True,
                disable_number_display=True
            ),
        ],
    ]
    layout = [[sg.Column(col_files), sg.Column(col)]]

    window = sg.Window(
        "Merging visualization",
        layout=layout,
        return_keyboard_events=True,
        location=(0, 0),
        use_default_focus=False,
    )

    # loop reading the user input and displaying image, filename
    while True:
        # read the form
        event, values = window.read()  # type: ignore
        if event in ("Exit", None):
            break
        image_idx = int(values["-SLIDER-THRESHOLD-"])  # type: ignore
        # update window with new image
        image_path = pred_image_dir / f"{image_idx}.png"
        image_elem.update(data=get_img_data(image_path, first=True))
    window.close()
