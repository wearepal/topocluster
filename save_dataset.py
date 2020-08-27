"""Simply call the main function"""
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from PIL import Image

from topocluster.configs import BaseArgs
from topocluster.data.data_loading import load_dataset
from topocluster.optimisation import get_class_id

if __name__ == "__main__":
    args = BaseArgs().parse_args()
    if args.dataset not in ("celeba", "cmnist"):
        raise ValueError("Dataset must be an image-dataset.")

    datasets = load_dataset(args)
    s_count = max(2, datasets.s_dim)

    subsets = {"train": datasets.train, "context": datasets.context, "test": datasets.test}
    save_dir = Path(args.save_dir) / args.dataset
    save_dir.mkdir(parents=True, exist_ok=True)

    for name, subset in subsets.items():
        subset_dir = save_dir / name
        subset_dir.mkdir(exist_ok=True)
        image_dir = subset_dir / "images"
        image_dir.mkdir(exist_ok=True)

        data: Dict[str, List[float]] = {key: [] for key in ["x", "s", "y", "cluster_id"]}

        for index, (x, s, y) in enumerate(subset):
            filename = f"{index}.jpg"
            x = x.permute(1, 2, 0).numpy()
            if args.dataset == "celeba":
                x = (x + 1) / 2
            x *= 255
            Image.fromarray(x.astype(np.uint8)).save(image_dir / filename)
            data["x"].append(f"images/{filename}")
            data["s"].append(s.item())
            data["y"].append(y.item())
            data["cluster_id"].append(
                get_class_id(s=s, y=y, s_count=s_count, to_cluster="both").item()
            )
        data = pd.DataFrame(data)
        data.to_csv(subset_dir / "data.csv")