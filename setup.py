from setuptools import find_packages, setup
import subprocess
import sys

setup(
    name="topocluster",
    version="0.1",
    author="",
    packages=find_packages(),
    description="Topological CLustering",
    python_requires=">=3.8",
    package_data={"topocluster": ["py.typed"]},
    install_requires=[
        "EthicML @ git+https://github.com/predictive-analytics-lab/EthicML.git",
        "gitpython",
        "lapjv",
        "numpy >= 1.15",
        "pandas >= 0.24",
        "pillow < 7.0",
        "pykeops",
        "scikit-image >= 0.14",
        "scikit-learn >= 0.20",
        "scipy >= 1.2.1",
        "torch >= 1.2",
        "torchvision >= 0.4.0",
        "tqdm >= 4.31",
        "typed-flags",
        "wandb == 0.8.27",
        "umap-learn",
    ],
)

try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "faiss-gpu"])
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "faiss-cpu"])
