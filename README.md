<div align="center">

# 🎭 CMV: Change My Voice

## ⚡️ Fast, Lightweight, Real-Time VC

</div>
 
## Description   
What it does

## Setup

### 1. Install Dependencies

```bash
# clone project
git clone https://github.com/puru-samal/cmv.git

# [Optional] Create a new conda environment
conda create -n your-env-name python=3.12
conda activate your-env-name

# install project
cd cmv
pip install -r requirements.txt
```

### 2. Download Dataset

To download and validate the **LibriTTS** dataset, simply run:

```bash
python src/setup/LibriTTS_extract.py
```

This will:

- Create a new `data/` directory (if it doesn’t already exist)
- Asynchronously download the entire LibriTTS dataset
- Perform basic integrity checks (via checksums)

> ⚙️ Configuration:
> You can modify the download settings (e.g., target directory, concurrency, number of retries, chunk size) in the config file at: `configs/setup/LibriTTS_download.yaml`

> ⚠️ Warning:
> The complete LibriTTS dataset is very large (~79 GB total).
> Make sure you have sufficient disk space and a stable internet connection before running the script.

The final directory structure will look like this:

```
data/
└── LibriTTS/
    ├── dev-clean.tar.gz
    ├── dev-other.tar.gz
    ├── test-clean.tar.gz
    ├── test-other.tar.gz
    ├── train-clean-100.tar.gz
    ├── train-clean-360.tar.gz
    └── train-other-500.tar.gz
```

### 3. Extract `tar.gz` files

## Imports

This project is setup as a package which means you can now easily import any file into any other file like so:

```python
from project.datasets.mnist import mnist
from project.lit_classifier_main import LitClassifier
from pytorch_lightning import Trainer

# model
model = LitClassifier()

# data
train, val, test = mnist()

# train
trainer = Trainer()
trainer.fit(model, train, val)

# test using the best model!
trainer.test(test_dataloaders=test)
```

### Citation

```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```
