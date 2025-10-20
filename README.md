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

## 🧩 Dataset Setup

### 1️⃣ Download the Dataset

To download and validate the **LibriTTS** dataset, run:

```bash
python src/setup/LibriTTS_download.py
```

This will:
- 🗂️ Create a data/ directory (if it doesn’t exist)
- ⚡ Asynchronously download all LibriTTS archive files
- 🔍 Verify file integrity via checksums

> ⚙️ Config:
> Adjust settings (target dir, concurrency, retries, chunk size) in `configs/setup/LibriTTS_download.yaml`

> ⚠️ Warning:
> The full LibriTTS dataset is **~79 GB** — ensure you have enough disk space and a stable internet connection.

<details> 
<summary>📁 <strong>Expected Directory Structure</strong></summary>
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
</details>

### 2️⃣ Extract Archives

Once downloaded, extract all `.tar.gz` files:

```bash
python src/setup/LibriTTS_download.py
```

This will:
- 🧩 Parallelly extract all archives into the data/ directory
- 🧹 Optionally delete archives and retry failed extractions

> ⚙️ Config:
> Control which files to extract, number of retries, and output path in `configs/setup/LibriTTS_extract.yaml`

> ⚠️ Reminder:
> Extraction also requires substantial free space (~79 GB).

<details> 
<summary>📂 <strong>Resulting Directory Structure</strong></summary>
```
data/
└── LibriTTS/
    ├── dev-clean
    ├── dev-other
    ├── test-clean
    ├── test-other
    ├── train-clean-100
    ├── train-clean-360
    ├── train-other-500
    ├── BOOKS.txt
    ├── CHAPTERS.txt
    ├── eval_sentences10.tsv
    ├── LICENSE.txt
    ├── NOTE.txt
    ├── reader_book.tsv
    ├── README_librispeech.txt
    ├── README_libritts.txt
    ├── speakers.tsv
    └── SPEAKERS.txt
```
</details>

3️⃣ Resample to 16 kHz

**LibriTTS** audio is originally at `24 kHz`. To ensure compatibility with the `HuBERT` feature extractor and improve efficiency, resample to `16 kHz`:

```bash
python src/setup/LibriTTS_convert.py
```

This will:
- 🔄 Parallelly convert and overwrite all .wav files to 16 kHz
- ♻️ Retry failed conversions automatically

> ⚙️ Config:
> Adjust folders and retries in `configs/setup/LibriTTS_convert.yaml`

> ⚠️ Reminder:
> Resampling is I/O heavy — allow time and ensure you have sufficient space.

✅ After conversion, the directory structure remains unchanged.

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
