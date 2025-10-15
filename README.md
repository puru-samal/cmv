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
pip install -e .
pip install -r requirements.txt
```

### 2. Download Dataset

Next step would involve downloading and pre-processing the dataset. To make training feasible, we train and evaluate on the `clean` subset of LibriTTS. The

```bash
# module folder
cd project

# run module (example: mnist as your main contribution)
python lit_classifier_main.py
```

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
