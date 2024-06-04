# **Toward Robustness of Arabic AI-Generated Text Detection: Tackling Diacritics Challenges**


## Overview

This repository contains the code and datasets for detecting AI-generated texts (AIGTs) and human-written texts (HWTs) in Arabic. Our focus is on addressing the challenges posed by diacritics in Arabic text using various transformer-based models, including AraELECTRA, AraBERT, XLM-R, and mBERT. The model architecture is adaptable to any language, although our primary focus is on Arabic due to the unique challenges it presents.

## Datasets

The `Dataset` directory includes the datasets used in our experiments. These datasets are prepared with Training, Validation, and Testing splits to facilitate model training and evaluation:
- **Custom Max**: The largest combined dataset of CustomPlus and Custom datasets.
- **CustomPlus**: Includes 9% diacritized religious texts to incorporate some diacritics during training.
- **Enhanced Diacritized Custom Dataset**: Dataset with examples duplicated to be half diacritized and half non-diacritized.
- **Not Enhanced Diacritized Custom Dataset**: Not used in the training, but attached due to make the reader aware of the enhanced Diacritized Custom dataset preprocessing.
- **Pure Custom Dataset**: Dataset containing only non-diacritized texts which is half of Diacritized Custom dataset.
- **Religious Diac Dataset**: Focuses on religious texts with a higher frequency of diacritics. It includes HWTs from religious texts and AIGTs from various fields.

## Design

Our models are designed to handle the complexities of diacritics in Arabic text, but the architecture is flexible enough to be adapted for other languages. We use transformer-based pre-trained models and fine-tune them on our custom datasets to enhance their ability to distinguish between AIGTs and HWTs. Additionally, we have another code branch called **Diacritics-free**, which implements a dediacritization filter as a preprocessing step for the dataset. This branch is focused on improving model performance by removing diacritics before training and evaluation.

## Results

All narrative results, logs, and performance metrics are included in the `Experiments Details and Detection Models` directory. Each experiment’s results and the corresponding models are provided in the `Results.txt` file, which includes links to download the model checkpoints and their outputs.


### Installation

To set up the environment, install the required packages using the `requirements.txt` file:
```sh
pip install -r requirements.txt
```

## Usage

To run the training and evaluation process, ensure that your dataset directory contains `Training.csv`, `Validation.csv`, and `Testing.csv` files. 

For inference, use the `Evaluator.py` module. This module is dedicated to evaluating model performance and requires the model checkpoint files, which can be downloaded from the `Experiments Details and Detection Models` directory.
## Running the Model

1. **Train the Model**:

    ```sh
    python main.py
    ```

2. **Evaluate the Model**:

    Ensure the `check_point` folder is available in the project directory and contains the `best_weight.pt` file of the model that matches the `config.json` file's `model_name`. Then run:

    ```sh
    python Evaluator.py
    ```

