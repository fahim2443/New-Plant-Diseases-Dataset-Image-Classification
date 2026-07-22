# New Plant Diseases Dataset Image Classification

This project trains a deep learning model to classify plant leaf images into disease categories using a custom Convolutional Neural Network (CNN) built with PyTorch. It is based on the "New Plant Diseases Dataset" and includes notebook-based training, checkpointing, class labeling, and image prediction workflows.

## Project Overview

The goal of this project is to identify plant diseases from leaf images and provide a predicted class along with a confidence score. The implementation uses a CNN trained on a dataset of plant disease images split into training, validation, and test folders.

## Dataset

The dataset is organized as follows:

- `datasets/train/` — training images
- `datasets/valid/` — validation images
- `datasets/test/` — test images used for inference

Each subfolder inside the training directory represents a class (for example, a specific plant disease or healthy condition). The notebook automatically creates a `classes.json` file from the folder names, which are used later for predictions.

## Model Architecture

The model is a custom PyTorch CNN with the following structure:

- Input: RGB images resized to `224 x 224`
- Convolution layers:
  - `Conv2d(3 -> 16)`
  - `Conv2d(16 -> 32)`
  - `Conv2d(32 -> 64)`
- Max pooling after each convolution block
- Flattening layer
- Fully connected layer with 512 units
- Dropout layer (`0.6`)
- Output layer with `38` classes

## Image Preprocessing

The notebook applies the following transforms before feeding images into the model:

- Resize to `(224, 224)`
- Convert to tensor
- Normalize using ImageNet-style mean and standard deviation:
  - Mean: `[0.485, 0.456, 0.406]`
  - Std: `[0.229, 0.224, 0.225]`

## Training Details

The training notebook uses:

- Framework: PyTorch
- Batch size: `32`
- Epochs: `10`
- Loss function: `CrossEntropyLoss`
- Optimizer: `Adam`
- Learning rate: `0.001`
- Weight decay: `1e-4`

The model is trained on GPU if available; otherwise, it uses CPU.

## Model Checkpointing

During training, the notebook monitors validation accuracy after each epoch and saves the best-performing model weights to:

- `best_model.pth`

This checkpoint is used later for inference.

## Results

The model was trained for 10 epochs on the plant disease dataset and achieved strong validation performance.

- Classes: 38
- Training images: 70,295
- Validation images: 17,572
- Test images: 33
- Best validation accuracy: 94.53%
- Best epoch: 9

The best checkpoint was saved at epoch 9 and stored in `best_model.pth`. The notebook also shows sample predictions on test images, with confidence scores ranging from about 65% to 99.99%.

## Inference

The notebook also includes a prediction pipeline that:

1. Loads the trained model from `best_model.pth`
2. Loads class names from `classes.json`
3. Applies the same preprocessing transform
4. Runs the model on an input image
5. Returns:
   - predicted class name
   - confidence score

## Files

- `notebook.ipynb` — training and inference workflow
- `best_model.pth` — trained model weights
- `classes.json` — class label list
- `datasets/` — dataset folders

## Requirements

The notebook uses the following Python libraries:

- `torch`
- `torchvision`
- `PIL`
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `tensorflow`
- `torchsummary`

## How to Run

1. Install the required dependencies.
2. Open `notebook.ipynb` in Jupyter Notebook or VS Code.
3. Run the cells in order to:
   - load the dataset
   - define and train the CNN
   - save the best model
   - generate predictions on test images

## Notes

- The model is designed for `38` output classes.
- The project uses a simple custom CNN rather than a pretrained transfer learning model.
- The notebook is a good starting point for learning image classification with PyTorch and plant disease datasets.