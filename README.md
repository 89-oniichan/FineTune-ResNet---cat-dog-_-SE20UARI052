# Fine-Tuning ResNet50 for Cat and Dog Classification

This repository contains code for fine-tuning a pre-trained ResNet50 model to classify images of cats and dogs using TensorFlow.

## Prerequisites

Before running the code, ensure that you have the required dependencies installed:

- Python (3.x recommended)
- TensorFlow
- TensorFlow Datasets

You can install these packages using `pip`:


## Logic Used

The code follows the following logic:

1. **Data Loading and Preprocessing**:
   - The code loads the "cats_vs_dogs" dataset using TensorFlow Datasets.
   - It resizes the images to the expected input size of the ResNet50 model (224x224).

2. **Data Batching and Shuffling**:
   - The data is shuffled and batched to create training, validation, and test datasets.

3. **Base Model Creation**:
   - A pre-trained ResNet50 model is used as the base model. The last few layers of this model are unfrozen for fine-tuning.

4. **Custom Layers for Classification**:
   - Custom layers are added to the base model for binary classification (cat or dog).

5. **Model Compilation**:
   - The model is compiled with a lower learning rate for fine-tuning.

6. **Training**:
   - The model is trained, including fine-tuning. The code allows for specifying the number of initial epochs and fine-tuning epochs.

7. **Model Evaluation**:
   - The trained model is evaluated on the test dataset, and accuracy is reported.

