---
language:
- en

library_name: tensorflow

tags:
  - computer-vision
  - image-classification
  - ai-generated-image-detection
  - cnn
  - cifake

datasets:
- CIFAKE

metrics:
- f1
- roc-auc

model-index:
  - name: AI Generated Image Detector
    results:
      - task:
          type: image-classification
          name: Image Classification (Real vs AI Generated)
        dataset:
          type: CIFAKE
          name: CIFAKE
          split: test
        metrics:
          - type: f1
            value:
            name: Test Macro F1
          - type: roc-auc
            value:
            name: Test ROC-AUC
---

# Model Card for AI Generated Image Detector

## Model Details

### Model Description

This model is a supervised binary image classification component designed to distinguish between real images and AI generated images. 

- **Model type:** Convolutional neural network (CNN)
- **Framework:** TensorFlow/Keras
- **Input:** RGB, 32×32
- **Model output:** Probability of `FAKE` from the sigmoid output.
- **Prediction output:** Binary class (`REAL` / `FAKE`) + probability of the predicted class.
- **Label encoding:** `REAL = 0`, `FAKE = 1`

#### Final Model Configuration

- **CNN architecture:**
  - **Architecture:** 2×Conv2D(32 filters) + Dense(64)
  - **Kernel size:** `TBD`
  - **Padding:** `TBD`
  - **Pooling:** 2×2; operation `TBD`
  - **Convolution stride:** 1
  - **Hidden activation:** ReLU
  - **Output activation:** Sigmoid

- **Training configuration:**
  - **Optimizer:** Adam
  - **Learning rate:** `TBD`
  - **Batch size:** `TBD`
  - **Dropout:** `TBD`
  - **Epochs:** 20

- **Developed by:** Maribel Preite, Luis Salinas, Rebeca Torrecilla, Aina Vila
- **Project:** Advanced Topics in Data Engineering II (TAED2), Universitat Politècnica de Catalunya (UPC)
- **License:** `TBD`

### Model Sources

- **Repository:** [GAN-Busters GitHub Repository](https://github.com/taed2-2627q1-gced-upc/taed2-GAN-Busters)
- **Dataset Card:** [CIFAKE Dataset Card](../data/dataset_card.md)
- **CIFAKE Dataset:** [CIFAKE on Kaggle](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
- **Reference baseline:** [CIFAKE Publication](https://ieeexplore.ieee.org/abstract/document/10409290) | [CIFAR-10 Hyperparameter Selection Study](https://researchonline.gcu.ac.uk/ws/portalfiles/portal/26022569/Paper103.pdf)
- **Demo/Application:** `TBD`

## Uses

### Direct Use

This model can be used to classify input images as `REAL` or `FAKE` and provide a probability estimate for the predicted class.

The model is integrated into an end-to-end MLOps pipeline, developed primarily for educational purposes to cover the entire machine learning lifecycle. AI generated image detection serves as the underlying machine learning task for the pipeline.

### Downstream Use

The model and pipeline may be reused and extended for further experimentation or development, with appropriate attribution to the original repository and implementation.

Potential downstream uses include:

- fine-tuning or retraining the model with additional or more diverse data
- experimenting with alternative model architectures or configurations
- extending or adapting the existing MLOps pipeline
- integrating the trained model into a larger application 

### Out-of-Scope Use

The model is not intended to provide definitive verification of image authenticity or to serve as a state of the art AI generated image detector. Its predictions should not be interpreted as conclusive evidence that an arbitrary image is real or AI generated.

## Training Details

### Training Data

The model is trained using the official CIFAKE training set containing 100,000 images. This set is further divided into fixed training and validation subsets:

- **Training subset:** 80,000 images (80%), consisting of 40,000 `REAL` and 40,000 `FAKE` images.
- **Validation subset:** 20,000 images (20%), consisting of 10,000 `REAL` and 10,000 `FAKE` images.

The split preserves both the target and semantic class distributions of the original training set and remains fixed across experiments to ensure consistent model comparison.

For details about CIFAKE's composition, provenance, semantic classes, and original splits, see the [Dataset Card](../data/dataset_card.md).

#### Preprocessing

The data pipeline performs the following preprocessing and validation steps:

- Load provided metadata or derive dataset information from the directory structure and filenames.
- Validate image paths, target labels (`REAL` / `FAKE`), semantic classes if available, and existing data splits.
- Check for duplicate image content across and within splits.
- Validate target and semantic class distributions.
- If no split is provided, generate one using the available class information for stratification.
- Convert non-RGB images to RGB.
- Center crop images when necessary to match the expected aspect ratio.
- Resize images to 32×32 using bilinear interpolation.
- Normalize RGB pixel values from `[0, 255]` to `[0, 1]`.

CIFAKE already provides valid and balanced train/test splits with uniform semantic class distributions, and all images are 32×32 RGB. The dataset therefore passes the corresponding validation checks without requiring resplitting, color conversion, cropping, or resizing. Pixel normalization is still applied before training to improve training stability.

### Training Configuration

Some parameters remain fixed based on the architectures investigated by [Bird & Lotfi (2024)](https://ieeexplore.ieee.org/abstract/document/10409290) and the findings by [Nazir, Patel & Patel (2018)](https://researchonline.gcu.ac.uk/ws/portalfiles/portal/26022569/Paper103.pdf) on hyperparameter tuning for a computer vision model trained on [CIFAR-10](https://cave.cs.toronto.edu/kriz/cifar.html).

- **Base architecture:** CNN with 2×Conv2D(32 filters) + Dense(64).
- **Input:** 32×32 RGB images with pixel values normalized to `[0, 1]`.
- **Loss function:** Binary cross entropy.
- **Optimizer:** Adam.
- **Epochs:** 20.
- **Model selection:** Macro F1 and ROC-AUC on the validation set are used to compare model configurations during experimentation.
- **Tracking:** MLflow for experiment tracking and CodeCarbon for emission tracking.

#### Phase 1: Architecture Selection

The first phase compares a predefined set of CNN configurations based on the architecture investigated by Bird & Lotfi (2024). The objective is to select the model architecture before further optimization of the training configuration.

Candidate configurations include variations in:

- **Kernel size:** `[3×3, 5×5]`
- **Padding:** `['valid', 'same']`
- **Pooling:** `['max', 'average']`, with 2×2 pooling size

Training hyperparameters remain fixed during Phase 1:

- **Learning rate:** `10⁻³`
- **Batch size:** `32`
- **Dropout:** `0`

The initial learning rate and batch size are informed by the results reported by [Nazir, Patel & Patel (2018)](https://researchonline.gcu.ac.uk/ws/portalfiles/portal/26022569/Paper103.pdf). Dropout is initially disabled because the baseline architecture is substantially simpler than the six-convolutional-layer architecture evaluated in their study. Regularization through dropout is therefore introduced and evaluated in Phase 2 only if overfitting is observed during Phase 1.

#### Phase 2: Hyperparameter Tuning

After selecting an architecture, the second phase evaluates training hyperparameters while keeping the selected model architecture fixed.

The experimental search includes:
- **Learning rate:** `[10⁻², 10⁻³, 10⁻⁴]`
- **Batch size:** `[32, 64, 128]`
- **Dropout:** `[0, 0.125, 0.25]`, evaluated if overfitting is observed during Phase 1

## Evaluation

### Evaluation data

#### CIFAKE Test Set

Final model performance is evaluated on the official CIFAKE test split, which remains isolated during model development.

The test set contains 20,000 images:
- **`REAL`:** 10,000 images
- **`FAKE`:** 10,000 images

The ten CIFAR-10 semantic categories are uniformly represented within both classes, with 1,000 images per semantic category.

For further details about the dataset and its predefined splits, see the [Dataset Card](../data/dataset_card.md).

#### External Evaluation

Additional external datasets may be used exclusively to evaluate model generalization beyond the CIFAKE distribution. These data are not used for training, validation, architecture selection, or hyperparameter tuning.

External evaluation may include images that differ from CIFAKE in terms of semantic content, original resolution, real image source, or generative model. 

### Factors

Model performance is evaluated across the following factors:

- **Semantic class:** Compare performance across the ten CIFAKE semantic classes.
- **Generative model:** Compare performance on synthetic images produced by generators not represented during training, when suitable external evaluation data are available.
- **Semantic domain:** Evaluate performance on visual content outside the ten CIFAKE semantic categories, when suitable external evaluation data are available.
- **Input preprocessing:** Evaluate external images with different original dimensions and aspect ratios to assess the effect of preprocessing to 32×32.

### Metrics

Model performance is evaluated primarily using **Macro F1** and **ROC AUC**, with class specific **precision** and **recall** providing additional information about classification behavior.

#### Macro F1

Macro F1 calculates the F1 score separately for `REAL` and `FAKE` classes and averages the two scores, giving both classes equal importance. Class predictions are obtained using a fixed decision threshold of `0.5`.

#### Precision and Recall

Precision and recall are reported separately for both `REAL` and `FAKE` to examine classification performance for each class.

- **Precision** measures how often predictions of a given class are correct.
- **Recall** measures how many images belonging to a given class are correctly identified.

Both metrics are calculated using the fixed `0.5` decision threshold.

#### ROC AUC

ROC AUC evaluates the model's ability to distinguish between `REAL` and `FAKE` using the predicted probabilities across all classification thresholds. Unlike Macro F1, precision, and recall, it is independent of the fixed `0.5` decision threshold.

### Results

#### CIFAKE Test Results

Final performance on the isolated CIFAKE test set:

- **Macro F1:** `TBD`
- **REAL precision:** `TBD`
- **REAL recall:** `TBD`
- **FAKE precision:** `TBD`
- **FAKE recall:** `TBD`
- **ROC-AUC:** `TBD`

Training and validation results from model experimentation, together with hyperparameters, learning curves, and relevant artifacts, are tracked and visualized using MLflow.

#### External Evaluation Results

Results from external evaluation datasets will be reported here if additional datasets are used to assess model generalization beyond the CIFAKE distribution.

## Bias, Risks, and Limitations

- **Generator dependency:** CIFAKE synthetic images were generated using [Stable Diffusion v1.4](https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf). The model may learn artifacts specific to this generator and may not generalize well to images produced by other generative models.

- **Limited semantic diversity:** CIFAKE is derived from the ten CIFAR-10 object/semantic classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. Performance may decrease for objects or visual content not represented during training.

- **Image representation:** The model operates on 32×32 RGB images. Inputs with different resolutions, aspect ratios, or color formats are preprocessed to match this representation, which may result in information loss and affect prediction quality.

- **Semantic distribution across generated splits:** When semantic class information is available, the pipeline can preserve both target and semantic class distributions when generating data splits. If semantic class information is unavailable, stratification can only be performed on the target (`REAL` / `FAKE`). As a result, semantic content may be unevenly distributed across training, validation, and test subsets, which could affect model performance.

- **Limited optimization search:** The final model is selected from a predefined set of experiments based on the CIFAKE publication and related CIFAR-10 research. Since only a limited number of configurations are tested, other architectures or hyperparameters may achieve better performance.

- **Reference reproducibility:** Some implementation and evaluation details of the original CIFAKE experiments are not fully specified, including the construction of the validation set, as well as other parts of the convolutional configuration. Their reported results may not be directly comparable to this implementation. This project uses a separate validation set for model selection and keeps the test set isolated until final evaluation.

### Recommendations

The model should not be treated as a production ready AI generated image detector without further training and evaluation.

When training on new data, users should review the dataset validation results before proceeding. The pipeline performs checks for issues such as invalid labels, duplicate image content, data leakage across splits, and class distributions. However, automated validation cannot guarantee that all issues will be detected.

Datasets with substantial `REAL` / `FAKE` class imbalance should be handled  before training. Depending on the dataset and intended application, appropriate strategies may include undersampling or oversampling. The pipeline reports class distributions rather than silently rebalancing the data.

When semantic class information is available, it should be provided so that generated splits can preserve both target and semantic distributions. Without semantic labels, stratification can only preserve the `REAL` / `FAKE` distribution, and semantic content may remain unevenly distributed across splits and affect model performance.

Before broader use, the model should be evaluated beyond the CIFAKE distribution, including images from different generative models and semantic domains, as well as images with different original resolutions and aspect ratios.

## Environmental Impact

The environmental impact of model training and experimentation is tracked using CodeCarbon.

- **Training hardware:** `TBD`
- **Compute environment:** `TBD`
- **Total compute time:** `TBD`
- **Energy consumed:** `TBD`
- **Carbon emitted:** `TBD`
- **Tracking tool:** CodeCarbon

## Model Card Authors
- Maribel Yazmin Preite: mayp@itu.dk ; maribel.yazmin.preite@estudiantat.upc.edu
- Luis Antonio Salinas: luis.antonio.salinas@estudiantat.upc.edu
- Rebeca Torrecilla: rebeca.torrecilla@estudiantat.upc.edu
- Aina Vila: aina.vila.arbusa@estudiantat.upc.edu