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

- **Experiment tracking:** MLflow
- **Data versioning:** DVC
- **Emission tracking:** CodeCarbon

- **Developed by:** Maribel Preite, Luis Salinas, Rebeca Torrecilla, Aina Vila
- **Project:** Advanced Topics in Data Engineering II (TAED2), Universitat Politècnica de Catalunya (UPC)
- **License:** `TBD`

### Model Sources

- **Repository:** [GAN-Busters GitHub Repository](https://github.com/taed2-2627q1-gced-upc/taed2-GAN-Busters)
- **Dataset Card:** [CIFAKE Dataset Card](../data/dataset_card.md)
- **CIFAKE Dataset:** [CIFAKE on Kaggle](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
- **Reference baseline:** [CIFAKE Publication](https://ieeexplore.ieee.org/abstract/document/10409290) | [CIFAR-10 Hyperparameter Selection Study](https://researchonline.gcu.ac.uk/ws/portalfiles/portal/26022569/Paper103.pdf)
- **Demo/Application:** `TBD`

--

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

--

## Bias, Risks, and Limitations

- **Generator dependency:** CIFAKE synthetic images were generated using [Stable Diffusion v1.4](https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf). The model may learn artifacts specific to this generator and may not generalize well to images produced by other generative models.

- **Image representation:** The model operates on 32×32 RGB images. Inputs with different resolutions, aspect ratios, or color formats are preprocessed to match this representation, which may result in information loss and affect prediction quality.

- **Limited object diversity:** CIFAKE is derived from the ten CIFAR-10 object/semantic classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. Performance may decrease for objects or visual content not represented during training.

- **Split stratification:** The provided CIFAKE split preserves both the `REAL`/`FAKE` distribution and the distribution of the ten CIFAR-10 object classes, with semantic class information encoded in the filenames. For new datasets without a split, the pipeline can preserve target class balance through stratification, while semantic class stratification is only possible when this information is available.

- **Reference reproducibility:** Some implementation and evaluation details of the original CIFAKE experiments are not fully specified, including the construction of the validation set, as well as other parts of the convolutional configuration. Their reported results may not be directly comparable to this implementation. This project uses a separate validation set for model selection and keeps the test set isolated until final evaluation.

- **Limited optimization search:** The final model is selected from a predefined set of experiments based on the CIFAKE publication and related CIFAR-10 research. Since only a limited number of configurations are tested, other architectures or hyperparameters may achieve better performance.

### Recommendations

The model should not be treated as a production ready AI generated image detector without further training and evaluation.

When retraining the model, balanced `REAL` and `FAKE` distributions are strongly suggested. Semantic class labels should also be provided when available, allowing the pipeline to preserve semantic class distributions across generated splits. A balanced representation of semantic classes is also recommended. If semantic class information is unavailable, the training data should instead contain a wide variety of objects and visual content to improve generalization to unseen content.

If pre-existing splits are provided, they should follow good practices to prevent data leakage. The pipeline will check for duplicate image content across splits and report potential inconsistencies before training.

Before production use, the model should be evaluated beyond the training distribution. This should include unseen object types, images from different real world sources and generative models, and common transformations such as cropping, rotation, resizing, and compression.

--

## Training Details

### Training Data

Dataset: **CIFAKE: Real and AI-Generated Synthetic Images**

The official CIFAKE training set contains 100,000 images, consisting of 50,000 `REAL` and 50,000 `FAKE` images.

- **Training subset:** 80,000 images (80%), consisting of 40,000 `REAL` and 40,000 `FAKE` images.
- **Validation subset:** 20,000 images (20%), consisting of 10,000 `REAL` and 10,000 `FAKE` images.
- **Image representation:** 32×32 RGB images.
- **Semantic classes:** Ten CIFAR-10 categories, uniformly represented within both `REAL` and `FAKE`.

The training and validation subsets preserve both the target and semantic class distributions of the original training set. A fixed split is used across experiments to ensure consistent model comparison.

For further details, see the [Dataset Card](../data/dataset_card.md).

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

- **Model:** CNN with 2×Conv2D(32 filters) + Dense(64).
- **Input:** 32×32 RGB images with pixel values normalized to `[0, 1]`.
- **Loss function:** Binary cross entropy.
- **Optimizer:** Adam.
- **Epochs:** 20.
- **Model selection:** Macro F1 and ROC-AUC on the validation set are used to compare model configurations during experimentation.
- **Tracking:** MLflow for experiment tracking and CodeCarbon for emission tracking.

#### Experimental Hyperparameters

- **Kernel size:** `[3×3, 5×5]`
- **Padding:** `['valid', 'same']`
- **Pooling:** `['max', 'average']`, with 2×2 pooling size
- **Learning rate:** `[10⁻², 10⁻³, 10⁻⁴]`
- **Batch size:** `[32, 64, 128]`
- **Dropout:** `[0, 0.125, 0.25]`, evaluated only if overfitting is observed

---

## Evaluation

### Testing Data

Dataset: **CIFAKE (official test split)**

- **Testing set:** 20,000 images, consisting of 10,000 `REAL` and 10,000 `FAKE` images.
- **Image representation:** 32×32 RGB images.
- **Semantic classes:** Ten CIFAR-10 categories, uniformly represented within both `REAL` and `FAKE`, with 1,000 images per category.

For further details, see the [Dataset Card](../data/dataset_card.md).

### Factors

Final model performance will be evaluated across the following factors:

- **Semantic class:** Compare performance across the ten CIFAKE semantic classes using the official test set.
- **External images:** Evaluate images from sources outside CIFAKE to assess performance beyond the training distribution.
- **Input preprocessing:** Evaluate external images with different original dimensions and aspect ratios to assess the effect of cropping and resizing to the required 32×32 input representation.

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

Final performance on the isolated CIFAKE test set:

- **Macro F1:** `TBD`
- **REAL precision:** `TBD`
- **REAL recall:** `TBD`
- **FAKE precision:** `TBD`
- **FAKE recall:** `TBD`
- **ROC-AUC:** `TBD`

Training and validation results from model experimentation, together with hyperparameters, learning curves, and relevant artifacts, are tracked and visualized using MLflow.

---

## Environmental Impact

The environmental impact of model training and experimentation is tracked using CodeCarbon.

- **Hardware type:** `TBD`
- **Compute environment:** Remote
- **Total compute time:** `TBD`
- **Energy consumed:** `TBD`
- **Carbon emitted:** `TBD`
- **Tracking tool:** CodeCarbon

## Model Card Authors
- Maribel Yazmin Preite: mayp@itu.dk ; maribel.yazmin.preite@estudiantat.upc.edu
- Luis Antonio Salinas: luis.antonio.salinas@estudiantat.upc.edu
- Rebeca Torrecilla: rebeca.torrecilla@estudiantat.upc.edu
- Aina Vila: aina.vila.arbusa@estudiantat.upc.edu