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
- **Output:** Binary class (`REAL` / `FAKE`) + corresponding probability
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
  - **Early stopping:** `TBD`

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

## Uses

### Direct Use

This model can be used to classify input images as `REAL` or `FAKE` and provide a probability estimate for the predicted class.

The model is integrated into an end-to-end MLOps pipeline, developed primarily for educational purposes to cover the entire machine learning lifecycle. AI generated image detection serves as the underlying machine learning task for the pipeline.

### Downstream Use

The model and pipeline may be reused and extended for further experimentation or development, with appropriate attribution to the original repository and implementation. Potential downstream uses include fine tuning or improving the model with additional data, reusing the pipeline with alternative model configurations, or integrating the model and pipeline into a larger application.

## Bias, Risks, and Limitations

- **Generator dependency:** CIFAKE synthetic images were generated using [Stable Diffusion v1.4](https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf). The model may learn artifacts specific to this generator and may not generalize well to images produced by other generative models.

- **Image representation:** The model operates on 32×32 RGB images. Inputs with different resolutions, aspect ratios, or color formats are preprocessed to match this representation, which may result in information loss and affect prediction quality.

- **Limited object diversity:** CIFAKE is derived from the ten CIFAR-10 object classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. Performance may decrease for objects or visual content not represented during training.

- **Split stratification:** The provided CIFAKE split preserves both the `REAL`/`FAKE` distribution and the distribution of the ten CIFAR-10 object classes, with semantic class information encoded in the filenames. For new datasets, the pipeline can preserve target class balance, but object level stratification is only possible when the semantic class information is available.

- **Reference reproducibility:** Some implementation and evaluation details of the original CIFAKE experiments are not fully specified, including the construction of the validation set, as well as other parts of the convolutional configuration. Their reported results may not be directly comparable to this implementation. This project uses a separate validation set for model selection and keeps the test set isolated until final evaluation.

- **Limited optimization search:** The final model is selected from a predefined set of experiments based on the CIFAKE publication and related CIFAR-10 research. Since only a limited number of configurations are tested, other architectures or hyperparameters may achieve better performance.


### Recommendations

The model should not be treated as a production ready AI generated image detector without further training and evaluation.

When retraining the model, balanced `REAL` and `FAKE` distributions should be maintained. Semantic class labels should also be provided, allowing the pipeline to preserve object distributions across generated splits. A balanced representation of semantic classes is recommended to avoid overrepresenting particular types of content. If semantic class information is unavailable, the training data should instead contain a wide variety of objects and visual content to improve generalization to unseen content.

If pre-existing splits are provided, they should follow good practices to prevent data leakage. The pipeline will check for duplicate filenames and duplicate image content across splits and report potential inconsistencies before training.

Before production use, the model should be evaluated beyond the training distribution. This should include unseen object types, images from different real world sources and generative models, and common transformations such as cropping, rotation, resizing, and compression.

## How to Get Started with the Model

Use the code below to get started with the model.

{{ get_started_code | default("[More Information Needed]", true)}}

## Training Details
-- EVERYTHING STILL NEEDS TO BE DEFINED--
### Training Data

The model is trained using the CIFAKE: Real and AI-Generated Synthetic Images dataset.

CIFAKE contains 120,000 images divided into two balanced classes:

REAL: 60,000 real images collected from the CIFAR-10 dataset.
FAKE: 60,000 AI-generated synthetic images generated using Stable Diffusion 1.4.
Training set: 100,000 images, consisting of 50,000 REAL and 50,000 FAKE images.
Testing set: 20,000 images, consisting of 10,000 REAL and 10,000 FAKE images.

The images are 32 × 32 RGB images and represent ten semantic categories corresponding to the CIFAR-10 classes.

Dataset source:

https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Preprocessing [optional]

{{ preprocessing | default("[More Information Needed]", true)}}


#### Training Hyperparameters

- **Training regime:** {{ training_regime | default("[More Information Needed]", true)}} <!--fp32, fp16 mixed precision, bf16 mixed precision, bf16 non-mixed precision, fp16 non-mixed precision, fp8 mixed precision -->

#### Speeds, Sizes, Times [optional]

<!-- This section provides information about throughput, start/end time, checkpoint size if relevant, etc. -->

{{ speeds_sizes_times | default("[More Information Needed]", true)}}

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

The CIFAKE test set contains 20,000 images:

10,000 REAL images.
10,000 AI-generated FAKE images.

#### Factors

- Image class: REAL vs. AI-GENERATED.
- CIFAR-10 semantic category.
- Image characteristics and visual content.
- Potential differences between the training distribution and external images.

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

{{ testing_metrics | default("[More Information Needed]", true)}}

### Results

{{ results | default("[More Information Needed]", true)}}

#### Summary

{{ results_summary | default("", true) }}

## Model Examination [optional]

<!-- Relevant interpretability work for the model goes here -->

{{ model_examination | default("[More Information Needed]", true)}}

## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** {{ hardware_type | default("[More Information Needed]", true)}}
- **Hours used:** {{ hours_used | default("[More Information Needed]", true)}}
- **Cloud Provider:** {{ cloud_provider | default("[More Information Needed]", true)}}
- **Compute Region:** {{ cloud_region | default("[More Information Needed]", true)}}
- **Carbon Emitted:** {{ co2_emitted | default("[More Information Needed]", true)}}

## Technical Specifications [optional]

### Model Architecture and Objective

{{ model_specs | default("[More Information Needed]", true)}}

### Compute Infrastructure

{{ compute_infrastructure | default("[More Information Needed]", true)}}

#### Hardware

{{ hardware_requirements | default("[More Information Needed]", true)}}

#### Software

{{ software | default("[More Information Needed]", true)}}

## Citation [optional]

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

{{ citation_bibtex | default("[More Information Needed]", true)}}

**APA:**

{{ citation_apa | default("[More Information Needed]", true)}}

## Glossary [optional]

<!-- If relevant, include terms and calculations in this section that can help readers understand the model or model card. -->

{{ glossary | default("[More Information Needed]", true)}}

## More Information [optional]

{{ more_information | default("[More Information Needed]", true)}}

## Model Card Authors [optional]

{{ model_card_authors | default("[More Information Needed]", true)}}

## Model Card Contact

{{ model_card_contact | default("[More Information Needed]", true)}}
