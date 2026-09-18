---
# For reference on dataset card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/datasetcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/datasets-cards

# Dataset Card for CIFAKE

CIFAke is a binary image classification dataset built to study whether computer vision models can distinguish real images from AI-generated synthetic photos. It combines the CIFAR-10 dataset with a matching set of generated images created with a diffusion model, giving a clean, balances benchmark for real vs fake photo detection. 

## Dataset Details

### Dataset Description

The CIFAKE dataset contains a mix of 60,000 synthetically-generated images and 60,000 real images (collected from CIFAR-10). It is specially designed for computer vision tasks and techniques realted to AI detection on images. 

The dataset contains two classes that define its origin: REAL and FAKE. The "REAL" ones were collected from Krizhhevsky & Hinton's CIFAR-10 dataset and the "FAKE" ones were generated with Stable Diffusion v1.4, using the same ten CIFAR-10 class labels (e.g. airplane, automobile, bird, cat...) as generation prompts so that both classes cover the same semantic categories. 

- **Created by:** Jordan J. Bird and Ahmad Lotfi
- **Shared by:** Jordan J. Bird (via Kaggle)
- **Language(s) (NLP):** Not applicable - this is an image dataset with no text content. 
- **License:** Published under the same MIT license as CIFAR-10.

### Dataset Sources 

<!-- Provide the basic links for the dataset. -->

- **Repository:** https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
- **Paper:** Bird, J.J. & Lotfi, A. (2024). "CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images." *IEEE Access.* Preprint: https://arxiv.org/abs/2303.14126

## Uses

<!-- Address questions around how the dataset is intended to be used. -->

### Direct Use
The CIFAKE dataset can be applied directly for specific, technical purposes, leveraging its ready-to-use structure:

<!-- This section describes suitable use cases for the dataset. -->
- Training and evaluating binary classifiers (as CNNs, Vision Transformers...) to detect whether an image is a real photograph or AI-generated. 
- Benchmarking explainability techniques for AI-generated image detection (using Grad-CAM to visualize which regions of an image drive the "fake" classification)
- Teaching and learning exercises around reproducible ML pipelines, given its small size and balanced classes 

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the dataset will not work well for. -->
Some of the main uses include the later points but there are certain uses that may be considered out-of-scope or inappropiate. Exemples of misuse could be the following: 

- Not suitable as a general-purpose forensic tool for verifying the authenticity of arbitrary, high-resolution, real-world images. The fake images were generated with a single model (Stable Diffusion v1.4) at low resolution (32x32, matching CIFAR-10) a couple of years ago (the AI-generated images' quality improves very quickly so the generated images today would be much more realistic), so the visual artifacts learned from this dataset may not transfer to images from newer or different generative models.
- Not intended to make legal, journallistic or high-stakes authenticity determinations about real images. 
- Not designed for detecting manipulated (edited/deepfake) images. It only covers fully synthetic and fully real images, not partial manipulations. 

## Dataset Structure

<!-- This section provides a description of the dataset fields, and additional information about the dataset structure such as criteria used to create the splits, relationships between data points, etc. -->
The CIFAKE dataset is organized as follows:
- Two top-level classes: `REAL` and `FAKE`
- 120,000 images in total: 60,000 REAL + 60,000 FAKE
- Images are 32x32 pixels, in RGB, matching the CIFAR-10 image format
- The dataset is pre-split into training and test sets (following the Kaggle distribution), with a balanced number of REAL/FAKE images in each split

## Dataset Creation

### Curation Rationale

<!-- Motivation for the creation of this dataset. -->

This dataset was created in response to the rapid improvement of generative image models, which has made it increasingly difficult for humans to tell AI-generated images apart from real photographs. The authors wanted a clean, controlled benchmark to study this problem and to build explainable detection methods. 

### Source Data

<!-- This section describes the source data (e.g. news text and headlines, social media posts, translated sentences, ...). -->

#### Data Collection and Processing

<!-- This section describes the data collection and processing process such as data selection criteria, filtering and normalization methods, tools and libraries used, etc. -->

- **REAL** images were taken directly from the existing CIFAR-10 dataset (Krizhevsky & Hinton, 2009), a well-established benchmark of 60,000 32x32 color photographs across 10 classes. 
- **FAKE** images were generated by CIFAKE authors using Stable Diffusion v1.4, prompted to reproduce the same 10 CIFAR-10 categories, then resized to 32x32 pixels to match the real images. 

#### Who are the source data producers?

<!-- This section describes the people or systems who originally created the data. It should also include self-reported demographic or identity information for the source data creators if this information is available. -->

The collectors of the real images were the CIFAR-10 team (Krizhevsky & Hinton). They took them from web-scraped photographs. 

The authors of the dataset (Jordan J. Bird & Ahmad Lofti) were the ones who generated the fake images using Stable Diffusion v1.4 .


#### Personal and Sensitive Information

<!-- State whether the dataset contains data that might be considered personal, sensitive, or private (e.g., data that reveals addresses, uniquely identifiable names or aliases, racial or ethnic origins, sexual orientations, religious beliefs, political opinions, financial or health data, etc.). If efforts were made to anonymize the data, describe the anonymization process. -->

The dataset does not contain personal or sensitive information.

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

- **Concept drift**: The fake images were generated with Stable Diffusion v1.4, a model from 2023. Generative models have advanced substantially since then, so classifiers trained only on CIFAKE may not generalize well to images from more recent and different generators. 
- **Low resolution**: The 32x32 image size (inherited from CIFAR-10) is much lower than typical real-world photos, which limits how well findings transfer to full-resolution AI-generated image detection. 
- **Single generator**: All fake images come from one model, so the dataset may encode generator-specific artifacts rather than "AI-generated" signals.

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

Users should treat CIFAKE as a pedagogical and benchmark dataset rather than a production-ready detector for modern AI-generated content. For real-world deployment, models should be validated on more diverse, higher-resolution, and more recent datasets covering multiple generative models. 

## Citation 

<!-- If there is a paper or blog post introducing the dataset, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

```bibtex
@article{bird2024cifake,
  title={CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images},
  author={Bird, Jordan J. and Lotfi, Ahmad},
  journal={IEEE Access},
  year={2024},
  publisher={IEEE}
}
```

**APA:**

Bird, J. J., & Lotfi, A. (2024). CIFAKE: Image classification and explainable identification of AI-generated synthetic images. *IEEE Access.*

## Glossary 

<!-- If relevant, include terms and calculations in this section that can help readers understand the dataset or dataset card. -->

- **REAL**: authentic photographs, sourced from CIFAR-10
- **FAKE**: synthetic images, generated by Stable Diffusion v1.4
- **Concept drift**: the phenomenon where a model's performance degrades beacuse the data it now sees differs from the data it was trained/evaluated on (here, older vs. newer generative models)

## More Information 

For our TAED2 project, we use CIFAKE as the primary training/evaluation dataset for our AI-generated image detection component, deployed as a REST API following the course's software engineering best practices. 

## Dataset Card Authors' Contact
- Luis Antonio Salinas: luis.antonio.salinas@estudiantat.upc.edu
- Rebeca Torrecilla: rebeca.torrecilla@estudiantat.upc.edu
- Aina Vila: aina.vila.arbusa@estudiantat.upc.edu


