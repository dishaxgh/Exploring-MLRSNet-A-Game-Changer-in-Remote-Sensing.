# Exploring MLRSNet: A Game-Changer in Remote Sensing. 

## Content

|Item|Links|
|:-:|:-:|
|[Presentation]()|[[_pdf_](https://github.com/Deep-Learning-IGP-TUBS-SoSe2024/Group_01/blob/main/Final_Project/01-Presentation/Final%20Presentation.pdf)], [[_pptx_](https://github.com/Deep-Learning-IGP-TUBS-SoSe2024/Group_01/blob/main/Final_Project/01-Presentation/Final%20Presentation.pptx)]|
|[Code]()|[[_Fine Tuning Models_](https://github.com/Deep-Learning-IGP-TUBS-SoSe2024/Group_01/tree/main/Final_Project/03-Code/EfficientNet%20B0%20Fine%20Tuning)], [[_Scratch Models_](https://github.com/Deep-Learning-IGP-TUBS-SoSe2024/Group_01/tree/main/Final_Project/03-Code/EfficientNet%20B0%20Scratch)]|
|[Report]()|[[_pdf_](https://github.com/Deep-Learning-IGP-TUBS-SoSe2024/Group_01/blob/main/Final_Project/02-Report/Report_Group%201.pdf)]|
|[Proposal]()|[[_pdf_](https://github.com/Deep-Learning-IGP-TUBS-SoSe2024/Group_01/blob/main/Final_Project/00-Proposal/Project%20Proposal.pdf)], [[_pptx_](https://github.com/Deep-Learning-IGP-TUBS-SoSe2024/Group_01/blob/main/Final_Project/00-Proposal/Project%20Proposal.pptx)]|

<details>
  <summary style="font-size:140px">Guidelines</summary>
  
### Model Training Pipeline

#### 1. Fine-Tuning Models
For fine-tuning, the data processing and training pipeline relies on two core utility files orchestrated by a set of main execution scripts.

*   `data_generator.py`: Responsible for creating batches of images and their corresponding labels for the training, validation, and testing sets. It includes the `DataGenerator` class, which handles image preprocessing (such as resizing and normalizing) and organizes the data into batches.
*   `dataset_mlrsnet.py`: Handles the overall dataset preprocessing and splits the data into different subsets.

**Main Execution Scripts:**
The utility files above are invoked by the main scripts, which orchestrate the overall process of model training and evaluation across different data splits:
*   `EfficientNetB0_20-10-70.py`
*   `EfficientNetB0_30-10-60.py`
*   `EfficientNetB0_40-10-50.py`

---

#### 2. Training Models from Scratch

When training models from scratch, the pipeline incorporates data augmentation to improve model robustness and generalization. 

*   `data_generator_augmented.py`: An additional generator file that applies data augmentation techniques to create enhanced batches of training images. 

**Main Execution Scripts:**
This augmented data generator is integrated with dedicated training scripts that manage the training process using specific learning rate optimization methods:
*   `train_decay_lr.py`: Trains the models utilizing a **step decay** learning rate scheduling method.
*   `train_plateau_lr.py`: Trains the models utilizing a **performance scheduling** method (reducing the learning rate on a plateau).

</details>

<details>
  <summary style="font-size:140px">Dataset</summary>
  
The dataset used in our experiments is [THIS DATASET](https://github.com/cugbrs/MLRSNet) 

### MLRSNet Dataset

MLRSNet offers a diverse collection of high-resolution satellite images, providing different perspectives of the world. The dataset is suitable for tasks such as **multi-label image classification**, **multi-label image retrieval**, and **image segmentation**.

**Dataset Overview:**
* **Total Images:** 109,161 remote sensing images.
* **Categories:** 46 distinct categories (ranging between 1,500 and 3,000 images per category).
* **Image Specifications:** Fixed size of `256×256` pixels, with spatial resolutions varying from approximately 10m to 0.1m.
* **Annotations:** Each image is tagged with multiple labels selected from 60 predefined class labels (each image contains between 1 to 13 labels).

**Dataset Structure:**
* `Images/`: Contains the 109,161 high-resolution images organized into the 46 categories.
* `Labels/`: Each category is accompanied by a corresponding `.csv` file listing the labels.
* `Categories_names.xlsx`: 
  * **Sheet 1:** Lists the 46 category names.
  * **Sheet 2:** Details the associated multi-labels for each category.
</details>

<details>
  <summary style="font-size:140px">Model</summary>
  
The core model employed in this project is the **EfficientNet-B0** base model. 

**Experimental Setup:**
* **Pre-trained Evaluation:** We compared the performance of the EfficientNet-B0 model (pre-trained on ImageNet weights) against other pre-trained models referenced in the original MLRSNet paper.
* **Training from Scratch:** To further evaluate its capabilities, we trained two EfficientNet-B0 models entirely from scratch. For these models, we used data split ratios different from those used in the original paper, and applied two distinct learning rate strategies:
  * **Strategy 1:** Utilized a **step decay** learning rate schedule.
  * **Strategy 2:** Implemented **performance scheduling** (reducing the learning rate based on plateauing metrics).

**Architecture References:**
For more visual details on the model's design, please refer to the diagrams below:
* [Concept behind Model Architecture](https://github.com/user-attachments/assets/171b6291-4350-4175-b835-c965c3280c19)
* [Detailed Model Architecture](https://github.com/user-attachments/assets/0d017480-c469-4044-ba35-97fee68b4b87)
  
</details>

<details>
  <summary style="font-size:140px">Results</summary>
  
#### 1. Fine-Tuning Models Evaluation

The tables below compare the fine-tuning performance of the **EfficientNet-B0** model against the baseline architectures evaluated in the original MLRSNet paper across different training set proportions (20%, 30%, and 40%).

##### a) Mean Average Precision (mAP %)
| Model | 20% Train | 30% Train | 40% Train |
| :--- | :---: | :---: | :---: |
| **MLRSNet-InceptionV3** | 81.50 | 82.33 | 84.84 |
| **MLRSNet-VGGNet16** | 67.88 | 72.66 | 75.39 |
| **MLRSNet-VGGNet19** | 66.12 | 69.53 | 73.60 |
| **MLRSNet-ResNet50** | 82.65 | 84.28 | 86.01 |
| **MLRSNet-ResNet101** | 83.26 | 84.19 | 85.72 |
| **MLRSNet-DenseNet121** | 75.96 | 77.99 | 80.25 |
| **MLRSNet-DenseNet169** | 82.16 | 86.42 | 87.35 |
| **MLRSNet-DenseNet201** | 87.25 | 87.84 | 88.77 |
| **MLRSNet-EfficientNetB0** | **93.10** | **94.10** | **94.58** |

##### b) F1 Score (Samples)
| Model | 20% Train | 30% Train | 40% Train |
| :--- | :---: | :---: | :---: |
| **MLRSNet-InceptionV3** | 0.7746 | 0.8016 | 0.8146 |
| **MLRSNet-VGGNet16** | 0.5743 | 0.6534 | 0.6855 |
| **MLRSNet-VGGNet19** | 0.5677 | 0.6120 | 0.6329 |
| **MLRSNet-ResNet50** | 0.7530 | 0.8176 | 0.8353 |
| **MLRSNet-ResNet101** | 0.7618 | 0.7703 | 0.8226 |
| **MLRSNet-DenseNet121** | 0.7154 | 0.7389 | 0.7571 |
| **MLRSNet-DenseNet169** | 0.8138 | 0.8408 | 0.8521 |
| **MLRSNet-DenseNet201** | 0.8381 | 0.8414 | 0.8538 |
| **MLRSNet-EfficientNetB0** | **0.8355** | **0.8492** | **0.8539** |

> **Key Takeaway:** Across all training ratios, **EfficientNet-B0** consistently outperforms all baseline architectures from the original paper in both Mean Average Precision (mAP) and F1 Score.

---

#### 2. Models Trained from Scratch Evaluation

Comparison of the EfficientNet-B0 model trained from scratch using two distinct learning rate adjustment strategies (**Performance Scheduling** vs. **Step Decay**):

| Metric | Performance Scheduling | Step Decay |
| :--- | :---: | :---: |
| **F1 Score (Samples)** | **0.8568** | 0.8395 |
| **Binary Accuracy** | **97.58%** | 97.15% |
| **ROC AUC Curve** | **99.22%** | 98.93% |
| **Mean Average Precision (mAP)** | **94.61%** | 93.53% |
| **Hamming Loss** *(lower is better)* | **2.41%** | 2.84% |

> **Key Takeaway:** Training from scratch with **Performance Scheduling** achieved superior overall performance compared to Step Decay across all evaluated metrics, achieving a peak mAP of **94.61%** and an ROC AUC of **99.22%**.


</details>
