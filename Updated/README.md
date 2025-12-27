# LightXrayNet - Lightweight CNN for Chest X-Ray Classification

A lightweight Convolutional Neural Network for multi-class chest X-ray image classification with automated preprocessing pipeline.

## 📋 Overview

LightXrayNet is designed to classify chest X-ray images into 9 different categories of thoracic conditions using a lightweight CNN architecture. The project includes a complete preprocessing pipeline and model training workflow.

## 🏥 Dataset

The dataset contains chest X-ray images categorized into 9 classes:

- **00** - Anatomia Normal (Normal Anatomy)
- **01** - Processos Inflamatórios Pulmonares (Pneumonia)
- **02** - Maior Densidade (Derrame Pleural, Consolidação Atelectasica, Hidrotorax, Empiema)
- **03** - Menor Densidade (Pneumotorax, Pneumomediastino, Pneumoperitonio)
- **04** - Doenças Pulmonares Obstrutivas (Enfisema, Broncopneumonia, Bronquiectasia, Embolia)
- **05** - Doenças Infecciosas Degenerativas (Tuberculose, Sarcoidose, Proteinose, Fibrose)
- **06** - Lesões Encapsuladas (Abscessos, Nódulos, Cistos, Massas Tumorais, Metastases)
- **07** - Alterações de Mediastino (Pericardite, Malformações Arteriovenosas, Linfonodomegalias)
- **08** - Alterações do Tórax (Atelectasias, Malformações, Agenesia, Hipoplasias)

### Dataset Link

📦 [Download Dataset](https://drive.google.com/drive/folders/1crCzXnJFBYjR2PCVk3kmnnta2Xe7vuAc?usp=sharing)

## 📁 Project Structure

```
Updated/
├── 01_CLAHE.py                    # CLAHE preprocessing
├── 02_Augment.py                  # Data augmentation
├── 03_Resize.py                   # Image resizing
├── 04_split_4kaggle.py            # Dataset splitting
├── 05_Train_CV_Comparison.py      # Model training with CV comparison
├── 06_LXNet.ipynb                 # Jupyter notebook for experiments
├── Dataset/
│   ├── 00_Raw_Dataset/            # Original images
│   ├── 01_CLAHE/                  # CLAHE enhanced images
│   ├── 02_AUGMENTED_BALANCED/     # Augmented and balanced dataset
│   └── 03_RESIZED_AUGMENTED_BALANCED/  # Final preprocessed dataset
├── Experimental Flowchart.drawio
└── Proposed Method.drawio
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install opencv-python numpy tensorflow keras scikit-learn matplotlib pandas
```

### Pipeline Execution

Follow these steps in order to preprocess the data and train the model:

#### 1. CLAHE Enhancement

Apply Contrast Limited Adaptive Histogram Equalization to enhance image contrast:

```bash
python 01_CLAHE.py
```

This will process images from `Dataset/00_Raw_Dataset/` and save enhanced images to `Dataset/01_CLAHE/`.

#### 2. Data Augmentation

Balance the dataset and apply augmentation techniques:

```bash
python 02_Augment.py
```

This will create augmented images in `Dataset/02_AUGMENTED_BALANCED/`.

#### 3. Image Resizing

Resize images to a uniform size for model input:

```bash
python 03_Resize.py
```

This will save resized images to `Dataset/03_RESIZED_AUGMENTED_BALANCED/`.

#### 4. Dataset Splitting

Split the dataset into training, validation, and test sets:

```bash
python 04_split_4kaggle.py
```

#### 5. Model Training

Train the LightXrayNet model with cross-validation comparison:

```bash
python 05_Train_CV_Comparison.py
```

## 📓 Jupyter Notebook

The project includes a Jupyter notebook (`06_LXNet.ipynb`) for interactive experimentation, visualization, and model development.

```bash
jupyter notebook 06_LXNet.ipynb
```

## 🏗️ Model Architecture

LightXrayNet is a lightweight CNN architecture optimized for:

- Reduced computational complexity
- Fast inference time
- High accuracy on chest X-ray classification
- Suitable for deployment on resource-constrained devices

## 📊 Results

The model performance and cross-validation results will be generated during training in `05_Train_CV_Comparison.py`.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or collaborations, please open an issue in the repository.

---

**Note**: Make sure to download the dataset from the provided Google Drive link before running the preprocessing pipeline.
