# SwineBERT-ICD: A Weak Supervision Model for Swine Health Classification

This repository contains the complete code and methodology for the DATA*6700 Major Research Project at University of Guelph in Summer 2025, **SwineBERT-ICD**. The project develops and validates a deep learning pipeline for the multi-label classification of unstructured swine clinical records into standardized ICD-11-aligned syndromes.

## Project Overview

The timely analysis of veterinary health records is critical for disease surveillance, yet most clinical data is locked in unstructured narratives, delaying outbreak detection. This project tackles this challenge by fine-tuning a VetBERT model on **48,675 clinical records** from the Animal Health Laboratory at the University of Guelph. 

It uses an advanced two-stage **"teacher-student" weak supervision methodology**, inspired by the PetBERT-ICD paper (Wang et al., 2023), to leverage a small, 2,000-record expert-annotated dataset to train a powerful model on the much larger unannotated corpus.

## Methodology Workflow

The project follows a multi-phase workflow, from initial data preparation to the final evaluation of a fine-tuned transformer model.

### Phase 1: Data Preparation
- Start with the raw 48k dataset
- Clean the data and perform an 80/20 split
- **Output:** Training Set (~38.4k records) and Test Set (~9.6k records)

### Phase 2: Annotation & Teacher Models
- Apply enriched sampling to the Training Set to get a 2,000-record sample
- Perform expert annotation to create the Gold Standard Dataset
- Train calibrated "teacher" models on the Gold Standard Dataset

### Phase 3: Weak Supervision & Training
- Use the 11 saved teacher models and the ~36.4k unannotated records
- Perform the pseudo-labeling process to create a weakly labeled dataset
- Fine-tune the VetBERT "student" model on the weakly labeled and gold-standard data
- **Output:** The final trained SwineBERT-ICD model

### Phase 4: Final Evaluation
- Take the held-out Test Set and sample 1,000 records for final expert annotation
- Use the final trained model to perform the final performance evaluation on this new test set

## Repository Structure

```
.
├── teacher_models_calibrated/   # Saved binary "teacher" models
├── final_swinebert_model/       # The final, fine-tuned SwineBERT-ICD model
├── data/                        # Contains raw, processed, and annotated data files
├── 01_Data_Preparation_for_Annotation.ipynb    # Notebook for cleaning and splitting the raw data
├── 02_Annotation_Modeling.ipynb      # Notebook for training the simple baseline model
├── 03_Train_Teacher_Models.ipynb  # Notebook to train the calibrated "teacher" models
├── 04_Pseudo_Labeling.ipynb     # Notebook to generate the weakly labeled dataset
├── 05_Train_Final_SwineBERT.ipynb # Notebook to fine-tune the final model and evaluation on the test set
└── README.md                    # This file
```

## Setup and Installation

This project uses a dedicated Conda environment to manage dependencies.

1. **Create and activate the Conda environment:**
   ```bash
   conda create -n swine_project python=3.11
   conda activate swine_project
   ```

2. **Install necessary packages:**
   ```bash
   conda install pytorch pandas jupyter notebook scikit-learn openpyxl -c pytorch -c conda-forge
   pip install transformers datasets accelerate huggingface_hub
   ```

## How to Run

To reproduce the results, run the Jupyter Notebooks in the following order:

1. `01_Data_Preparation_for_Annotation.ipynb`: Cleans the raw data and creates the initial train/test splits and the 2,000-record sample for annotation
2. `02_Annotation_Modeling.ipynb`: Trains a simple baseline model on the expert-annotated data to establish a performance benchmark
3. `03_Train_Teacher_Models.ipynb`: Trains the 11 calibrated binary "teacher" models and saves them to the `teacher_models_calibrated/` directory
4. `04_Pseudo_Labeling.ipynb`: Uses the saved teacher models to generate the `pseudo_labeled_dataset.csv`
5. `05_Train_Final_SwineBERT.ipynb`: Fine-tunes the final `SwineBERT-ICD` model on the pseudo-labeled data and saves it to the `final_swinebert_model/` directory; Loads the final model and evaluates its performance on the held-out, expert-annotated test set

## Final Results

The final SwineBERT-ICD model was evaluated on a held-out, 1,000-record expert-annotated test set. On a core set of six clinically meaningful and well-supported labels, the model achieved a **MACRO F1-SCORE of 0.68**.

| Label                   | F1-Score | Precision | Recall |
|-------------------------|----------|-----------|--------|
| Monitoring              | 0.84     | 0.85      | 0.82   |
| Unknown                 | 0.78     | 0.75      | 0.82   |
| Respiratory System      | 0.69     | 0.63      | 0.77   |
| Digestive System        | 0.67     | 0.81      | 0.57   |
| Pregnancy/Puerperium    | 0.64     | 0.70      | 0.58   |
| Infectious Diseases     | 0.46     | 0.61      | 0.37   |

## Acknowledgements

This research was made possible by the data provided by the **Animal Health Laboratory, University of Guelph** and the expert clinical guidance of **Dr. Zvonimir Poljak**.
