# TAKECare
The data and code for the TAKECare framework (TAKECare: A Temporal-Hierarchical Framework with Knowledge Fusion for Personalized Clinical Predictive Modeling) for review. 

Related code and data will be published in a new repository after review.

## 1. Install dependecies
Install the required packages, We recommend installing these key packages with the following command.

```
pip install torch  # version >= '1.10.1+cu113'
pip install nltk
pip install transformers
pip install scikit-learn==0.24.2
conda install -c conda-forge rdkit
```

Finally, other packages can generally be installed with `pip` or `conda` command.
```
pip install [xxx] # any required package if necessary
```

## 2. Data

For readers who use PyHealth package: For MIMIC-III and MIMIC-IV: refer to https://pyhealth.readthedocs.io/en/latest/api/datasets.html; 

For readers who follow previous works:
### Step 1: Certificate registration and input preparation
Get the certificate first, and then download the MIMIC-III and MIMIC-IV datasets.
+ MIMIC-III: https://physionet.org/content/mimiciii/1.4/
+ MIMIC-IV: https://physionet.org/content/mimiciv/
```
cd ./data
wget -r -N -c -np --user [account] --ask-password https://physionet.org/files/mimiciii/1.4/
wget -r -N -c -np --user [account] --ask-password https://physionet.org/files/mimiciv/
```

Get the following input files from https://github.com/ycq091044/SafeDrug and put them in the folder **./data/input/**.
+ RXCUI2atc4.csv: NDC-RXCUI-ATC4 mapping
+ rxnorm2RXCUI.txt: NDC-RXCUI mapping
+ drugbank_drugs_info.csv: Drugname-SMILES mapping
+ drug-atc.csv: CID-ATC mapping
+ drug-DDI.csv: DDI information coded by CID

### Step 2: Load the data and merge the original tables.
After downloading the raw dataset, put these required files in the folder path: **./data/mimic-iii/**.
+ DIAGNOSES_ICD.csv, PROCEDURES_ICD.csv, PRESCRIPTIONS.csv (diagnosis, procedure, prescription information)
+ D_ICD_DIAGNOSES.csv, D_ICD_PROCEDURES.csv (dictionary tables for diagnosis and procedure)


## 3. Folder Specification

- `data/`  only contains part of the data. See the **Data** section for more details
  - `input/` 
    - `drug-atc.csv`, `ndc2atc_level.csv`, `ndc2rxnorm_mapping.txt`: mapping files for drug code transformation
    - `idx2ndc.pkl`: It maps ATC code to rxnorm code and then query to drugbank
    - `idx2drug.pkl`: Drug ID to drug SMILES string dictionary
      
  - `output/`
    - `voc_final.pkl`: diag/prod/med index to code dictionary
    - `ddi_final.pkl`: ddi adjacency matrix
    - `ddi_matrix_H.pkl`: DDI-H mask structure
    - `records_final.pkl`: The final diagnosis-procedure-medication EHR records of each patient.
      
  - `ddi_mask_H.py`: The python script responsible for generating `ddi_mask_H.pkl` and `substructure_smiles.pkl`.
  - `processing.py`: The python script responsible for generating `voc_final.pkl`, `records_final.pkl`, and `ddi_final.pkl`   

- `src/` folder contains all the source code
  - `baselines.py`: Code for baseline models used for comparison (including: RETAIN, GRAM, KAME, StageNet, HiTANet, GCT, TRANS, G-BERT, GAMENet, SafeDrug, COGNet, DrugRec)
  - `code_initializer.py`: Medical code initialization (UMLS graph construction, SapBERT encoding, knowledge path aggregation, InfoNCE regularization)
  - `patient_encoder.py`: Patient representation learning (temporal hypergraph propagation via LeafMP, hierarchical co-occurrence via AnceMP)
  - `model.py`: Full model definition integrating code initializer and patient encoder, with clinical prediction heads and loss functions
  - `data_and_utils.py`: Unified module for data loading, MedKG graph construction, patient splitting, and utility functions
  - `training.py`: Training and evaluation routines for clinical prediction tasks (e.g., diagnosis and prescription)
  - `main.py`: Entrypoint to run training or testing
  - `Statistic_DDI_rate.py`: Script for computing DDI rate and counting adverse interactions


## 4. Run our project

```
python main.py [-h] [--Test] [--model_name MODEL_NAME]
                       [--resume_path RESUME_PATH] [--lr LR]
                       [--target_ddi TARGET_DDI] [--kp KP] [--dim DIM]
```


## 5. Tips
Welcome to contact me for any question.



