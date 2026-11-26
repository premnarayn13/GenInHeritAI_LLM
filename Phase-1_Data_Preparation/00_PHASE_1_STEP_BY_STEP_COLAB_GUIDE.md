# GenInherit: Phase 1 Data Preparation
## Complete Step-by-Step Click-by-Click Guide for Google Colab

```text
====================================================================================================
  GENINHERIT-LLM | PHASE 1 DATA PREPARATION PLAYBOOK
  TARGET: Click-by-Click, Cell-by-Cell Instructions to Acquire 1.3 GB Datasets, Build the BioDSL
          Vocabulary, and Generate Preprocessed Training Shards Directly on Google Colab & Drive
====================================================================================================
```

This guide explains **exactly what to do, where to go, what to click, and which commands to run**—both outside Colab (on Google Drive) and inside Google Colab.

---

## PART 1: Outside of Colab (What to Do on Your Laptop & Google Drive)

You do **not** need to install Python, PyTorch, or any heavy software on your laptop. You only need your web browser.

### Step 1.1: Open Google Drive
1. Open your browser and go to: [https://drive.google.com](https://drive.google.com).
2. Log in with your standard Google account.

### Step 1.2: Create the GenInherit Directory Structure
1. In your **My Drive**, click the **"+ New"** button on the top-left $\to$ Click **"New folder"**.
2. Name the folder: `geninherit` and click **Create**.
3. Double-click to open `geninherit`.
4. Inside `geninherit`, create three sub-folders:
   - `data`
   - `runs`
   - `code`
5. Inside `data`, create one sub-folder:
   - `shards`

Your Google Drive folder structure should look like this:
```text
My Drive/
└── geninherit/
    ├── code/
    ├── data/
    │   └── shards/
    └── runs/
```

### Step 1.3: Upload the Phase-1 Python Scripts
1. Open the folder `My Drive/geninherit/code/` in Google Drive.
2. Drag and drop the following Python files from your local computer (`Implementation/Phase-1_Data_Preparation/`):
   - `01_download_raw_data.py`
   - `02_synthetic_pedigree_generator.py`
   - `03_build_biodsl_tokenizer.py`
   - `04_preprocess_and_shard.py`
   - `05_verify_phase1_shards.py`
   - `run_phase1_pipeline.py`

*(Alternatively, you can copy-paste them directly into Colab cells as shown below!)*

---

## PART 2: Inside Google Colab (What to Click & What to Run)

### Step 2.1: Open Google Colab
1. In a new browser tab, go to: [https://colab.research.google.com](https://colab.research.google.com).
2. A pop-up will appear. Click **"New notebook"** on the bottom right.
3. On the top-left where it says `Untitled0.ipynb`, rename it to:
   `GenInherit_Phase1_Data_Prep.ipynb`

### Step 2.2: Set the Free GPU Runtime
1. In the top menu bar, click on **Runtime**.
2. In the dropdown menu, click **Change runtime type**.
3. Under **Hardware accelerator**, select **T4 GPU** (Standard on the free tier).
4. Click the blue **Save** button.

---

## PART 3: The Exact Cells to Run Inside Google Colab

Run these cells one by one by clicking the **Play button (▶)** on the left of each cell (or press `Shift + Enter`).

---

### Cell 1: Mount Google Drive
*This links your Google Drive so that all data is saved permanently to the cloud, not your laptop.*

```python
# CELL 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

import os
# Set up workspace path on Google Drive
WORKSPACE_DRIVE = "/content/drive/MyDrive/geninherit"
LOCAL_RAW_DIR = "/content/raw_data"
LOCAL_SHARDS_DIR = f"{WORKSPACE_DRIVE}/data/shards"

os.makedirs(LOCAL_RAW_DIR, exist_ok=True)
os.makedirs(LOCAL_SHARDS_DIR, exist_ok=True)
print("✓ Google Drive mounted successfully at /content/drive")
```
> **What to Click When Running Cell 1:**  
> A pop-up box will appear saying *"Permit this notebook to access your Google Drive files?"* $\to$ Click **"Connect to Google Drive"** $\to$ Select your Google Account $\to$ Click **"Allow"**.

---

### Cell 2: Install Minimal Dependencies
*Takes about 15 seconds.*

```python
# CELL 2: Install required libraries
!pip install -q torch transformers tokenizers safetensors
print("✓ Dependencies installed successfully.")
```

---

### Cell 3: Download Raw Genomic Datasets (~1.3 GB)
*This downloads ClinVar, HPO phenotypes, MANE canonical transcripts, and ClinGen validity directly into Colab at cloud speeds.*

```python
# CELL 3: Download Public Genomic Data
!python /content/drive/MyDrive/geninherit/code/01_download_raw_data.py /content/raw_data
```

**Expected Output:**
```text
================================================================================
  GENINHERIT PHASE 1: RAW DATASET ACQUISITION ENGINE
================================================================================
Downloading [clinvar_summary]: ClinVar Variant Summary
  Source URL: https://ftp.ncbi.nlm.nih.gov/...
  -> [variant_summary.txt.gz] 100% (150.2 MB) @ 25.4 MB/s
✓ Completed download: /content/raw_data/variant_summary.txt.gz
...
✓ Phase 1 Dataset Acquisition Completed Successfully.
```

---

### Cell 4: Generate 15,000 Multigenerational Family Pedigrees (~500 MB)
*Generates 15,000 realistic 3-4 generation family trees across all 70+ archetypes (Autosomal Dominant with reduced penetrance, Autosomal Recessive first-cousin loops, X-Linked Recessive, and De Novo mutations).*

```python
# CELL 4: Generate Synthetic Cohort
!python /content/drive/MyDrive/geninherit/code/02_synthetic_pedigree_generator.py \
    /content/raw_data/synthetic_pedigrees.json 15000
```

**Expected Output:**
```text
================================================================================
  GENINHERIT PHASE 1: MULTIGENERATIONAL SYNTHETIC COHORT GENERATOR
  Target Families Count: 15,000
================================================================================
  -> Generated 2,500/15,000 families (1450 fam/sec)...
  -> Generated 5,000/15,000 families (1480 fam/sec)...
  ...
✓ Cohort Generation Complete in ~10.5s!
  File Size on Disk: ~48.5 MB
================================================================================
```

---

### Cell 5: Build the BioDSL 4,096-Token Vocabulary & Sanity Check
*Creates `bioDSL_vocab.json` directly in your Google Drive and runs the 100% round-trip sanity check.*

```python
# CELL 5: Build BioDSL Tokenizer
!python /content/drive/MyDrive/geninherit/code/03_build_biodsl_tokenizer.py \
    /content/drive/MyDrive/geninherit/bioDSL_vocab.json
```

**Expected Output:**
```text
================================================================================
  GENINHERIT PHASE 1: BioDSL 4,096-TOKEN HYBRID VOCABULARY BUILDER
================================================================================
  [+] Control & Pedigree Structure       : Added  128 tokens
  [+] Pillar 1 Feature Atoms             : Added  480 tokens
  [+] Alphabets, Digits & HGVS Idioms    : Added  112 tokens
  [+] Chromosomes & Transcripts          : Added   96 tokens
  [+] Evidence, ACMG & Assay Tokens      : Added  192 tokens
  [+] Curated Disease Genes              : Added 1400 tokens
  [+] Curated HPO Clinical Terms         : Added 1100 tokens
  [+] Learned BPE Multi-Token Slots      : Added  588 tokens
================================================================================
✓ BioDSL Vocabulary Successfully Built (Exactly 4,096 tokens)!
✓ 100% Round-trip Tokenizer Sanity Check PASSED.
================================================================================
```

---

### Cell 6: Preprocess & Pack into Binary Shards on Google Drive
*This reads the cohort, converts families into PSTS-v1 sequence tokens, attaches the Wright-Malecot kinship Fourier numeric channel, and saves binary PyTorch shards (`shard_0000.pt`, etc.) directly into Google Drive.*

```python
# CELL 6: Streaming Preprocessor & Sharder
!python /content/drive/MyDrive/geninherit/code/04_preprocess_and_shard.py \
    /content/raw_data/synthetic_pedigrees.json \
    /content/drive/MyDrive/geninherit/bioDSL_vocab.json \
    /content/drive/MyDrive/geninherit/data/shards \
    2500
```

**Expected Output:**
```text
================================================================================
  GENINHERIT PHASE 1: STREAMING PREPROCESSOR & BINARY SHARDER
================================================================================
  Total Cohort: 15,000 families
  -> Train: 13,500 | Val IID: 750 | Val Comp: 750
  ✓ Saved Shard 0000: 2,500 families (~38.2 MB) -> shard_0000.pt
  ✓ Saved Shard 0001: 2,500 families (~38.2 MB) -> shard_0001.pt
  ✓ Saved Shard 0002: 2,500 families (~38.2 MB) -> shard_0002.pt
  ✓ Saved Shard 0003: 2,500 families (~38.2 MB) -> shard_0003.pt
  ✓ Saved Shard 0004: 2,500 families (~38.2 MB) -> shard_0004.pt
  ✓ Saved Shard 0005: 1,000 families (~15.3 MB) -> shard_0005.pt
✓ Sharding Completed!
  Validation Sets: val_iid.pt, val_comp.pt generated.
================================================================================
```

---

### Cell 7: Audit & Verify All Shards
*Runs the health scorecard to verify tensor shapes, loss weights, and Fourier numbers.*

```python
# CELL 7: Run Shard Health Audit
!python /content/drive/MyDrive/geninherit/code/05_verify_phase1_shards.py \
    /content/drive/MyDrive/geninherit/data/shards
```

**Expected Output:**
```text
================================================================================
  PHASE 1 FINAL READINESS SCORECARD
================================================================================
  Total Shards Audited     : 8 (6 training shards + 2 validation sets)
  Total Binary Shard Size  : ~235 MB
  Validation Sets Present  : True (val_iid.pt, val_comp.pt)
  Status                   : 100% READY FOR PHASE 2 MODEL TRAINING ON COLAB!
================================================================================
```

---

## PART 4: Verification Checklist (How to Know You Are Done)

Go to your Google Drive in another tab and verify these files exist:
1. `My Drive/geninherit/bioDSL_vocab.json` (Should be ~120 KB).
2. `My Drive/geninherit/data/shards/` contains:
   - `shard_0000.pt` through `shard_0005.pt`
   - `val_iid.pt`
   - `val_comp.pt`

**When all files are verified, Phase 1 is 100% complete! You are ready to launch Phase 2 (Base Model Pre-Training).**

