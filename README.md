# Visual Question Answering on CLEVER Dataset

This repository implements a Visual Question Answering (VQA) system evaluated on the **CLEVER dataset** (COL774 A4 assignment). The task involves answering questions about images that require reasoning about object properties and relationships.

---

## Dataset

**CLEVER Dataset (COL774 A4):**
Available on Kaggle: [https://www.kaggle.com/datasets/aayushkt/col774-a4-dataset](https://www.kaggle.com/datasets/aayushkt/col774-a4-dataset)

**Directory Structure:**

```
CLEVR_COL774_A4/
│
├── images/
│   ├── testA/
│   │   ├── <image1>.png
│   │   └── ...
│   ├── testB/
│   │   ├── <image2>.png
│   │   └── ...
│   ├── trainA/
│   │   ├── <image3>.png
│   │   └── ...
│   ├── valA/
│   │   ├── <image4>.png
│   │   └── ...
│   └── valB/
│       ├── <image5>.png
│       └── ...
│
└── questions/
    ├── CLEVR_testA_questions.json
    ├── CLEVR_testB_questions.json
    ├── CLEVR_trainA_questions.json
    ├── CLEVR_valA_questions.json
    └── CLEVR_valB_questions.json
```

---

## Getting Started

### Requirements

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### Running Inference

**Test A and Test B:**

```bash
python3 part11.py \
  --model_path <path_to_checkpoint> \
  --mode inference \
  --dataset <path_to_dataset_folder>
```

**All metrics for Dataset A:**

```bash
python3 part10b.py \
  --model_path <path_to_checkpoint> \
  --mode inference \
  --dataset <path_to_dataset_folder>
```

---

## Evaluation

* Accuracy metrics are computed separately for Test A and Test B.
* `part11.py` and `part10b.py` contain evaluation logic for respective test splits and overall performance.

---

## 📎 Citation

If you use this codebase or dataset, please consider citing the original CLEVER dataset authors.

---

## 📬 Contact

For questions, feel free to open an issue or contact the maintainer.
