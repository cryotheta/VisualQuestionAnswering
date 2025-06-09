# 🧠 Visual Question Answering on CLEVER Dataset

This repository implements a Visual Question Answering (VQA) system evaluated on the **CLEVER dataset**. The task involves answering questions about images that require reasoning about object properties and relationships.

---

## 📁 Dataset

**CLEVER Dataset:**
Available on Kaggle: [https://www.kaggle.com/datasets/aayushkt/col774-a4-dataset](https://www.kaggle.com/datasets/aayushkt/col774-a4-dataset)

**Structure:**

```
dataset_folder/
│
├── images/
│   ├── <image1>.png
│   ├── ...
│
└── questions/
    ├── questions_test_A.json
    ├── questions_test_B.json
    └── questions_train.json
```

---

## 🚀 Getting Started

### 🔧 Requirements

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 🏁 Running Inference

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

## 📊 Evaluation

* Accuracy metrics are computed separately for Test A and Test B.
* `part11.py` and `part10b.py` contain evaluation logic for respective test splits and overall performance.

---

## 📎 Citation

If you use this codebase or dataset, please consider citing the original CLEVER dataset authors.

---

## 📬 Contact

For questions, feel free to open an issue or contact the maintainer.
