# Task 38 – Classification Accuracy (Iris Dataset)

**Veda Technology · AI & ML Track · Level 1 · Day 38**

## Objective
Calculate the accuracy of a classification model and compare predicted values with actual target values, to understand one of the most basic classification evaluation metrics.

## Tools Used
- Python
- Pandas
- Scikit-learn
- Jupyter Notebook

## Dataset
Iris dataset from `sklearn.datasets.load_iris`.

- 150 flowers, 4 numerical features (sepal length, sepal width, petal length, petal width)
- Target: species with 3 classes (setosa, versicolor, virginica), 50 samples each
- A **multiclass** classification problem, with balanced classes

## Files in this Repository
| File | Description |
|------|-------------|
| `classification_accuracy.ipynb` | Notebook with training, predictions and accuracy |
| `iris.csv` | Dataset used |
| `README.md` | Approach, results and interpretation |

## 1. Model Setup
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```
The data was split 80/20 (120 training rows, 30 test rows). `stratify=y` keeps 10 test samples of each species, and `random_state=42` makes the result reproducible. A Logistic Regression model was trained, and accuracy was measured only on the 30 unseen test rows.

## 2. Actual vs Predicted Values
Actual and predicted species were compared row by row:

```python
results = pd.DataFrame({
    "actual": data.target_names[y_test.values],
    "predicted": data.target_names[y_pred],
})
results["correct"] = results["actual"] == results["predicted"]
```

Out of 30 test samples, **29 predictions matched** the actual species and **1 did not**: one *versicolor* flower was predicted as *virginica*. The full table is shown in the notebook output.

## 3. Accuracy Score
```python
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
```

**Accuracy = correct predictions / total predictions = 29 / 30 = 0.9667**

| Metric | Value |
|--------|-------|
| Correct predictions | 29 |
| Total test samples | 30 |
| Accuracy (decimal) | 0.9667 |
| **Accuracy (percentage)** | **96.67%** |
| Training accuracy | 97.5% |

A manual calculation (`correct / len(y_test)`) gives the same value as `accuracy_score`.

## 4. Confusion Matrix (extra)

| | Predicted setosa | Predicted versicolor | Predicted virginica |
|---|---|---|---|
| **Actual setosa** | 10 | 0 | 0 |
| **Actual versicolor** | 0 | 9 | 1 |
| **Actual virginica** | 0 | 0 | 10 |

## 5. Interpretation of the Model Performance
- **The model performs well.** With 3 balanced classes, random guessing would give about 33% accuracy. A score of 96.67% shows that the model has learned the patterns in the measurements.
- **The only mistake is between versicolor and virginica.** Setosa was classified perfectly (10 of 10), and so was virginica (10 of 10). One versicolor flower was confused with virginica, which is expected because their petal measurements partly overlap.
- **No sign of overfitting.** Training accuracy (97.5%) and test accuracy (96.67%) are close.
- **Limitation:** The test set has only 30 rows, so a single mistake changes the accuracy by about 3.3%. A different train-test split can give a different value (a test split of 30% gave 93.3% in a quick check). The accuracy should be read as an estimate, not an exact value.

## Key Learnings
- Accuracy is the share of predictions that are correct, and it can be written as a decimal or a percentage.
- Always compare accuracy with a baseline (here about 33% for random guessing).
- Looking at the wrong predictions shows *where* the model fails, which accuracy alone does not.
- A small test set makes accuracy unstable.

## Interview Questions – Key Answers
1. **What is classification accuracy?** The percentage of samples the model classifies correctly out of all samples tested.
2. **How is accuracy calculated?** `accuracy = correct predictions / total predictions`. Here, 29 / 30 = 0.9667 = 96.67%.
3. **Can accuracy always be trusted as the only evaluation metric?** No. When classes are imbalanced, accuracy can be misleading. For example, if 95% of samples belong to one class, a model that always predicts that class gets 95% accuracy while never detecting the other class. Metrics such as the confusion matrix, precision and recall should also be checked, especially in risky problems such as medical diagnosis.

## How to Run
```bash
pip install pandas scikit-learn jupyter
jupyter notebook classification_accuracy.ipynb
```
