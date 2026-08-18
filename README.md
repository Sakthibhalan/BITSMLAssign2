# BITSMLAssign2
BITS WILP M.Tech AI/ML ML-Assignment 2 for Semester 1

## 1. Problem Statement
The purpose is to setup multi-suite machine learning classification models to accurately diagnose breast cancer, using features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass, the models predict whether a tumor is malignant (cancerous) or benign (non-cancerous). This dashboard allows for real-time evaluation and comparison of these models to determine the most effective diagnostic tool.

## 2. Dataset Description
* **Dataset:** Breast Cancer Wisconsin (Diagnostic)
* **Instances:** 569
* **Features:** 30 numeric predictive attributes (e.g., radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension).
* **Target Variable:** `diagnosis` (Binary: 1 = Malignant, 0 = Benign)

---

## 3. Breast Cancer Diagnostics - ML Classifier Dashboard

**Live Streamlit App:** https://bitsmlassign2-xkfes4f7dhqumdppdapprbt.streamlit.app/
**GitHub Repository:** https://github.com/Sakthibhalan/BITSMLAssign2

---

## 4. Model Comparison Matrix

| Model | Accuracy | AUC Score | Precision | Recall | F1 Score | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.9561 | 0.9977 | 0.9750 | 0.9070 | 0.9398 | 0.9068 |
| **Decision Tree** | 0.9474 | 0.9440 | 0.9302 | 0.9302 | 0.9302 | 0.8880 |
| **K-Nearest Neighbors** | 0.9561 | 0.9959 | 1.0000 | 0.8837 | 0.9383 | 0.9086 |
| **Naive Bayes** | 0.9737 | 0.9984 | 1.0000 | 0.9302 | 0.9639 | 0.9447 |
| **Random Forest** | 0.9649 | 0.9953 | 0.9756 | 0.9302 | 0.9524 | 0.9253 |



---

## 5. Observations & Conclusion

### Key Observations per Model:
In the context of cancer diagnosis, **Recall** (Sensitivity) is the most critical metric. Minimizing False Negatives (predicting a tumor is benign when it is actually malignant) is heavily prioritized to ensure patients receive life-saving treatment.

* **Logistic Regression:** Showed excellent class separability with an AUC of 0.9977. But, it struggled with false negatives, resulting in a lower Recall of 0.907.
* **Decision Tree:** While it achieved a Recall of 0.9302, it performed the poorest overall across all other metrics, scoring the lowest Accuracy (0.9474), AUC (0.9440), and MCC (0.8880), indicating it is the least robust model of the suite.
* **K-Nearest Neighbors (KNN):** Achieved perfect Precision (1.0000), meaning every tumor it flagged as malignant was indeed malignant. However, this came at the cost of the lowest Recall in the entire group (0.8837). In a clinical setting, missing actual positive cancer cases makes this model too risky to deploy.
* **Random Forest:** Proved to be a very strong, well-rounded ensemble model. It also had a highest Recall (0.9302) while maintaining high Accuracy (0.9649) and a high AUC (0.9953), successfully smoothing out the variances seen in the standalone Decision Tree.
* **Naive Bayes:** Delivered exceptional performance across the board. It achieved perfect Precision (1.0000), the highest Accuracy (0.9737), the highest AUC (0.9984), the highest F1 Score (0.9639), the highest MCC (0.9447), and tied for the highest Recall (0.9302). 

### Overall Winner
Based on the suggested evaluation metrics, we see that Naive Bayes should be declared the overall winner.

It not only achieved the highest overall predictive accuracy (97.37%) and structural correlation (MCC of 0.9447), but it successfully maximized Recall (93.02%) while maintaining perfect Precision (100%). This means it is highly effective at identifying malignant tumors without raising any false alarms, making it the safest and most reliable diagnostic classifier for this dataset.
