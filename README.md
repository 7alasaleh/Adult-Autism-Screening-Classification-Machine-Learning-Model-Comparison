# Adult Autism Screening Classification: Machine Learning Model Comparison

## 1. Project Overview

This project was developed as part of a **Big Data (Büyük Veri)** university course. Its objective is to preprocess the Adult Autism Screening dataset, compare six machine learning classifiers, and evaluate two ensemble learning approaches: Soft Voting and Weighted Voting.

The project includes missing-value imputation, categorical encoding, feature scaling, cross-validation, model evaluation, confusion matrices, ROC curves, and comparative visualizations.

The target variable is `Class/ASD`, which indicates the questionnaire-based classification of an individual as `YES` or `NO`.

> **Important limitation:** The target label is derived from the questionnaire scores included in the dataset. Therefore, the models primarily learn to reconstruct the questionnaire-based classification. The results must not be interpreted as evidence of independent clinical diagnostic performance.

## 2. Objectives

The main objectives are to:

- Inspect and understand the dataset.
- Identify missing values and handle them without deleting records.
- Prepare numerical and categorical features for machine learning.
- Compare StandardScaler and MinMaxScaler.
- Compare six individual classification algorithms.
- Evaluate Soft Voting and Weighted Voting ensembles.
- Measure model performance using multiple evaluation metrics.
- Generate tables, charts, confusion matrices, and classification reports.
- Document the results for reproducibility and further development.

## 3. Dataset

The project uses the **Adult Autism Screening dataset**, distributed through Kaggle and originally associated with the UCI Adult Autism Screening dataset.

The dataset contains questionnaire responses and demographic or background information.

### Dataset summary

| Property | Value |
|---|---:|
| Initial number of records | 704 |
| Initial number of columns | 21 |
| Target variable | `Class/ASD` |
| `NO` records | 515 (73.15%) |
| `YES` records | 189 (26.85%) |
| Features after removing excluded columns | 18 |
| Records removed during cleaning | 0 |

The original class distribution is moderately imbalanced, with more `NO` than `YES` records. Consequently, Accuracy alone is not sufficient to assess model performance.

### Features

The questionnaire contains ten binary score features:

- `A1_Score` through `A10_Score`

Other features include age, gender, ethnicity, jaundice history, autism-related background information, country of residence, previous screening information, and relationship to the individual completing the screening.

The target column is `Class/ASD`.

### Target construction and feature selection

The dataset's `result` column represents the sum of `A1_Score` through `A10_Score`. The target classification is derived from this score:

- Scores from 0 to 6 correspond to `NO`.
- Scores from 7 to 10 correspond to `YES`.

To avoid including a direct duplicate of the target-generating score, the `result` column was excluded from the feature set. The ten individual questionnaire scores were retained.

The constant `age_desc` column was also removed because it does not provide useful discriminatory information.

This feature-selection decision is important to document: although `result` is excluded, the individual questionnaire scores still contain the information used to derive the target.

## 4. Data Cleaning and Preprocessing

All 704 records were retained. Missing and invalid values were handled through imputation rather than deleting rows.

### Missing-value analysis

| Feature | Missing or invalid values |
|---|---:|
| `ethnicity` | 95 |
| `relation` | 95 |
| `age` | 3 |

The age value of 383 was treated as invalid and converted to a missing value. Two additional age values were already missing.

The literal `?` values were converted to proper missing values before preprocessing.

### Imputation strategy

| Feature type | Imputation method |
|---|---|
| Numerical features | Median |
| Categorical features | Most frequent value (mode) |

The imputation strategy was compared with alternative approaches using cross-validation. The selected median-and-mode configuration was simple, interpretable, and tied for the best observed mean Macro F1 score.

### Imputation comparison

The following results were obtained using Random Forest and five-fold stratified cross-validation on the training data.

| Imputation configuration | Mean Accuracy | Mean Macro F1 | Mean Balanced Accuracy | Std. Macro F1 |
|---|---:|---:|---:|---:|
| Median + Mode + Mode | 0.9591 | 0.9462 | 0.9344 | 0.0221 |
| Mean + Mode + Mode | 0.9591 | 0.9462 | 0.9344 | 0.0221 |
| Mean + Unknown + Unknown | 0.9591 | 0.9462 | 0.9343 | 0.0209 |
| Median + Unknown + Unknown | 0.9591 | 0.9462 | 0.9343 | 0.0209 |

The selected configuration was median imputation for age and most-frequent-value imputation for categorical features. The results were effectively tied with the best alternatives, so the simpler, interpretable configuration was retained.

### Categorical encoding

Categorical features were transformed using `OneHotEncoder` with `handle_unknown="ignore"` so that previously unseen categories in validation or test data would not cause encoding errors.

### Feature scaling

Two scaling methods were compared:

- **StandardScaler:** Standardizes numerical features using the training data's mean and standard deviation.
- **MinMaxScaler:** Scales numerical features to a specified range, normally 0 to 1.

Preprocessing was implemented through Scikit-learn pipelines and a `ColumnTransformer`. This ensures that imputation, encoding, and scaling are fitted within each training fold during cross-validation, reducing the risk of preprocessing leakage.

## 5. Train-Test Split and Validation

The dataset was divided using a stratified train-test split:

| Parameter | Value |
|---|---|
| Training proportion | 80% |
| Test proportion | 20% |
| Random state | 42 |
| Split method | Stratified |
| Cross-validation method | `StratifiedKFold` |
| Number of folds | 5 |

The final test set contained 141 records:

| Test-set class | Number of records |
|---|---:|
| `NO` | 103 |
| `YES` | 38 |
| **Total** | **141** |

Cross-validation was performed on the training set during model comparison. The held-out test set was then used for the final evaluation.

The test results should be interpreted in the context of the target construction and the model-selection process described in this README.

## 6. Machine Learning Algorithms

Six individual classifiers were evaluated.

| Algorithm | Description |
|---|---|
| Support Vector Machine (SVM) | Finds a decision boundary that separates classes, with `probability=True` enabled for Soft Voting and ROC-AUC evaluation. |
| K-Nearest Neighbors (KNN) | Predicts a class based on nearby training samples. |
| Random Forest | Combines predictions from multiple decision trees. |
| Gaussian Naive Bayes | Uses Bayes' theorem with a Gaussian likelihood assumption for numerical features. |
| Decision Tree | Learns a sequence of feature-based decision rules. |
| AdaBoost | Builds an ensemble of weak learners, giving more attention to difficult training examples. |

The ensemble methods were:

- **Soft Voting:** Averages the predicted class probabilities from the six classifiers.
- **Weighted Voting:** Combines class probabilities using predefined weights.

The Weighted Voting weights were:

| Classifier | Weight |
|---|---:|
| SVM | 5 |
| KNN | 4 |
| Random Forest | 3 |
| Gaussian Naive Bayes | 1 |
| Decision Tree | 2 |
| AdaBoost | 6 |

The weights were informed by the earlier cross-validation experiments. They were not tuned using the test set.

## 7. Evaluation Metrics

Multiple metrics were used to evaluate the classifiers, especially because the class distribution is imbalanced.

| Metric | Meaning |
|---|---|
| Accuracy | Proportion of all predictions that are correct. |
| Macro F1 Score | Average F1 score across both classes, giving each class equal weight. |
| F1 Score (`YES`) | Harmonic mean of Precision and Recall for the positive class. |
| Balanced Accuracy | Average of Sensitivity and Specificity. |
| Precision (`YES`) | Proportion of positive predictions that are correct. |
| Sensitivity / Recall | Proportion of actual `YES` cases correctly identified. |
| Specificity | Proportion of actual `NO` cases correctly identified. |
| FPR | False Positive Rate: proportion of actual `NO` cases incorrectly predicted as `YES`. |
| FDR | False Discovery Rate: proportion of `YES` predictions that are incorrect. |
| FNR | False Negative Rate: proportion of actual `YES` cases incorrectly predicted as `NO`. |
| ROC-AUC | Measures the ability to distinguish between the two classes across classification thresholds. |
| Confusion Matrix | Shows true negatives, false positives, false negatives, and true positives. |

For the confusion matrix and positive-class metrics, `YES` was treated as the positive class.

The confusion matrix is structured as follows:

| | Predicted NO | Predicted YES |
|---|---:|---:|
| Actual NO | TN | FP |
| Actual YES | FN | TP |

Accuracy was calculated both from the confusion matrix and using Scikit-learn's `accuracy_score`. The two calculations should agree.

## 8. Cross-Validation Results: Scaling Comparison

The following table summarizes selected results from the five-fold cross-validation experiments comparing scaling methods.

| Scaling | Model | Mean Accuracy | Mean Macro F1 | Mean Balanced Accuracy | Mean ROC-AUC |
|---|---|---:|---:|---:|---:|
| StandardScaler | AdaBoost | 0.9982 | 0.9977 | 0.9967 | 1.0000 |
| MinMaxScaler | AdaBoost | 0.9982 | 0.9977 | 0.9967 | 1.0000 |
| StandardScaler | SVM | 0.9752 | 0.9686 | 0.9725 | 0.9978 |
| MinMaxScaler | SVM | 0.9698 | 0.9624 | 0.9731 | 0.9966 |
| StandardScaler | KNN | 0.9538 | 0.9421 | 0.9475 | 0.9917 |
| StandardScaler | Random Forest | 0.9538 | 0.9388 | 0.9244 | 0.9926 |
| MinMaxScaler | KNN | 0.9449 | 0.9314 | 0.9413 | 0.9794 |
| StandardScaler | Decision Tree | 0.8845 | 0.8554 | 0.8604 | 0.8604 |
| StandardScaler | Gaussian Naive Bayes | 0.3215 | 0.3034 | 0.5027 | 0.5084 |

### Interpretation

- AdaBoost achieved the strongest cross-validation results with both scaling methods.
- SVM and KNN performed better in Macro F1 with StandardScaler than with MinMaxScaler in these experiments.
- Decision Tree and Random Forest are generally less sensitive to feature scaling than distance-based algorithms.
- Gaussian Naive Bayes performed poorly on this representation of the data. Its Gaussian feature-distribution assumption may be unsuitable for the combination of binary questionnaire responses and one-hot-encoded categorical features.

These findings describe the tested configurations; they do not establish that one scaling method is universally superior.

## 9. Soft Voting vs. Weighted Voting

Both ensemble approaches were evaluated with five-fold stratified cross-validation on the training set.

| Model | Mean Accuracy | Mean Macro F1 | Mean Balanced Accuracy | Mean ROC-AUC | Std. Macro F1 |
|---|---:|---:|---:|---:|---:|
| Weighted Voting | 0.9716 | 0.9640 | 0.9660 | 0.9960 | 0.0083 |
| Soft Voting | 0.9414 | 0.9278 | 0.9411 | 0.9905 | 0.0180 |

Weighted Voting outperformed Soft Voting on the reported cross-validation metrics and showed lower fold-to-fold variation in Macro F1.

This suggests that weighting the classifiers differently was beneficial for the selected configuration. However, the weights were informed by earlier cross-validation results, so these values should be viewed as model-development results rather than an entirely independent comparison.

## 10. Final Test-Set Results

The eight models were fitted on the training data and evaluated on the held-out test set.

### Overall comparison

The table is sorted by Macro F1 score.

| Model | Accuracy | Macro F1 | Balanced Accuracy | ROC-AUC |
|---|---:|---:|---:|---:|
| AdaBoost | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Weighted Voting | 0.9787 | 0.9728 | 0.9688 | 0.9969 |
| SVM | 0.9716 | 0.9634 | 0.9557 | 0.9985 |
| KNN | 0.9645 | 0.9560 | 0.9674 | 0.9950 |
| Soft Voting | 0.9504 | 0.9394 | 0.9577 | 0.9936 |
| Random Forest | 0.9291 | 0.9011 | 0.8684 | 0.9922 |
| Decision Tree | 0.8511 | 0.8093 | 0.8067 | 0.8067 |
| Gaussian Naive Bayes | 0.3121 | 0.2838 | 0.5125 | 0.5125 |

### Confusion matrix counts

| Model | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| AdaBoost | 103 | 0 | 0 | 38 |
| Weighted Voting | 102 | 1 | 2 | 36 |
| SVM | 102 | 1 | 3 | 35 |
| KNN | 99 | 4 | 1 | 37 |
| Soft Voting | 97 | 6 | 1 | 37 |
| Random Forest | 103 | 0 | 10 | 28 |
| Decision Tree | 93 | 10 | 11 | 27 |
| Gaussian Naive Bayes | 8 | 95 | 2 | 36 |

### Interpretation

**AdaBoost** achieved perfect classification on the held-out test set, with 103 true negatives and 38 true positives. No false positives or false negatives were observed.

**Weighted Voting** was the second-best model by Accuracy and Macro F1, correctly classifying 138 of 141 test records.

**SVM** obtained the highest ROC-AUC among the models in this table at 0.9985, slightly above Weighted Voting's 0.9969. This illustrates why ROC-AUC and classification-threshold metrics can produce different rankings.

**KNN** achieved a high Macro F1 score and Balanced Accuracy, with only one false negative and four false positives.

**Random Forest** had relatively high Accuracy but missed 10 of the 38 actual `YES` cases, giving it a Sensitivity of approximately 73.68%.

**Decision Tree** performed less strongly than the leading models across the main metrics.

**Gaussian Naive Bayes** performed poorly overall and produced many false-positive predictions. Its Accuracy was only 31.21%.

### Why did AdaBoost achieve 100%?

The target label is derived from the questionnaire score, and the individual score features were retained. This allows a model to reconstruct the target-generating rule from the available features.

Therefore, AdaBoost's perfect score is a result observed on this particular dataset and test split. It should not be interpreted as proof of perfect generalization to new populations or as evidence of clinical diagnostic accuracy.

## 11. Visualizations and Generated Artifacts

The notebook generates comparison charts, confusion matrices, classification reports, and CSV files for further analysis.

The outputs are saved under:

`/kaggle/working/project_outputs/`

The main generated artifacts include:

| File or folder | Description |
|---|---|
| `voting_comparison.csv` | Cross-validation comparison of Soft Voting and Weighted Voting. |
| `test_metrics_summary.csv` | Detailed final test metrics for all eight models. |
| `final_model_comparison.csv` | Final comparison table. |
| `final_model_ranking.csv` | Compact model ranking table. |
| `comparison_main_metrics.png` | Comparison of Accuracy, Macro F1, and Balanced Accuracy. |
| `comparison_sensitivity_specificity.png` | Comparison of Sensitivity and Specificity. |
| `comparison_error_rates.png` | Comparison of FPR, FDR, and FNR. |
| `comparison_roc_auc.png` | ROC-AUC comparison chart. |
| `roc_curves_test_set.png` | ROC curves for all eight models. |
| `test_f1_macro_comparison.png` | Test-set Macro F1 comparison. |
| `confusion_matrices/` | Individual confusion matrix images. |
| `classification_reports/` | Individual classification reports. |
| `final_comparison_summary.txt` | Text summary of the final comparison. |

Only include files and folders in the GitHub repository that you have actually exported or copied from the Kaggle working directory.

## 12. Tools and Libraries

The project uses Python and common machine learning and data analysis libraries:

- **Pandas** — data manipulation and tabular results.
- **NumPy** — numerical operations.
- **SciPy** — reading the ARFF dataset.
- **Matplotlib** — visualizations.
- **Scikit-learn** — preprocessing, model training, cross-validation, metrics, and ensemble learning.

The project was developed in a Kaggle Notebook environment.

## 13. Reproducibility

The experiment uses fixed random states where applicable, including `random_state=42` for the train-test split and relevant model configurations.

To reproduce the project:

1. Obtain the Adult Autism Screening dataset from the configured Kaggle dataset source.
2. Load the ARFF file and perform the documented cleaning steps.
3. Split the dataset into stratified training and test sets.
4. Run the imputation and scaling comparisons.
5. Evaluate the six individual classifiers using five-fold stratified cross-validation.
6. Compare Soft Voting and Weighted Voting on the training data.
7. Fit the selected model configurations on the training set.
8. Evaluate the models on the held-out test set.
9. Export the comparison tables, reports, and visualizations.

Exact results can vary with changes in library versions, preprocessing, random seeds, or model configurations.

## 14. Limitations and Responsible Interpretation

Several limitations should be considered when interpreting the results:

1. **Derived target:** `Class/ASD` is derived from the questionnaire scores included as features. The task therefore measures reconstruction of a questionnaire-based label, not independent autism diagnosis.
2. **Dataset representativeness:** Performance on this dataset does not guarantee performance on different populations or data collection settings.
3. **Class imbalance:** The `NO` class is more common than the `YES` class, which is why Macro F1, Balanced Accuracy, Sensitivity, and Specificity are reported alongside Accuracy.
4. **Model selection:** Cross-validation results informed model and weighting choices. Once the held-out test results have been examined, the same test set should not be reused to tune models while continuing to present its scores as an untouched final evaluation.
5. **Gaussian Naive Bayes assumptions:** The tested Gaussian Naive Bayes configuration may not be well suited to the feature representation.
6. **No clinical validation:** The models have not been validated as clinical decision-support systems and should not be used to diagnose individuals.

Future work could investigate alternative validation strategies, independent datasets, calibration, feature-importance analysis, and evaluation on data whose target labels are established independently of the input questionnaire scores.

## 15. Conclusion

This project provides a complete machine learning workflow for the Adult Autism Screening dataset, including data cleaning, imputation, feature encoding, scaling experiments, cross-validation, ensemble learning, test-set evaluation, and visualization.

AdaBoost achieved the strongest numerical results on the held-out test set, followed by Weighted Voting and SVM. However, because the target is derived from the retained questionnaire scores, the high performance primarily reflects the ability to reproduce the questionnaire's classification rule.

The main contribution of the project is the systematic comparison of multiple machine learning approaches and the documentation of their performance using several complementary metrics, rather than relying on Accuracy alone.

---

**Project type:** University course project — Big Data (Büyük Veri)  
**Task:** Binary classification  
**Dataset:** Adult Autism Screening  
**Environment:** Kaggle Notebook  
**Language:** Python

### Experiment: Classification Without Questionnaire Scores

To investigate the effect of removing questionnaire-related features, a second experiment was conducted using only eight demographic and background features: `age`, `gender`, `ethnicity`, `jundice`, `austim`, `contry_of_res`, `used_app_before`, and `relation`.

The questionnaire score features (`A1_Score`–`A10_Score`) were removed, while `result` remained excluded as in the original experiment. The target variable, train-test split, preprocessing pipeline, classifier configurations, and weighted voting coefficients were kept unchanged to make the comparison as consistent as possible. Unlike the initial experiment, this experiment evaluated the models directly on the held-out test set without repeating the preprocessing and scaling comparisons.

The results showed a substantial decrease in performance for most classifiers. SVM achieved the highest Macro F1-score (0.6464), while Random Forest achieved the highest accuracy (75.18%). However, Random Forest's sensitivity was only 31.58%, demonstrating that accuracy alone can be misleading when evaluating imbalanced classification problems. Weighted Voting achieved a Macro F1-score of 0.5932, indicating that its original weighting scheme did not retain the same performance after questionnaire features were removed.

These findings suggest that questionnaire responses strongly influence the prediction of the target label in this dataset. Since the target was derived from questionnaire scores, neither experiment should be interpreted as evidence of independent clinical diagnostic performance. Further evaluation using an independently established clinical target would be required to investigate that question.
