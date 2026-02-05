# Machine Learning Project: Stroke Risk Prediction

## Overview
This project develops and evaluates multiple machine learning models to predict the likelihood of stroke using structured healthcare data. The work emphasizes methodological rigor, reproducibility, and comparative evaluation under severe class imbalance, aligning with best practices in applied data science and biomedical machine learning.

The primary objective is binary classification of the target variable **`stroke`**, motivated by the clinical importance of early stroke risk identification to support preventive interventions and improved patient outcomes.

The dataset used in this study is publicly available from Kaggle under the title *Healthcare Dataset Stroke Data*.

---

## Dataset Description
- **Observations:** 5,110 patients  
- **Features:** 12 demographic and clinical variables  
- **Target:** `stroke` (binary, highly imbalanced)

Key predictors include age, hypertension, heart disease, body mass index (BMI), average glucose level, smoking status, and sociodemographic indicators. Exploratory analysis revealed **age** as the strongest correlated feature with stroke occurrence, consistent with established clinical knowledge. Hypertension and heart disease also exhibited positive associations, supporting their inclusion as meaningful clinical risk indicators.

The dataset exhibits substantial class imbalance, with approximately 95% of observations corresponding to non-stroke cases and only 5% representing stroke events.

---

## Exploratory Data Analysis
Exploratory analysis was conducted to assess feature distributions, missingness, and correlations with the target variable. Histograms were used to identify skewness in continuous variables, while a correlation heatmap highlighted relationships between predictors and stroke occurrence.

Age demonstrated the strongest correlation with stroke, followed by hypertension and heart disease. Although BMI and average glucose level showed weaker linear correlations, their established medical relevance justified their inclusion, particularly for models capable of capturing non-linear effects.

---

## Methods

### Study Design
This study follows a supervised machine learning framework for binary classification. Model development and evaluation were conducted using pipeline-based workflows to ensure reproducibility, minimize data leakage, and enable fair comparison across modeling strategies.

### Data Splitting Strategy
The dataset was partitioned into training (64%), validation (16%), and test (20%) sets using stratified sampling to preserve the underlying class distribution. The validation set was used for hyperparameter tuning and model selection, while the test set was reserved for final performance assessment.

### Preprocessing and Feature Engineering
All preprocessing steps were encapsulated within pipeline objects. Numerical features (age, BMI, average glucose level) were standardized using z-score normalization. Categorical variables (gender, marital status, work type, residence type, and smoking status) were encoded using one-hot encoding with safeguards for unseen categories.

Missing values in the BMI feature were imputed using the median to preserve distributional properties without introducing bias. Non-informative identifier fields were removed prior to modeling.

### Class Imbalance Handling
Given the severe class imbalance, the Synthetic Minority Over-sampling Technique (SMOTE) was employed to generate synthetic minority class samples. SMOTE was applied **only within training folds** using `imblearn` pipelines to prevent data leakage. Both unbalanced and SMOTE-balanced configurations were evaluated to assess the impact of resampling on model performance.

### Model Training and Hyperparameter Optimization
Six classification models were evaluated:
- Logistic Regression  
- Decision Tree  
- Support Vector Machine (SVM)  
- Random Forest  
- Gradient Boosting  
- Feedforward Neural Network  

Hyperparameter tuning was performed using grid search with cross-validation for models sensitive to parameter selection, particularly ensemble methods and neural networks. Optimization was guided by weighted F1-score to account for class imbalance.

### Evaluation Metrics
Model performance was assessed using:
- **Weighted F1-score**, to balance precision and recall across imbalanced classes  
- **ROC AUC**, to measure discrimination ability independent of classification thresholds  

Metrics were computed on the validation set for model comparison, with final assessment performed on the held-out test set.

---

## Results Summary
Validation performance varied substantially across models and balancing strategies:

| Model | Validation F1 | Validation ROC AUC |
|------|--------------|-------------------|
| Gradient Boosting (Unbalanced) | 0.9272 | **0.8763** |
| Logistic Regression (Unbalanced) | 0.9272 | 0.8596 |
| Logistic Regression (Balanced) | 0.8052 | 0.8547 |
| Random Forest (Balanced) | 0.8863 | 0.8013 |
| Neural Network (Unbalanced) | 0.9210 | 0.7432 |
| SVM (Unbalanced) | 0.9272 | 0.3900 |

**Key observations:**
- Gradient Boosting achieved the strongest ROC AUC, demonstrating robust discrimination without oversampling.
- Logistic Regression performed competitively while offering interpretability, an important consideration for clinical applications.
- SMOTE did not universally improve performance, highlighting the need for model-specific imbalance strategies.
- High F1 scores paired with low ROC AUC in some models suggest sensitivity to class imbalance and potential overfitting.

---

## Conclusions
This study demonstrates that carefully tuned ensemble models can outperform more complex architectures when applied to imbalanced clinical datasets. Gradient Boosting and Random Forest models exhibited strong generalization, while Logistic Regression provided a transparent and interpretable baseline with competitive performance.

The results underscore the importance of pipeline-based preprocessing, thoughtful imbalance handling, and metric selection in healthcare-focused machine learning applications.

---

## Future Work
Planned extensions include:
- Improved integration of SMOTE within nested cross-validation
- Feature interaction and domain-informed feature engineering
- Model calibration analysis for clinical deployment
- Integration of explainability methods such as SHAP

---

## Reproducibility
All preprocessing, resampling, and modeling steps are implemented using reproducible pipelines. The complete workflow is contained within the provided Jupyter notebook, ensuring transparency and repeatability of results.

---

## Author
**Rumbidzai Mushamba**  
MSc Applied Data Science, Clarkson University
