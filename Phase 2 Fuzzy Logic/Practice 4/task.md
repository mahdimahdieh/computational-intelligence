# Fuzzy Rule-Based Classification for Predicting Student Academic Outcomes

---

## Project Goal:
Design and implement a fuzzy rule-based classification model to predict student academic outcomes (graduate, dropout, or enrolled) using a dataset with demographic, academic, and socio-economic features. The model must:

- Use fuzzy logic to represent features.
- Employ a genetic algorithm (GA) to optimize rule selection and improve interpretability.

**Note:** The implementation must be done **from scratch** without using libraries like `skfuzzy`.

---

## Project Tasks:

### 1. Data Processing and Exploration:
- Load the dataset and resolve any formatting issues.
- Preprocess the data (handle missing values, convert data types, etc.).
- Conduct exploratory data analysis (EDA) and visualize key patterns.
- Split the dataset into training (80%) and testing (20%) sets.

**Expected Output:**
- A concise EDA report with observations and a cleaned, split dataset.

**Feature Selection Guide:**
- Not all features need to be included in the model.
- Retain only the most relevant features.
- Use a minimum of 10 features for the model.

**Feature Selection Methods:**
- **Correlation Analysis:**  
  - Select features with high correlation to the target variable.  
  - Remove multicollinear features.
- **Information Gain/Mutual Information:**  
  - Use functions like `mutual_info_classif` from `scikit-learn`.
- **Simple Statistical Models:**  
  - Run basic models (e.g., decision trees) to evaluate feature importance and retain high-importance features.

---

### 2. Fuzzification of Features:
- Apply triangular membership functions to continuous features (e.g., admission grade, unemployment rate).
- Assign fuzzy labels (low, medium, high) to features.

**Expected Output:**
- Fuzzification code for continuous and categorical features.
- Plots of membership functions for at least two continuous features.

**Guide for Fuzzifying Binary Features:**
- **Crisp-like Fuzzification:**  
  - For binary features (e.g., "International"), assign full membership (1) to the respective set (e.g., "International" or "Non-International").  
  - Suitable for true binary features like `Gender`, `International`, `Debtor`, `Scholarship holder`, etc.
- **Soft Fuzzification:**  
  - Use continuous membership functions (e.g., sigmoid) for probabilistic or uncertain models.  
  - For this project, crisp-like fuzzification is recommended for true binary features.

---

### 3. Fuzzy Rule Extraction (Wang-Mendel Method):
- Generate fuzzy IF-THEN rules from the training data.
- Each rule includes an antecedent and a consequent (class label) with a confidence weight.
- Retain the strongest rule for each unique antecedent set.

**Expected Output:**
- Code for rule generation and weight calculation.
- A set of extracted fuzzy rules with interpretation.

---

### 4. Rule Selection Using Genetic Algorithm (GA):
- Encode rules as binary chromosomes.
- Define a fitness function that rewards classification accuracy and penalizes excessive rule count.
- Execute the GA to select the optimal subset of rules.

**Expected Output:**
- An optimized set of rules with code and an explanation of the GA process.

---

### 5. Fuzzy Inference for Classification:
- Apply the selected fuzzy rules to classify test samples.
- Aggregate rule contributions and determine the predicted class.

**Expected Output:**
- Fuzzy inference code and prediction results.

---

### 6. Model Evaluation:
- Assess performance using the following metrics:  
  - Accuracy  
  - Recall  
  - Precision  
  - F1 Score  
  - Confusion Matrix  
- Analyze the impact of class imbalance.  
- **Optional:** Apply SMOTE to balance classes.

**Expected Output:**
- An evaluation report with metrics and a confusion matrix.
- A discussion on class imbalance and proposed solutions.

---

### 7. Interpretation and Visualization:
- Plot fuzzy membership functions.
- Visualize rule activation for test samples.
- Interpret key rules and their influence on model decisions.

**Expected Output:**
- Plots and graphs of membership functions.
- Interpretation of rules and sample activations.

---

## Final Deliverables:
- Complete Python code implementing all project tasks.
- Plots and visualizations of membership functions and rules.
- A final project report summarizing methodology, results, and interpretations.
- A list of challenges encountered and suggestions for future improvements.

---

## Suggested Tools and Libraries:
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---

## Optional Tasks:
- Develop a graphical user interface (GUI) for interactive inference.
- Compare the model with alternatives (e.g., decision trees or SVM).

