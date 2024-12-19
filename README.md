# Industrial Copper Modeling

## Project Overview

The copper industry generates data related to sales and pricing. However, this data often suffers from issues like skewness and noise, making manual predictions inefficient and suboptimal. This project addresses these challenges by building:

1. A **Regression Model** to predict the `Selling_Price` of copper.
2. A **Classification Model** to predict `Status` (WON/LOST) of leads.

Additionally, a **Streamlit web application** is developed to make predictions interactively.

---

## Skills Takeaways

- Python scripting
- Data preprocessing
- Exploratory Data Analysis (EDA)
- Machine Learning (Regression & Classification)
- Streamlit web application development

---

## Domain

**Manufacturing**

---

## Problem Statement

The project focuses on:

- Handling skewed and noisy data.
- Automating pricing predictions with a regression model.
- Classifying sales leads (WON/LOST) using a classification model.
- Developing an interactive application to simplify predictions.

---

## Data Description

- **id**: Unique identifier for transactions.
- **item_date**: Date of transaction.
- **quantity_tons**: Quantity in tons.
- **customer**: Customer identifier.
- **country**: Country of the customer.
- **status**: Transaction status (e.g., WON, LOST).
- **item_type**: Category of the items.
- **application**: Application or use case of the items.
- **thickness**: Thickness of items.
- **width**: Width of items.
- **material_ref**: Reference for materials.
- **product_ref**: Reference for products.
- **delivery_date**: Delivery date of items.
- **selling_price**: Selling price of items.

---

## Approach

### 1. Data Understanding
- Analyze variable types (continuous/categorical).
- Identify rubbish or null values (e.g., `Material_Reference` starting with `00000`).
- Treat reference columns as categorical variables.

### 2. Data Preprocessing
- Handle missing values using mean/median/mode.
- Treat outliers using IQR or Isolation Forest.
- Correct skewness using transformations like log or Box-Cox.
- Encode categorical variables appropriately.

### 3. Exploratory Data Analysis (EDA)
- Visualize skewness and outliers using Seaborn's `boxplot`, `distplot`, or `violinplot`.

### 4. Feature Engineering
- Drop highly correlated columns using a heatmap.
- Engineer new features as necessary.

### 5. Model Building & Evaluation
#### Regression:
- Predict `Selling_Price`.
- Use tree-based models (e.g., Random Forest, XGBoost).

#### Classification:
- Predict `Status` (WON/LOST).
- Evaluate models using metrics like Accuracy, F1 Score, and AUC.

### 6. Streamlit Application
- Input values for prediction.
- Predict `Selling_Price` or `Status`.

---

## Streamlit Workflow

1. Select the task: **Regression** or **Classification**.
2. Enter values for each required column.
3. Click "Predict" to see the result.

---

## Learning Outcomes

- Proficiency in Python libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Streamlit.
- Hands-on experience with data preprocessing, EDA, and feature engineering.
- Building and optimizing ML models.
- Creating an interactive web application.
- Understanding real-world challenges in the manufacturing domain.

---

## How to Run the Project

### 1. Clone the Repository
```bash
git clone <repository_url>
cd <repository_name>
