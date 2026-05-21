# Customer Campaign Engagement and Response Analytics Report: CE-3

This repository contains a comprehensive data analysis, pre-processing, and feature selection pipeline for customer marketing campaign responses based on the **CE-3** Jupyter Notebook and the **marketing_campaign.csv** dataset.

---

## Executive Summary
Every marketing contact has a cost, and every positive response yields revenue. The goal of this analysis is to **maximize profit** by targeting only receptive customers, minimizing wasted budget, and avoiding customer unsubscribe fatigue.

Out of over 30 candidate variables, the multi-method selection voting framework (using Forward Selection, Backward Elimination, Lasso, Information Gain, and Fisher Score) isolated a highly stable four-feature subset: **AcceptedCmp5**, **AcceptedCmp3**, **Recency**, and **Marital_Status_Married**.

---

## 1. Introduction
In modern business, blasting every customer with the same marketing campaign is expensive and ineffective. Instead, companies use **Targeted Marketing**—using data to identify and reach only the customers who are most likely to respond.

This analysis covers the step-by-step pipeline in the **CE-3** Jupyter Notebook, which analyzes a database of **2,240 customers** to predict who will accept a new promotional campaign (the **Response** column). The analysis moves from raw data cleaning to finding the top 4 factors that drive customer engagement.

---

## 2. Problem Statement

### The Business Goal
Every marketing contact has a cost, and every positive response yields revenue. The goal is to **maximize profit** by targeting only receptive customers, minimizing wasted budget, and avoiding customer unsubscribe fatigue.

### The Technical Challenge
The raw customer database is noisy and complex. To build an accurate predictive model, we must solve several problems:
1. **Missing Data:** Household Income has 24 missing values.
2. **Noisy Text Data:** Relationship status contains inconsistent values like "YOLO", "Alone", and "Absurd".
3. **Outliers & Extreme Spikes:** A few wealthy customers spent huge amounts, which can skew our metrics.
4. **Too Many Columns:** With over 30 candidate features, we need to isolate the few that truly matter.

---

## 3. Methodology (Step-by-Step)

The notebook prepares the data for machine learning using a clean, 6-step pipeline:

* **Step 1: Initial Check**
  * *What was done:* Scanned data dimensions (2,240 rows by 29 columns) and checked for duplicates (0 found).
  * *Why it matters:* Establishes the baseline quality of the dataset.

* **Step 2: Missing Income Imputation**
  * *What was done:* Used an **Iterative Imputer** to predict and fill the 24 missing income values based on other purchasing habits.
  * *Why it matters:* Avoids deleting valuable customer records by filling gaps using smart algorithms instead of simple averages.

* **Step 3: Text Consolidation**
  * *What was done:* Cleaned the Marital Status column by mapping rare strings: "Alone" became "Single", and "YOLO" / "Absurd" became "Other".
  * *Why it matters:* Groups similar customer profiles together to prevent noise.

* **Step 4: Outlier Capping (IQR)**
  * *What was done:* Identified extreme values using the **Interquartile Range (IQR) method** and capped them (e.g., restricted extremely high spending or income spikes).
  * *Why it matters:* Neutralizes massive spikes without deleting customer rows.

* **Step 5: Skewness Reduction**
  * *What was done:* Applied a log transformation (`np.log1p`) to highly skewed spending and income columns.
  * *Why it matters:* Evens out the distributions so models can learn patterns more easily.

* **Step 6: Feature Engineering**
  * *What was done:* Derived 4 custom columns:
    * **Age** = 2026 - Year of Birth
    * **Total Spending** = Sum of Wines + Fruits + Meat + Fish + Sweets + Gold purchases
    * **Family Size** = Kids at home + Teens at home + 1
    * **Customer Tenure** = 2026 - Year of Customer Registration
  * *Why it matters:* Combines raw data into high-value business metrics.

---

## 4. Dataset Description

The dataset tracks **2,240 customers** across five primary categories of information:

* **Demographics:** Customer Age, Education, Marital Status, Household Income, and registration date.
* **Family Structure:** Number of kids and teenagers living in the household.
* **Product Purchases (Last 2 Years):** Amounts spent on Wines, Fruits, Meat, Fish, Sweets, and Gold.
* **Purchase Channels:** Number of purchases made via web, catalog, retail stores, or using discount deals.
* **Campaign History:** Binary indicators of whether the customer accepted offers in previous campaigns (Campaign 1 to Campaign 5).
* **Target Variable (Response):** 1 if the customer accepted the new campaign offer, 0 if they did not (Conversion Rate = **14.91%**).

---

## 5. Data Acquisition Methods

The data analyzed in this notebook is compiled from typical core business systems:
1. **CRM Databases:** Captures customer profiles, birth years, and marital statuses during loyalty card registrations.
2. **Point-of-Sale (POS) & E-Commerce Servers:** Records physical store transactions and web purchases.
3. **Web Traffic Analytics:** Monitors monthly visits to the website.
4. **Marketing Automation Logs:** Tracks historical campaign email clicks and offer acceptances.

*Note: All personally identifiable information (PII) like names or phone numbers was stripped and replaced with a synthetic anonymous ID to comply with data privacy laws.*

---

## 6. Key Analysis & Interpretations

### 6.1 Product & Spend Insights
* The average total spending per customer over 2 years is **$605.80**.
* **Wine** is the absolute dominant category, averaging **$303.94** (accounting for **50.17%** of total spend).
* **Meat** is the second largest, averaging **$166.95** (accounting for **27.56%** of total spend).
* Together, **Wine and Meat account for 77.73% of all sales**, indicating these are the primary drivers of customer value.

### 6.2 Skewness & Outliers
Before pre-processing, monetary columns had extreme right-skewness and high outlier counts (e.g., 248 outliers in sweets spending and a massive skewness score of **6.75** in Income). Capping these spikes at upper IQR limits and applying the log transformation successfully normalized the columns.

### 6.3 Feature Selection: Comparing the 5 Methods
To identify which features best predict if a customer will respond to a campaign, the notebook evaluated the dataset using five separate data science techniques:

1. **Forward Selection (Wrapper Method):** Gradually adds columns that improve prediction the most.
   * *Features Selected:* Teenhome, Recency, AcceptedCmp3, AcceptedCmp5, AcceptedCmp1
2. **Backward Elimination (Wrapper Method):** Gradually removes columns to keep a strong baseline.
   * *Features Selected:* Recency, AcceptedCmp3, AcceptedCmp5, Marital_Status_Married, Marital_Status_Together
3. **Lasso Regularization (Embedded Method):** Softly filters out highly redundant variables.
   * *Features Selected:* 19 features (demographics, spends, channels)
4. **Information Gain (Filter Method):** Measures which columns reduce prediction uncertainty.
   * *Features Selected (Top 5):* MntMeatProducts, Income, AcceptedCmp3, Total_Spending, AcceptedCmp5
5. **Fisher Score (Filter Method):** Evaluates which columns separate positive/negative classes best.
   * *Features Selected (Top 5):* Total_Spending, Marital_Status_Married, AcceptedCmp4, NumStorePurchases, AcceptedCmp5

---

### 6.4 The Final Top 4 Features Explained
By combining all five methods, the notebook isolated the four features chosen by **at least 3 out of 5 techniques**. These are the most stable, non-redundant, and powerful predictors in the entire dataset:

#### 1. AcceptedCmp5 (Prior Campaign 5 Response)
* **Responders:** 27.54% accepted | **Non-Responders:** 3.73% accepted
* **Interpretation:** Customers who accepted Campaign 5 are **7.3 times more likely** to buy again. This indicates strong high-end brand loyalty.

#### 2. AcceptedCmp3 (Prior Campaign 3 Response)
* **Responders:** 23.05% accepted | **Non-Responders:** 4.51% accepted
* **Interpretation:** Customers who accepted Campaign 3 are **5.1 times more likely** to convert. Campaign 3 targeted deal-seekers, marking value sensitivity.

#### 3. Recency (Days Since Last Purchase)
* **Responders Avg:** 35.38 days | **Non-Responders Avg:** 51.51 days
* **Interpretation:** Responders purchased **16 days more recently** than non-responders. Active customers are in a shopping state of mind.

#### 4. Marital_Status_Married (Is Married)
* **Responders:** 29.34% married | **Non-Responders:** 40.19% married
* **Interpretation:** Responders are **27% less likely to be married**. Married households have shared budgets and higher decision friction, whereas single individuals exhibit more purchase autonomy.

*Note: An audit of the Pearson correlation heatmap shows virtually zero correlation between these four selected columns (all under 0.10). This means each of the four features represents a completely unique signal, preventing model confusion (multicollinearity).*

---

## 7. Conclusion & Strategic Recommendations

### Core Findings
Predicting customer conversion does not require analyzing dozens of complex variables. By focusing strictly on **Prior Campaign Engagement (3 & 5)**, **Transactional Recency**, and **Married Status**, we can accurately forecast promotional success.

### Strategic Recommendations
1. **Target the Loyalty Loop:** Allocate a guaranteed portion of the budget to prior Campaign 3 & 5 responders. They convert at 5x to 7x the average rate.
2. **Target Active Windows:** Focus outbound offers on customers who purchased in the last 30 days. Place customers with more than 60 inactive days into a separate win-back flow.
3. **Customize by Household:** Unmarried customers convert easily; send them personalized, individual offers. For married households, use family-centric values or multi-buy discounts to reduce purchase friction.

---

## 8. References

1. **Scikit-learn Documentation:** *Imputation of missing values* and *Sequential Feature Selection* (Pedregosa et al., 2011).
2. **Database Marketing Principles:** *Strategic Database Marketing* (Hughes, 2006). Explains the foundations of RFM (Recency, Frequency, Monetary) modeling.
3. **Lasso Theory:** *Regression shrinkage and selection via the lasso* (Tibshirani, 1996).
4. **Information Theory:** *Elements of Information Theory* (Cover & Thomas, 2006). Explains the math behind Mutual Information and Information Gain.
