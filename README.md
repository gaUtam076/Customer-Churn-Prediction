**Customer Churn Prediction & Analytics Dashboard**

Project Overview

This project predicts which telecom customers are likely to churn (cancel their subscription) using machine learning.

Features included:
- Complete data cleaning pipeline
- Exploratory Data Analysis (EDA)
- Multiple ML models trained and compared
- Interactive Streamlit web app
- Power BI dashboard for business insights
- Best model saved using Joblib

--------------------------------------------------

Dataset
--------------------------------------------------
Downloaded from Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

IBM Telco Customer Churn Dataset

- Rows: 7,043 customers
- Features: 21 columns
- Target Variable: Churn (Yes / No)

Key Features:
- Demographics
- Internet and phone services
- Contract details
- Payment methods
- Monthly charges
- Customer tenure

--------------------------------------------------

Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Power BI
- Plotly

--------------------------------------------------
Machine Learning Models
------------------------------------------------
Models Used:
- Logistic Regression
- Decision Tree
- Random Forest

Best Model Performance (Random Forest):
- Accuracy: ~81%
- ROC-AUC: ~87%

--------------------------------------------------

Key Insights

- Month-to-month customers showed higher churn rates
- Customers with higher monthly charges were more likely to churn
- Low-tenure customers had higher churn risk
- Electronic check users showed increased churn

--------------------------------------------------

Features
Power BI Dashboard:
- Churn trends
- Revenue insights
- Customer segmentation
- KPI analysis

--------------------------------------------------
Page -1 Executive Overview

The Executive Overview page provides a high-level snapshot of customer churn performance and business health.
🔹 Key Insights
Total Customers: 7043
Retained Customers: 6566
Churned Customers: 477
Avg Monthly Charge: $68.02
Monthly Revenue at Risk: $33.4K
🔹 Business Understanding
Most customers are retained, showing stable customer loyalty
Churn still causes significant monthly revenue loss
Contract type directly impacts churn behavior
Customers with lower tenure are more likely to churn
🔹 Main Visuals

✅ KPI Cards
✅ Donut Chart — Churn Distribution
✅ Bar Chart — Churn by Contract Type
✅ Trend Line — Churn by Tenure

---------------------------------------------------------------
Page 2 — Customer Segmentation
The Customer Segmentation page helps identify which customer groups are more likely to churn.

🔹 Key Insights
Fiber optic users form the largest customer group
Payment methods influence churn behavior
Short-tenure customers show higher churn rates
High-charge customers can become churn risks
🔹 Business Understanding

This page helps businesses:

Target high-risk customer segments
Improve customer retention strategies
Analyze customer behavior patterns
Personalize marketing campaigns
🔹 Main Visuals

✅ Interactive Slicers
✅ Internet Service vs Churn Analysis
✅ Churn Heatmap
✅ Payment Method Analysis
✅ Customer Risk Scatter Plot

---------------------------------------------------------------------
Page 3 — Revenue Impact
The Revenue Impact page measures the financial effect of customer churn.

🔹 Key Insights
Month-to-month contracts contribute the highest revenue risk
Early-tenure customers generate larger revenue loss
Payment methods affect revenue stability
Revenue at risk exceeds $33K monthly
🔹 Business Understanding

This page helps organizations:

Understand financial impact of churn
Prioritize retention investments
Forecast revenue loss
Improve contract retention planning
🔹 Main Visuals

✅ Waterfall Chart — Revenue by Contract
✅ Area Chart — Revenue by Tenure
✅ Treemap — Revenue by Payment Method
✅ KPI Summary Table
