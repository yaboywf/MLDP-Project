# Slide 1. Title, CreditShield
- Hi, I am Dylan. This is CreditShield.
- Goal, predict credit card default risk.
- Output, a risk score and decision support for lenders. 

# Slide 2. Problem Statement

- Banks lose money when high risk customers default.
- Approving too many risky customers increases bad debt.
- Rejecting too many safe customers reduces revenue.
- We want a model that balances risk control and approvals. 

# Slide 3. About the Data

- Dataset, Taiwan credit card clients.
- Target, default next month, 1 means default, 0 means not default.
- Data includes bills, payments, and customer profile fields.
- The classes are imbalanced, so accuracy alone is misleading. 

# Slide 4. Web App Demo

- This Streamlit app lets you input customer data.
- It returns a predicted default risk score.
- You can adjust the decision threshold to match business risk tolerance.
- This turns the model into a practical decision tool. 

# Slide 5. Model Development Strategy
- Start with Logistic Regression as a baseline.
- Add Decision Tree to capture non linear patterns.
- Use Random Forest for stronger generalization and stability.
- Use cross validation and tuning to avoid overfitting. 

# Slide 6. Data Processing and Feature Engineering
- Standardize numeric features, encode categorical features.
- Build summary features over a 6 month window to represent behaviour.
- Examples:
    - PAY_MEAN, repayment consistency.
    - BILL_STD, spending volatility.
    - PAY_BILL_RATIO, repayment burden.
- These are chosen for interpretability and stronger signal. 

# Slide 7. Why This Approach
- Logistic Regression, easy to explain to stakeholders.
- Tree models, handle interactions and non linear risk patterns.
- Random Forest, reduces single tree instability.
- We prioritize recall to catch more defaulters and reduce losses. 

# Slide 8. Model Results
- Logistic Regression catches more defaulters but triggers more false alarms.
- Random Forest improves balance, higher precision with good recall.
- This reduces wrong rejections of good customers while still flagging risk.
- We selected Random Forest as the final model. 

# Slide 9. Value Proposition and Future Enhancement
- Value:
    - Better risk screening.
    - Fewer costly defaults.
    - Better approval quality and profitability.
- Next steps:
    - Tune threshold by cost of false approvals vs false rejections.
    - Add class weighting or resampling for higher recall if needed.
    - Monitor drift and retrain with new data. 

<hr>

Good morning. I am Dylan, and this is CreditGuard, a machine learning model that predicts credit card default risk.

Banks face a trade off. Approve high risk customers and suffer losses. Reject too many and lose revenue. Hence, the goal of this model is to help lenders make better approval decisions using predictive modelling.

We used the Taiwan credit card dataset consisting of 30,000 records and 23 features. The target variable is binary, default or no default. The dataset is imbalanced, where there are significantly more non defaulters than defaulters. Because of this, accuracy alone is not a reliable metric.

Our approach started with Logistic Regression as a baseline model as it is easy to understand and provides a benchmark. Then we tested a Decision Tree to capture non linear patterns. Finally, we used Random Forest with Random Search to improve stability and generalization.

Before modeling, we performed feature engineering. Instead of using six separate monthly bill and payment values, we created summary features over a six month period. For example, BILL_TOTAL captures total debt, while BILL_STD captures spending volatility. These features reflect financial logic and improve signal quality.

For evaluation, we used ROC AUC, recall, and precision. Since missing defaulters leads to financial loss, recall for the default class is critical. Among the models, Random Forest achieved the highest ROC AUC and best balance between recall and precision, so we selected it as the final model.

To end off, CreditGuard reduces bad debt risk by identifying high risk customers before approval and improves decision consistency through data driven risk scoring.