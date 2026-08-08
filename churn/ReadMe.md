# Telecom Customer Analysis Churn Prediction AI Model

## Project Overview

This project delivers an end‑to‑end Telecom Customer Churn Prediction solution, leveraging two state‑of‑the‑art AI models:

- Balanced Random Forest: An ensemble of decision trees trained on class‑balanced samples to mitigate the impact of imbalanced churn labels.
- LightGBM (Light Gradient‑Boosting Machine): A highly efficient, leaf‑wise boosting algorithm that iteratively corrects its own errors for faster convergence and superior performance on large datasets.

Both models are trained on rich, synthetic telecom datasets—incorporating usage patterns, billing details, and support interactions—to predict which customers are at highest risk of churn. By identifying high‑risk segments in advance, operators can deploy targeted retention offers and optimize churn‑prevention campaigns, ultimately safeguarding revenue and improving customer lifetime value.

## Model Options

Both the Balanced Random Forest and LightGBM classifiers are well‑suited for binary churn prediction, but they achieve this in different ways that are worth understanding:

- Balanced Random Forest: By explicitly tuning the sampling_strategy, we ensure each tree sees a balanced mix of churn and non‑churn examples. This approach trades a small drop in recall for a substantial gain in precision—meaning when the model flags a customer as “churn,” it’s much more often correct. In practice, that trade‑off can reduce wasted retention offers and focus resources on truly at‑risk subscribers.
- LightGBM: This gradient‑boosting framework grows trees leaf‑wise rather than level‑wise, and it builds each tree to correct the errors of its predecessors via gradient descent on the loss function. The result is typically faster training and strong performance on large, high‑dimensional datasets. LightGBM’s ability to handle complex patterns often translates into a more accurate fit for churn data, as we observed in our experiments.

While both models achieved the expected accuracy on our synthetic dataset, LightGBM proved to be the better fit here—offering quicker convergence and slightly higher overall predictive power. Understanding these architectural differences can guide you in choosing the right model for real‑world churn analytics.

## Data Structure

| Nr | Column                   | Dtype   |
|----|--------------------------|---------|
| 0  | CustomerID               | object  |
| 1  | BillingCycleStart        | object  |
| 2  | BillingCycleEnd          | object  |
| 3  | CallMinutes              | int64   |
| 4  | DataUsageGB              | float64 |
| 5  | SMSCount                 | int64   |
| 6  | InternetSessions         | int64   |
| 7  | TotalBillAmount          | float64 |
| 8  | PaymentStatus            | object  |
| 9  | PaymentDueDate           | object  |
| 10 | LateFee                  | float64 |
| 11 | DiscountsApplied         | float64 |
| 12 | SupportInteractionCount  | int64   |
| 13 | LastInteractionDate      | object  |
| 14 | PrimaryIssueType         | object  |
| 15 | ResolutionTimeMinutes    | int64   |
| 16 | SatisfactionScore        | int64   |
| 17 | EscalationFlag           | bool    |
| 18 | SupportChannel           | object  |
| 19 | HasChurned               | bool    |

## Dataset

The synthetic dataset used to train these models is available on Hugging Face:
https://huggingface.co/datasets/Omer1234567/telco-churn-synthetic

You can also regenerate it yourself with `01-telco-customer-churn-data-generator.py`. Note that the generator is not seeded, so re-running it produces different values, not an exact copy of the published dataset.

## Model Accuracy Results

BalancedRandomForestClassifier Results:<br/>
Confusion Matrix:<br/>
[[24787    15]<br/>
 [ 8363    42]]<br/>

Classification Report:<br/>
|               | precision | recall | f1-score | support |
|---------------|-----------|--------|----------|---------|
| **False**     | 0.75      | 1.00   | 0.86     | 24802   |
| **True**      | 0.74      | 0.00   | 0.01     |  8405   |
| **accuracy**  |           |        | 0.75     | 33207   |
| **macro avg** | 0.74      | 0.50   | 0.43     | 33207   |
| **weighted avg** | 0.74   | 0.75   | 0.64     | 33207   |

Accuracy Score:<br/>
0.7477037973921161<br/>

LightGBM Results:<br/>
Confusion Matrix:<br/>
 [[24024   778]<br/>
 [ 7385  1020]]<br/>

Classification Report:<br/>
|               | precision | recall | f1-score | support |
|---------------|-----------|--------|----------|---------|
| **False**     | 0.76      | 0.97   | 0.85     | 24802   |
| **True**      | 0.57      | 0.12   | 0.20     |  8405   |
| **accuracy**  |           |        | 0.75     | 33207   |
| **macro avg** | 0.67      | 0.54   | 0.53     | 33207   |
| **weighted avg** | 0.71   | 0.75   | 0.69     | 33207   |

Accuracy: 0.7541783358930346<br/>

## Steps to Run

You can choose to run the project locally or build and run it in Docker:

To run locally:
- Open and execute all cells in churn/02-telco-churn-model-training.ipynb.
- Launch the server with:
```
cd churn 
python 03-telco-churn-server.py
```


To run in Docker:
```
cd churn
docker build -t my-churn-app .
docker run --rm -d \
  -p 35000:35000 \
  --name churn-server \
  my-churn-app
```
## Steps to Test

Use the curl command with the -X POST option to invoke each endpoint

Potential Churning Customer Test: <br>

```
curl -X POST -H "Content-Type: application/json" -d '{
  "CustomerID": "CUST0000001",
  "BillingCycleStart": "2020-01-01",
  "BillingCycleEnd": "2020-01-31",
  "CallMinutes": 144,
  "DataUsageGB": 60.19,
  "SMSCount": 39,
  "InternetSessions": 47,
  "TotalBillAmount": 102.19,
  "PaymentStatus": "Paid",
  "PaymentDueDate": "2020-02-05",
  "LateFee": 0.0,
  "DiscountsApplied": 16.53,
  "SupportInteractionCount": 3,
  "LastInteractionDate": "2020-01-11",
  "PrimaryIssueType": "Service Quality",
  "ResolutionTimeMinutes": 117,
  "SatisfactionScore": 2,
  "EscalationFlag": true,
  "SupportChannel": "Email"
}' http://localhost:35000/predict-lgbm
```

Potential Non-Churning Customer Test: <br>
```
curl -X POST -H "Content-Type: application/json" -d '{
  "CustomerID": "CUST0000001",
  "BillingCycleStart": "2020-01-01",
  "BillingCycleEnd": "2020-01-31",
  "CallMinutes": 144,
  "DataUsageGB": 60.19,
  "SMSCount": 39,
  "InternetSessions": 47,
  "TotalBillAmount": 102.19,
  "PaymentStatus": "Paid",
  "PaymentDueDate": "2020-02-05",
  "LateFee": 0.0,
  "DiscountsApplied": 16.53,
  "SupportInteractionCount": 1,
  "LastInteractionDate": "2020-01-11",
  "PrimaryIssueType": "Service Quality",
  "ResolutionTimeMinutes": 117,
  "SatisfactionScore": 2,
  "EscalationFlag": true,
  "SupportChannel": "Email"
}' http://localhost:35000/predict-lgbm
```
