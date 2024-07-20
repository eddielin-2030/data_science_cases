# Gradient Boosting Machines (GBM)

## 1. Intuition about Gradient Boosting Machines
Gradient Boosting Machines (GBM) is an ensemble learning method that builds multiple decision trees sequentially. Each tree attempts to correct the errors of the previous one. GBM combines the predictions of several base estimators to improve robustness over a single estimator. It uses a gradient descent approach to minimize the loss function and optimize the model performance.

## 2. Application: Case study where we can apply Gradient Boosting Machines
**Case Study:** Predicting Customer Churn

In the telecommunications industry, predicting customer churn is crucial for retaining customers and minimizing losses. We can use Gradient Boosting Machines to model the relationship between customer characteristics (such as contract type, monthly charges, and tenure) and the likelihood of churn. By fitting a GBM model to historical data, we can predict the probability of churn for new customers.

## 3. Elaboration: Math and Statistics behind Gradient Boosting Machines
Gradient Boosting involves three main components:
1. A loss function to be optimized (e.g., mean squared error for regression, log-loss for classification).
2. A weak learner (typically a decision tree) to make predictions.
3. An additive model to minimize the loss function by adding weak learners sequentially.

The model starts with an initial prediction, and then iteratively adds new trees that fit the residual errors from the previous trees. The update rule for each iteration $m$  is:

$$
F_m(x) = F_{m-1}(x) + h_m(x)
$$

where $F_m(x)$ is the prediction from the $m$-th iteration, and $h_m(x)$ is the new tree fitted to the residuals.

The weights for the updates are determined using gradient descent to minimize the chosen loss function.

## 4. Python Code to perform Gradient Boosting Machines for the problem presented in the case study
The Python code for performing Gradient Boosting Machines on a customer churn prediction dataset is provided in a separate file: [gbm.py](./gbm.py).

## 5. Insights summary
Gradient Boosting Machines are powerful and flexible algorithms that can handle various types of data and model complex relationships. In the context of predicting customer churn, the GBM model leverages multiple decision trees to capture the factors contributing to churn.

### Metrics Results
- **Accuracy:** 0.50
- **Precision:** 0.435
- **Recall:** 0.455
- **F1-score:** 0.444

These metrics indicate that the model has room for improvement in predicting customer churn. The precision, recall, and F1-scores highlight the trade-offs between false positives and false negatives. The recall for class 1 (churn) indicates that the model struggles to identify all churning customers accurately.

In our case study, the GBM model provides moderate performance metrics, demonstrating its potential to generalize to new data. The evaluation metrics provide insights into the model's effectiveness, such as its ability to correctly classify customers who are likely to churn, which is crucial for making informed retention strategies in the telecommunications industry.
