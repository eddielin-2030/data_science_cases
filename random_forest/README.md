# Random Forest

## 1. Intuition about Random Forest
Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the mode of the classes (for classification) or mean prediction (for regression) of the individual trees. It leverages the power of multiple models to improve the overall performance and reduce the risk of overfitting. Random Forest introduces randomness into the model building process by selecting random subsets of features and samples, which helps in creating diverse trees.

## 2. Application: Case study where we can apply Random Forest
**Case Study:** Predicting Loan Default

In the banking sector, predicting whether a customer will default on a loan is a critical task. We can use Random Forest to model the relationship between customer characteristics (such as credit score, income, loan amount, and employment history) and the likelihood of defaulting on a loan. By fitting a Random Forest model to historical data, we can predict the probability of default for new loan applications.

## 3. Elaboration: Math and Statistics behind Random Forest
Random Forest builds multiple decision trees using different subsets of the training data and features. The algorithm involves two main steps:
1. Bootstrap Aggregation (Bagging): Random samples are drawn with replacement from the training data to build each tree.
2. Random Feature Selection: A random subset of features is chosen at each split in the tree to introduce additional randomness.

The final prediction is made by aggregating the predictions of all the trees:
- For classification: the mode of the class predictions.
- For regression: the mean of the predictions.

The formula for the prediction in regression is:
\[ \hat{y} = \frac{1}{T} \sum_{t=1}^{T} \hat{y}_t \]

where \( \hat{y}_t \) is the prediction from the \( t \)-th tree and \( T \) is the total number of trees.

## 4. Python Code to perform Random Forest for the problem presented in the case study
The Python code for performing Random Forest on a loan default prediction dataset is provided in a separate file: [random_forest.py](./random_forest.py).

## 5. Insights summary
Random Forest is a powerful and flexible algorithm that can handle both classification and regression tasks. In the context of predicting loan defaults, the Random Forest model leverages multiple decision trees to capture complex relationships between customer characteristics and the likelihood of default.

### Metrics Results
- **Accuracy:** 0.565
- **Precision:** 0.552 for class 0, 0.583 for class 1
- **Recall:** 0.646 for class 0, 0.485 for class 1
- **F1-score:** 0.595 for class 0, 0.530 for class 1

These metrics indicate that the model has room for improvement in predicting loan defaults. The precision, recall, and F1-scores highlight the trade-offs between false positives and false negatives. High recall for class 0 indicates that most of the non-defaults are correctly identified, while lower recall for class 1 suggests that the model struggles to identify all defaults accurately.

In our case study, the Random Forest model provides reasonable performance metrics, demonstrating its potential to generalize to new data. The evaluation metrics provide insights into the model's effectiveness, such as its ability to correctly classify loan defaults, which is crucial for making informed lending decisions and managing risk in the banking industry.
