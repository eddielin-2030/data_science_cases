# Light Gradient Boosting Machine (LightGBM)

## 1. Intuition about Light Gradient Boosting Machine
LightGBM is a gradient boosting framework that uses tree-based learning algorithms. It is designed to be efficient and scalable, capable of handling large datasets with higher accuracy and faster training speeds compared to other boosting algorithms. LightGBM uses histogram-based algorithms to bucket continuous feature values into discrete bins, significantly speeding up the training process.

## 2. Application: Case study where we can apply LightGBM
**Case Study:** Predicting Credit Card Fraud

In the financial sector, detecting credit card fraud is crucial for preventing financial losses and maintaining customer trust. We can use LightGBM to model the relationship between transaction characteristics (such as amount, time, and merchant category) and the likelihood of a transaction being fraudulent. By fitting a LightGBM model to historical transaction data, we can predict the probability of fraud for new transactions.

## 3. Elaboration: Math and Statistics behind LightGBM
LightGBM builds decision trees sequentially, with each new tree correcting the errors of the previous ones. The key features of LightGBM include:
1. Gradient-based One-Side Sampling (GOSS): LightGBM retains instances with large gradients and performs random sampling on instances with small gradients to reduce computation.
2. Exclusive Feature Bundling (EFB): LightGBM bundles mutually exclusive features to reduce the number of features and speed up training.
3. Histogram-based Decision Tree Learning: LightGBM uses histograms to bucket continuous feature values into discrete bins, reducing the complexity of finding the best split.

The objective function for LightGBM is optimized using gradient descent, and the model combines the predictions of all trees to minimize the loss function.

## 4. Python Code to perform LightGBM for the problem presented in the case study
The Python code for performing LightGBM on a credit card fraud detection dataset is provided in a separate file: [lightgbm.py](./litegbm.py).

## 5. Insights summary
LightGBM is a powerful and efficient algorithm that can handle large-scale datasets and complex relationships. In the context of predicting credit card fraud, the LightGBM model leverages multiple decision trees to capture the factors contributing to fraudulent transactions.

These metrics indicate that the model can predict credit card fraud but it can use some improvement. High precision and recall are essential in accurately identifying fraudulent transactions and minimizing false positives.

In our case study, the LightGBM model provides performance metrics, demonstrating its ability to generalize well to new data. The evaluation metrics provide insights into the model's effectiveness, such as its ability to correctly classify fraudulent transactions, which is crucial for preventing financial losses and maintaining customer trust in the financial sector.
