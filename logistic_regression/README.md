# Logistic Regression

## 1. Intuition about Logistic Regression
Logistic Regression is a statistical method used for binary classification problems, where the outcome is a categorical variable with two possible values (e.g., yes/no, true/false, 0/1). Unlike Linear Regression, which predicts continuous values, Logistic Regression predicts the probability that a given input belongs to a particular class. The model uses a logistic function (also known as the sigmoid function) to squeeze the output of a linear equation between 0 and 1.

## 2. Application: Case study where we can apply Logistic Regression
**Case Study:** Predicting Customer Churn

In the telecommunications industry, predicting customer churn is critical for retaining customers. We can use Logistic Regression to model the likelihood of a customer leaving the service based on various features such as monthly charges, contract type, tenure, and support calls. By fitting a logistic model to historical data, we can predict the probability of churn for each customer.

## 3. Elaboration: Math and Statistics behind Logistic Regression
Logistic Regression models the probability \( $P$ \) that the dependent variable \( $y$ \) belongs to a particular class. The model can be expressed as:

$$ P(y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n)}} $$

where:
- \( $y$ \) is the dependent variable (0 or 1).
- \( $x_1$, $x_2$, $\ldots$, $x_n$ \) are the independent variables.
- \( $\beta_0$ \) is the intercept.
- \( $\beta_1$, $\beta_2$, $\ldots$, $\beta_n$ \) are the coefficients.

The goal is to estimate the coefficients \( $\beta$ \) that maximize the likelihood of the observed data. This is achieved using maximum likelihood estimation.

The decision boundary for classification is typically set at a probability threshold of 0.5. If \( $P(y=1|X)$ \) is greater than 0.5, the model predicts class 1; otherwise, it predicts class 0.

## 4. Python Code to perform Logistic Regression for the problem presented in the case study
The Python code for performing Logistic Regression on a customer churn dataset is provided in a separate file: [logistic_regression.py](./logistic_regression.py).

## 5. Insights summary
Logistic Regression is a powerful tool for binary classification problems. In the context of predicting customer churn, it helps identify the factors that influence whether a customer will leave or stay. By understanding these factors, businesses can take proactive measures to retain customers. The simplicity and effectiveness of Logistic Regression make it a popular choice for many classification tasks.
