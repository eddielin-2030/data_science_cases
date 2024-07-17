# Naive Bayes Classifier

## 1. Intuition about Naive Bayes Classifier
Naive Bayes is a probabilistic classifier based on Bayes' Theorem. It is called "naive" because it assumes that the features are conditionally independent given the class label, which is a simplification that often does not hold true in real-world scenarios. Despite this naive assumption, Naive Bayes classifiers perform surprisingly well in various applications, particularly in text classification and spam detection. The classifier calculates the posterior probability for each class and predicts the class with the highest probability.

## 2. Application: Case study where we can apply Naive Bayes Classifier
**Case Study:** Classifying Movie Reviews as Positive or Negative

Sentiment analysis of movie reviews is a common application of text classification. We can use the Naive Bayes Classifier to model the relationship between words in a review and the sentiment of the review (positive or negative). By fitting a Naive Bayes model to historical labeled review data, we can classify new reviews based on their word content.

## 3. Elaboration: Math and Statistics behind Naive Bayes Classifier
Naive Bayes classifiers apply Bayes' Theorem with strong (naive) independence assumptions. Bayes' Theorem is expressed as:

\[ P(C|X) = \frac{P(X|C) \cdot P(C)}{P(X)} \]

where:
- \( P(C|X) \) is the posterior probability of class \( C \) given the feature vector \( X \).
- \( P(X|C) \) is the likelihood of feature vector \( X \) given class \( C \).
- \( P(C) \) is the prior probability of class \( C \).
- \( P(X) \) is the prior probability of feature vector \( X \).

For classification, we are interested in finding the class \( C \) that maximizes the posterior probability \( P(C|X) \). Using the independence assumption, the likelihood \( P(X|C) \) can be decomposed into the product of individual feature likelihoods:

\[ P(X|C) = P(x_1|C) \cdot P(x_2|C) \cdot \ldots \cdot P(x_n|C) \]

The Naive Bayes classifier predicts the class \( C \) that maximizes \( P(C|X) \).

## 4. Python Code to perform Naive Bayes Classifier for the problem presented in the case study
The Python code for performing Naive Bayes Classifier on a movie review sentiment analysis dataset is provided in a separate file: [naive_bayes.py](./naive_bayes.py).

## 5. Insights summary
Naive Bayes classifiers are simple yet effective for various classification problems, particularly those involving text data. In the context of classifying movie reviews as positive or negative, the Naive Bayes model leverages word frequencies to determine the sentiment. 

### Metrics Results
- **Accuracy:** Indicates the overall correctness of the model. 
- **Precision:** The ratio of true positive predictions to the total predicted positives. It reflects how many selected items are relevant.
- **Recall:** The ratio of true positive predictions to the actual positives. It reflects how many relevant items are selected.

From the generated Naive Bayes model, we observe that certain words strongly influence the classification of reviews as positive or negative. The model's ability to handle high-dimensional data and its efficiency make it a popular choice for text classification tasks like sentiment analysis and spam detection.

In our case study of classifying movie reviews, the model’s evaluation metrics provide insights into its performance. High precision indicates that when the model predicts a review as positive, it is often correct. High recall indicates that the model successfully identifies most of the positive reviews. These metrics are crucial for understanding the trade-offs between false positives and false negatives in the context of our business question: ensuring that positive reviews are correctly identified to maintain user satisfaction and improve recommendation systems.
