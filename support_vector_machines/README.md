# Support Vector Machines (SVM)

## 1. Intuition about Support Vector Machines
Support Vector Machines (SVM) are a powerful and versatile supervised learning model used for classification and regression tasks. The core idea of SVM is to find the optimal hyperplane that best separates the data points of different classes. The hyperplane is chosen to maximize the margin, which is the distance between the hyperplane and the closest data points from each class, known as support vectors. SVM can handle linear and non-linear classification using kernel tricks to transform the input space.

## 2. Application: Case study where we can apply Support Vector Machines
**Case Study:** Classifying Emails as Spam or Not Spam

Email spam detection is a common application of classification algorithms. We can use Support Vector Machines to model the relationship between email features (such as word frequencies, presence of specific keywords, and email length) and the likelihood of the email being spam. By fitting an SVM model to historical labeled email data, we can classify new emails as spam or not spam.

## 3. Elaboration: Math and Statistics behind Support Vector Machines
SVM aims to find the hyperplane that best separates the classes. For a linearly separable dataset, the decision boundary can be expressed as:

${w}$ $\cdot$ ${x}$ $- b = 0$

where:
- \( ${w}$ \) is the weight vector perpendicular to the hyperplane.
- \( ${x}$ \) is the input feature vector.
- \( $b$ \) is the bias term.

The optimization problem for SVM is to maximize the margin \( $\frac{2}{\|\mathbf{w}\|}$ \) subject to the constraint that all data points are correctly classified. This can be formulated as:

$ \min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 $

subject to:

$ y_i (\mathbf{w} \cdot \mathbf{x}_i - b) \geq 1 $

For non-linearly separable data, SVM uses kernel functions to map the input space to a higher-dimensional space where a linear separation is possible. Common kernels include:
- Linear kernel
- Polynomial kernel
- Radial Basis Function (RBF) kernel

## 4. Python Code to perform Support Vector Machines for the problem presented in the case study
The Python code for performing Support Vector Machines on an email spam detection dataset is provided in a separate file: [svm_spam.py](./svm_spam.py).

## 5. Insights summary
Support Vector Machines are effective for high-dimensional classification problems. In the context of classifying emails as spam or not spam, SVM can handle various features and capture complex relationships between them. The evaluation metrics from the model, such as accuracy, precision, recall, and F1-score, provide a comprehensive view of the model's performance.

From the generated SVM model, we observe that features like specific keywords and email length significantly influence the classification. The ability of SVM to create non-linear decision boundaries using kernel tricks makes it a robust choice for spam detection and other classification tasks with complex patterns.
