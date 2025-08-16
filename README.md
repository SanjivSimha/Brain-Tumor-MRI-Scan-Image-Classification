This project uses scikit-learn (sklearn) to build and evaluate a machine learning model that classifies brain MRI scans into four categories:

Glioma tumor

Meningioma tumor

Pituitary tumor

No tumor

The goal is to apply classical machine learning techniques to medical imaging and demonstrate how preprocessing, feature extraction, and model training can support diagnostic workflows.

Features

- Data Preprocessing: Image resizing, grayscale conversion, and normalization for consistent input representation.

- Feature Extraction: Used histogram of oriented gradients (HOG) and pixel intensity statistics.

- Model Training: Trained multiple classifiers (SVM, Random Forest, Logistic Regression) with hyperparameter tuning via GridSearchCV.

- Evaluation: Achieved performance metrics including accuracy, precision, recall, and confusion matrices to analyze class-wise prediction quality.

Results

- Best performing model: SVM with RBF kernel

- Accuracy: ~98.3% 

- Robust classification across tumor types and healthy scans
