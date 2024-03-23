Title:
Predicting Rumor Labels using Machine Learning Models

Description:
This repository contains code for training and evaluating various machine learning models to predict rumor labels based on text data. The models are trained on a dataset (omg.csv) containing text samples and corresponding rumor labels.

Files:
omg.csv: Dataset containing text samples and rumor labels.
rumor_label_prediction.ipynb: Jupyter Notebook containing the Python code for data preprocessing, model training, evaluation, and saving the trained model as a pickle file.
linear_svc_model.pkl: Pickle file containing the trained LinearSVC model.
README.md: This file.
Usage:
Data Exploration and Preprocessing:

The omg.csv file contains the dataset. Load the dataset and explore its structure using pandas.
Preprocess the text data (e.g., removing stopwords, tokenization, vectorization).
Model Training and Evaluation:

Train multiple classification models (LinearSVC, Random Forest, KNN, Decision Tree) on the preprocessed text data.
Evaluate each model's performance using accuracy and classification report.
Model Serialization:

Save the trained LinearSVC model as a pickle file (linear_svc_model.pkl) using the pickle module.
Model Deployment:

Load the serialized model from linear_svc_model.pkl for inference.
Use the loaded model to make predictions on new text data.
Requirements:
Python 3.x
Libraries: numpy, pandas, matplotlib, seaborn, scikit-learn
Instructions:
Clone the repository: git clone <repository_url>
Navigate to the directory: cd <repository_folder>
Install the required libraries: pip install -r requirements.txt
Execute the Jupyter Notebook rumor_label_prediction.ipynb to train models and save the LinearSVC model.
Use the saved model (linear_svc_model.pkl) for prediction by providing input text.

Contributors:
1.Pallaka chakradhar
2.Paidi Ramyasri
3.Kakunuri Naga jyothi

Contact:
For any inquiries or suggestions, feel free to contact chakradharp025@gmail.com or open an issue in the repository.
