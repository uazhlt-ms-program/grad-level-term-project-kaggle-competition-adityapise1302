'''
    Name: Aditya Pise
    Class: CSC 539
    Description: My Implementation for the CSC539 Class Competition.
'''
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

class TextClassifier:
    """
    A text classification class that uses NLTK for text preprocessing and
    sklearn's Logistic Regression for text classification through a pipeline
    with TF-IDF vectorization.

    Methods:
    - stem_text: Processes text by stemming.
    - load_data: Loads data from a CSV file and maps labels if necessary.
    - prepare_training_data: Prepares training data by balancing classes.
    - visualize_data_distribution: Plots the distribution of data labels.
    - train_model: Trains the logistic regression model.
    - predict: Makes predictions using the trained model.
    - create_submission: Generates a CSV file for submission.
    """
    def __init__(self):
        """
        Initializes the TextClassifier with a PorterStemmer and a pipeline
        consisting of a TF-IDF Vectorizer and Logistic Regression classifier.
        """
        self.stemmer = PorterStemmer()
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(sublinear_tf=True)),
            ('clf', LogisticRegression(multi_class='multinomial', solver='lbfgs'))
        ])

    def stem_text(self, text):
        """
        Stems the text provided using NLTK's PorterStemmer.

        Parameters:
        - text (str): The text to be stemmed.

        Returns:
        - str: The stemmed version of the input text.
        """
        words = word_tokenize(text)
        stemmed_words = [self.stemmer.stem(word) for word in words]
        return ' '.join(stemmed_words)

    def load_data(self, filepath, labels_map=None):
        """
        Loads data from a specified CSV file and optionally maps labels to a new value.

        Parameters:
        - filepath (str): Path to the CSV file to be loaded.
        - labels_map (dict, optional): A dictionary mapping original labels to new labels.

        Returns:
        - DataFrame: The loaded data with optional new label mappings.
        """
        data = pd.read_csv(filepath)
        # Check if 'TEXT' and 'LABEL' columns exist, rename if necessary
        if 'TEXT' not in data.columns and 'LABEL' not in data.columns and len(data.columns) >= 2:
            data.columns = ['TEXT', 'LABEL'] + list(data.columns[2:])
        if labels_map and 'LABEL' in data.columns:
            data['LABEL'] = data['LABEL'].map(labels_map)
        return data

    def prepare_training_data(self, main_data, additional_data):
        """
        Prepares training data by balancing the classes based on the input datasets.

        Parameters:
        - main_data (DataFrame): The primary dataset.
        - additional_data (DataFrame): An additional dataset used to balance the classes.

        Returns:
        - DataFrame: The prepared and balanced training dataset.
        """
        len_positives = len(main_data[main_data["LABEL"] == 0]) - len(main_data[main_data["LABEL"] == 1])
        len_negatives = len(main_data[main_data["LABEL"] == 0]) - len(main_data[main_data["LABEL"] == 2])
        filtered_positive_df = additional_data[additional_data["LABEL"] == 1].head(len_positives)
        filtered_negative_df = additional_data[additional_data["LABEL"] == 2].head(len_negatives)
        training_df = pd.concat([main_data, filtered_positive_df, filtered_negative_df], ignore_index=True)
        training_df['TEXT'].fillna('nan', inplace=True)
        training_df['STEMMED_TEXT'] = training_df['TEXT'].apply(self.stem_text)
        return training_df

    def visualize_data_distribution(self, data):
        """
        Visualizes the distribution of labels in the dataset using a histogram.

        Parameters:
        - data (DataFrame): The dataset whose label distribution is to be visualized.
        """
        plt.hist(data['LABEL'], bins=[-0.5, 0.5, 1.5, 2.5], edgecolor='black', color='skyblue')
        plt.xticks(ticks=[0, 1, 2], labels=['Not a Review', 'Positive Review', 'Negative Review'])
        plt.title('Distribution of Review Categories')
        plt.xlabel('Review Category')
        plt.ylabel('Frequency')
        plt.show()

    def train_model(self, X, y):
        """
        Trains the model using the specified features and labels.

        Parameters:
        - X (Series): The features for training.
        - y (Series): The labels for training.
        """
        self.pipeline.fit(X, y)

    def predict(self, X):
        """
        Predicts labels for the given input features.

        Parameters:
        - X (Series): Features for which predictions are to be made.

        Returns:
        - ndarray: Predicted labels.
        """
        return self.pipeline.predict(X)

    def create_submission(self, ids, predictions, filename):
        """
        Creates a submission file from the predictions.

        Parameters:
        - ids (Series): The IDs for the test samples.
        - predictions (ndarray): The predicted labels for the test samples.
        - filename (str): The path where the submission file will be saved.
        """
        submission_df = pd.DataFrame({
            'ID': ids,
            'LABEL': predictions
        })
        submission_df.to_csv(filename, index=False)
        print("created " + filename)

classifier = TextClassifier()
print("Creating Training Data Set")
train_data = classifier.load_data('./data/train.csv')
additional_data = classifier.load_data("./IMDB_Dataset.csv", labels_map={'positive': 1, 'negative': 2})
training_df = classifier.prepare_training_data(train_data, additional_data)
# classifier.visualize_data_distribution(training_df)

X, y = training_df['STEMMED_TEXT'], training_df['LABEL']
classifier.train_model(X, y)

print("Predicting Test Set")
test_df = classifier.load_data("./data/test.csv")
test_df['TEXT'].fillna('nan', inplace=True)
test_df['STEMMED_TEXT'] = test_df['TEXT'].apply(classifier.stem_text)
y_pred = classifier.predict(test_df['STEMMED_TEXT'])

classifier.create_submission(test_df['ID'], y_pred, 'submission.csv')
print("Done creating submission.csv")

