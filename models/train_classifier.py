# import libaries
import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import pandas as pd
import pickle
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator,TransformerMixin

def load_data(database_filepath):
    
    """
    Load Data from the Database Function
    
    Arguments:
        database_filepath -> Path to SQLite destination database
    Output:
        X -> a dataframe containing features
        Y -> a dataframe containing labels
        category_names -> List of categories name
    """
    
    # Create engine - connection to database
    engine = create_engine('sqlite:///' + database_filepath)
    
    # Read from the table and save clean data into dataframe
    df = pd.read_sql_table('MyDataset', engine)
    
    # Save message - the model input - in the variable X
    X = df['message']
    
    # Save basically all the category values in the results matrix Y
    Y = df.drop(['message', 'genre', 'id', 'original'], axis=1)

    return X, Y, Y.columns

def tokenize(text):
    
    """
    Tokenize function
    
    Arguments:
        text -> Text message which needs to be tokenized
    Output:
        clean_tokens -> List of tokens extracted from the provided text
    """
    
    # Define regex for detecting URLs
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # Find all urls in the text
    detected_urls = re.findall(url_regex, text)
    
    # Iterate over all detected URLs and replace the original URL with a placeholder
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # Tokenize the sentences into words
    tokens = word_tokenize(text)
    
    # Initialize a Lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Create empty list
    clean_tokens = []
    
    # Iterate over words/tokens, lemmatize, lower and strip whitspaces.
    # Append the refactored words to clean_tokens list
    for tok in tokens:
        try:
            clean_tok = lemmatizer.lemmatize(tok).lower().strip()
            clean_tokens.append(clean_tok)
        except:
            continue

    return clean_tokens


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Starting Verb Extractor class
    
    This class extract the starting verb of a sentence,
    creating a new feature for the ML classifier
    """

    # Define function for determine if first word is a verb
    def starting_verb(self, text):
        
        # Split text into sentences
        sentence_list = nltk.sent_tokenize(text)
        
        # Iterate over detected sentences
        for sentence in sentence_list:
            
            # Split sentences into words and check if the tag of the first word
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    # Given it is a tranformer we can return the self 
    def fit(self, X, y=None):
        return self

    # Override the transform method since we want to create a transformer
    # Apply the pre-defined starting_verb function to X
    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged) 


def build_model():
    
    """
    Build Pipeline function
    
    Output:
        A Scikit ML Pipeline/GridSearchCV that process text messages and apply a classifier.
        
    """
    
    pipeline = Pipeline(
    [
        ('featureUnion', FeatureUnion([
            ('textPipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()), 
            ])),
            ('starting_verb', StartingVerbExtractor())
        ])),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # Create parameters for GridSearch
    parameters = {
        'clf__estimator__n_estimators': [10, 11, 12],
        'clf__estimator__n_jobs': [1, 2, 3]
    }

    # Initialize GridSerachCV object
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    # Find best parameters for pipeline
    cv.fit(X_train, y_train)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    
    """
    Evaluate Model function
    
    This function applies a ML pipeline to a test set and prints out the model performance (accuracy and f1score)
    
    Arguments:
        pipeline -> A valid scikit ML Pipeline
        X_test -> Test features
        Y_test -> Test labels
        category_names -> label names (multi-output)
    """
    
    # Use the test dataset to predict labels
    y_test_predict = model.predict(X_test)
    
    # Save predictions within a pandas dataframe
    y_test_predict = pd.DataFrame(y_test_predict , columns=Y_test.columns.values)
    
    # Iterate over columns within the prediction results and print classification_report for individual labels
    for column in Y_test.columns.values:
        print(classification_report(Y_test[column].values,  y_test_predict[column].values, target_names = [column]))


def save_model(model, model_filepath):
    """
    Save Pipeline function
    
    This function saves trained model as Pickle file, to be loaded later.
    
    Arguments:
        pipeline -> GridSearchCV or Scikit Pipelin object
        pickle_filepath -> destination path to save .pkl file
    
    """
    
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()