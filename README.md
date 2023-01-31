# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage



### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

There should be no necessary libraries to run the code here beyond common Python libaries.  The code should run with no issues using Python version 3.*.

## Project Motivation<a name="motivation"></a>

This project uses a pre-labeled dataset from Appen (previous FigureEight) containing tweets and messages from real-life disaster events. Within the project a NLP and machine learning pipeline is implemented and evaluated in order to categorize the messages. 
The project structure is like this:
- ETL Pipeline for cleaning the dataset
- NLP Pipeline for preparing the dataset for a machine learning pipeline (lower case, remove whitespace, lemmatize, split, ...)
- ML Pipeline in order to predict categorization
- Display results on a WebApp using plotly 

## File Descriptions <a name="files"></a>

- app/templates/*: templates/html files for web app
- data/process_data.py: Extract Train Load (ETL) pipeline used for data cleaning, feature extraction, and storing data in a SQLite database
- models/train_classifier.py: Machine learning pipeline that loads data, trains a model, and saves the trained model as a .pkl file
- run.py: This file can be used to launch the web app and display

## Execution <a name="execution"></a>

To run ETL pipeline to clean data and store the processed data in the database execute:
- python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response_db.db

To run the ML pipeline that loads data from DB, trains classifier and saves the model as a pickle file 
- python models/train_classifier.py data/disaster_response_db.db models/classifier.pkl

Run the following command in the app's directory to run the web app:
- python run.py

In order to see the web app, open a browser and switch to:
- http://0.0.0.0:3001/

## Acknowledgements<a name="acknowledgment"></a>

Must give credit to Appen for the data. 
I also want to thank Udacity for the great introduction!

