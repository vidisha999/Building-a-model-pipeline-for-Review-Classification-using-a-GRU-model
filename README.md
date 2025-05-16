# Building-a-model-pipeline-for-Review-Classification-using-a-GRU-model

## Project Description
The purpose of this project is to build a review classification model, that does sentiment analysis of the review text data from an application which is scaled in the rage of 1 to 5, to identify 5 as a positive sentiment and 1 as a negative sentiment. The deep learning model was built using GRU (Gated Recurrent Unit) which is a variant of RNN ( Recurrent Neural Network). The model is trained on the training review data and their properties to predict the sentiment. This project outlines an automated model pipeline from building model to its deployment enabling the use of deployed model in a scalable way to make real-time decisions based on the sentiment predictions of the app reviews.

## GRU ---importance of GRU as a RNN 
## Objective 
## Modular code overview 
## Data 
The dataset contains `content` and `score` columns that respectively represent the review text and the rating given by the user based on their review. The score contains values between 0-5, reflecting the sentiment expressed in the review.

## Model Buiding Pipeline 
The model building pipeline contains several steps
1. Preprocessing the dataset:
 - Clean the dataset by removing stopwords and lemmatizing text to ensure an appropriate format.
 - Tokenize the cleaned data for further processing.

2. Model training
 - Split the preprocessed dataset to training and testing sets.
 - Build a GRU (Gated Recurrent Unit) model and train it using the training dataset.

3. Model deployment
- Develop a REST API using the Flask web framework to deploy the trained model.
- Implement a model engine script using subprocesses to automate the execution of multiple processes.
    

### Preprocessing the dataset 
#### 1. Cleaning the dataset 

The text data is often in incosistent formats which must be structured and standerdized to an appropriate format for the machine learning model, enabling machine learning models to deliver more accurate predictions and reliable insights.

Stopwords are common words that are found in the text data which don't add a significant meaning to text analysis. Removing stopwords helps improve model efficiency and reduces noise in the dataset. These words vary by language, and predefined lists exist for different NLP applications. Stop words ca be accessed using the text corpus of the **NLTK** library which is a common python library used in NLP ( Natural Language  Processing)tasks.

Text data often contains different variations of the same words, introducing variability that needs to be standardized during preprocessing.Lemmatization is a data preprocessing technniqe that transforms words into their dictionary form (lemma), based on the context of the words while preserving their contextual meaning and linguistic integrity.

The **cleaning(df,stopwords)** function within [preprocessing script](SRC/ML_pipeline_vidisha/preprocess_vidisha.py) demonstrates a sequence of steps used in data cleaning process. 
```python

import nltk
import textblob
from textblob import Word # use for lemmatization 
from nltk.corpus import stopwords
stop_words=stopwords.word('english') # should remove these words

def cleaning(df,stop_words):
    df['content'] = df['content'].apply(lambda x: ' '.join(x.lower() for x in x.split())) # convert words in the review text to lowercase 
    df['content'] = df['content'].str.replace("[^0-9a-zA-Z\s]+", '') # Use a regex to replace special characters with spaces
    df['content'] = df['content'].apply(lambda x: ' ' .join ( x for x in x.split() if x not in stop_words)) # remove stopwords 
    df['content'] = df['content'].apply(lambda x: ' '.join([Word(x).lemmatize() for x in x.split()])) # lemmatize each word in review text
    return df
```

- The **wordcloud** library is used to  generate a word cloud that visually represents the most frequently occuring words in the prepocessed text data. The below image gives a quick insight about the dominant trends or themes in the app review text data, which are `app`, `design`,`good` and `easy`.

  ![wordcloud](Images/image-gru.png)

#### 2. Tokenizing the data
Cleaned text data must be transformed into a structured format that machine learning algorithms can process and learn from the data for text analysis.Tokenization is the process of breaking down text into smaller units, known as tokens, which can be words, phrases, sentence or subwords.By enabling ML models to nderstand individual words and their contextual relationships in the text data it helps the model to effective analyze the text setiments.Effective tokenization enhances model performance by improving the accuracy and relevance of text-based predictions.

In the model API, users can choose whether to train a new dataset, make real-time predictions on review data, or deploy the model. To automate the tokenization process based on the user's selection, the **tokenize(df,df_new,is_train)** function is defined in the [preprocessing script](SRC/ML_pipeline_vidisha/preprocess_vidisha.py). If user selects training option, the function builds a tokenizer using the training data and save the [trained tokenizer](Output/tokenizer.pkl) as a pickle file for future use.












## The model engine 
In order to build a robust and streamlined model pipeline, the python submodule **subprocess** was used to ochestrate multiple tasks in parallel using pre-defined functions. It leverages the subprocess module to trigger and run external scripts in seperate processes enabling modular execution of training, prediction or depoyment of the model in both development or production environment. The [model engine](SRC/engine_vidisha.py) demonstrates the ochestration of seperate scripts for data preprocessing, model training and deployment.
