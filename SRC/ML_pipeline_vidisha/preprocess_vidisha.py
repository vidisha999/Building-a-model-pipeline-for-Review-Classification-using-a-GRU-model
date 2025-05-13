
# import necessary libraries 
import tensorflow as tf
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import pickle
from textblob import Word
from nltk.corpus import stopwords
from nltk.corpus import wordnet
nltk.download('omw-1.4')
import re
from ML_pipeline_vidisha import utils_vidisha



## Preprocess Text data
def cleaning(df,stop_words):
    df_tmp = df.copy()
   
    df_tmp['content'] = df_tmp['content'].apply(lambda x: ' '.join(x.lower() for x in x.split())) # converting words to lowercase 
    df_tmp['content'] = df_tmp['content'].str.replace("[^0-9a-zA-Z\s]+", '') # replacing special characters with spaces
    #df_tmp['content'] = df_tmp['content'].str.replace(r"[d]",'') ## replacing digits,numbers with spaces
    df_tmp['content'] = df_tmp['content'].apply(lambda x: ' ' .join ( x for x in x.split() if x not in stop_words)) # removing stopwords 
    df_tmp['content'] = df_tmp['content'].apply(lambda x: ' '.join([Word(x).lemmatize() for x in x.split()])) # lemmatize each word 
    return df_tmp


## Tokenize  cleaned data 

def tokenize(df,df_new,is_train):
    if is_train==1:
        tokenizer=Tokenizer(num_words=utils_vidisha.input_length,split=' ') #build tokenizer 
        tokenizer.fit_on_texts(df_new['content'].values) # fit tokenizer and tokenize the words in the training data
        with open('Output/tokenizer.pkl','wb') as handle:  # Save the tokenizer as pkl file for future use
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        x=tokenizer.texts_to_sequences(df['content'].values) # tokenize the text in new data using the trainedd tokenizer
        x=pad_sequences(x,utils_vidisha.input_length)
        return x

    else:
        with open('output/tokenizer.pkl','rb') as handle: # if not training load the already trained and saved tokenizer
            tokenizer=pickle.load(handle)
        x=tokenizer.texts_to_sequences(df['content'].values)
        x=pad_sequences(x,utils_vidisha.input_length)
        return x
        
 ## Function to call dependent functions 

def apply(path,is_train):
    print('preprocessing started....')

    df=pd.read_csv(path)[['content','score']] # get the dataframe
    stop_words=stopwords.words('english')
    print(f'Training data has {df.shape[0]} examples')
    df_new=cleaning(df,stop_words) # apply the pre dedfined cleannin8g function 

    y_data=pd.get_dummies(df['score']) # Convert categorical data in target variable for numerical data using one-hot encoding 
    x_data=tokenize(df,df_new,is_train)

    print('Data preprocessing completed')
    return x_data,y_data
    

## get prediction given a single review 

def get_prediction(review,ml_model):
    df = pd.DataFrame([review],columns ='content') # convert the Review string to a dataframe
    stop_words = stopwords.words('english')
    df_new = cleaning(df,stop_words) # preprocess data using the cleaning function

    x_data = tokenize(df,df_new,is_train=0) # get the tokenized xx data at the prediction phase 
    prediction = list(ml_model.predict(x_data)[0]) # Extract the first item of the prediction output, as it may contain a 2D format
    max_value=max(prediction)
    max_index= prediction.index(max_value) # Find the classification class with highest probability
    index=max_index+1 # return a 1-based index for the classification class
    
        
        

