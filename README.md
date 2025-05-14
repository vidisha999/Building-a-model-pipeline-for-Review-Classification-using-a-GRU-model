# Building-a-model-pipeline-for-Review-Classification-using-a-GRU-model

## Project Description
The purpose of this project is to build a review classification model, that does sentiment analysis of the review text data from an application which is scaled in the rage of 1 to 5, to identify 5 as a positive sentiment and 1 as a negative sentiment. The deep learning model was built using GRU (Gated Recurrent Unit) which is a variant of RNN ( Recurrent Neural Network). The model is trained on the training review data and their properties to predict the sentiment. This project outlines an automated model pipeline from building model to its deployment enabling the use of deployed model in a scalable way to make real-time decisions based on the sentiment predictions of the app reviews.

## GRU ---importance of GRU as a RNN 
## Objective 
## Modular code overview 
## Data 
The dataset contains `content` and `score` columns that respectively represent the review text and the rating given by the user based on their review. The score contains values between 0-5, reflecting the sentiment expressed in the review.

## The model engine 
In order to build a robust and streamlined model pipeline, the python submodule **subprocess** was used to ochestrate multiple tasks in parallel using pre-defined functions. It leverages the subprocess module to trigger and run external scripts in seperate processes enabling modular execution of training, prediction or depoyment of the model in both development or production environment. The [model engine](SRC/engine_vidisha.py) demonstrates the ochestration of seperate scripts for data preprocessing, model training and deployment.
