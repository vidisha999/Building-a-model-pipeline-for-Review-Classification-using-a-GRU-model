import tensorflow
from tensorflow.keras.models import load_model


top_words = 5000
input_length = 500




## Save the trained model 
def save_model(model):
    model.save('Output/gru_model_vidisha.h5')
    return True 


## Load pre-trained model 
def load_model(model_path):
    model = None
    try:
        model = tensorflow.keras.models.load_model(model_pah)
    except:
        print("please enter the correct path")
        exit(0)
    return model 














