# Use Flask web framework to build the REST API to deploy the model by allowing access of the python functions through web accessible endpoints

from flask import Flask,request  
import ML_pipeline_vidisha.utils_vidisha
from ML_Pipeline_vidisha.preprocess_vidisha import apply_prediction

# creates a Flask app instance to register endpoints & "__name__" python built in variable determines how the script is executed 
app=Flask(__name__)

model_path='Output/gru_model_vidisha.h5'
ml_model=utils_vidisha.load_model(model_path)

## HTTP method is how client communicates with the sever, POST HTTP method is used to send review data in JSON format to the server "app" at the local endpoint using a POST request URL "/get-review-score"

@app.post("/get-review-score")
def get_image_class():
    data=request.get_json() # parse JSON directly from the body of HTTP request to a python dictionary
    review=data['review'] # extract the value or content in the review key 
    prediction=apply_prediction(review,model)
    output= { "Review Score" : prediction}
    return output 


# Only allow Flask to run the app/API, when script is executed directly and won't run if the file is imported as a module 
# Runs at the port "50001" and with the host "0.0.0.0" allowing the sever accessible to any IP address 
if __name__== "__main__":
    app.run(host='0.0.0.0', port=5001)
    

























from flask import Flask, request

import Utils
# from ML_Pipeline.Preprocess import apply_prediction

app = Flask(__name__)

model_path = '../output/gru-model.h5'
ml_model = Utils.load_model(model_path)


@app.post("/get-review-score")
def get_image_class():
    from ML_Pipeline.Preprocess import apply_prediction
    data = request.get_json()
    review = data['review']
    prediction = apply_prediction(review, ml_model)
    output = {"Review Score": prediction}
    return output


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
