import subprocess # To sseperately execute ochestrated multiple processes

from ML_pipeline_vidisha import train_model_vidisha
from ML_pipeline_vidisha.preprocess_vidisha import apply 
from ML_pipeline_vidisha.utils_vidisha import load_model,save_model 

## input training = 0, prediction = 1, deployment = 2
val= int(input("Train - 0\nPredict - 1\nDeploy - 2\n Enter your value: ")) # create a basic input system
if val == 0:
    x_train,y_train=apply("Data/review_data.csv",is_train=1) # Implement apply function 
    ml_model=train_model_vidisha.fit(x_train,y_train) # Implement fit model's fit function
    model_path = save_model(ml_model) # save the trained model
    print(" Trained model is saved in Output/gru_model_vidisha.h5")

elif val==1:
    model_path="Output/gru_model_vidisha.h5"
    ml_model=load_model(model_path) # load pre-trained model for prediction 
    x_test,y_test=apply(model_path,is_train=0)
    print( f"Testing accuracy of the model : {ml_model.evaluate(x_test,y_test)[1]*100.0} % ")
    
else:
    ## for production deployment
    ''' process=subprocess.Popen(['sh','ML_pipeline_vidisha/wsgi.sh'],
    stdout=subprocess.PIPE,stderr=subprocess.PIPE,uinversal_newlines=True)''' # open the shell script

    ## for development deployment 
    process=subprocess.Popen(['python','ML_pipeline_vidisha/deploy.py'], # execute script in a new process, uses .py file
                             stdout=subprocess.PIPE,   #captures standard output of the process
                             stderr=subprocess.PIPE,   # captures standard error of the process
                             uinversal_newlines=True)  # output and error messages are return as strings rather than byte format

    for stdout_line in process.stdout:
        print(stdout_line) 
        
    stdout_line,stderr=process.commuincate()  # blocking call waits for process to finish executing
    print(stdout_line,stderr)              






