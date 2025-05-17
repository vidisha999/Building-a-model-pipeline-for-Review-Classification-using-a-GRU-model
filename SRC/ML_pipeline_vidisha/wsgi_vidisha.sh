# gunicorn is a WSGI(Web server Gateway Interface), a standard interface that handles the communication between a web server( HTTP requests)and Python web application (Flask).


#!/bin/bash
gunicorn -b 0.0.0.0:5001 -w 4 -t 5 wsgi_vidisha:app


## setting up a gunicorn server to run the wsgi application 
## 1. Runs the gunicorn server 
## 2. Binds the server to port 5001 making it accessible from all network ineferences
## 3. -w 4  uses 4 worker processes to handle requests.
## 4. -t 5 Sets a timeout of 5 seconds before killing a worker if it doesn’t respond.
## 5. Specify the wsgi entry point for the defined app 
