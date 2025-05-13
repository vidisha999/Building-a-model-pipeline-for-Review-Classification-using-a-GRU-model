from tensorflow.keras.layers import Dense, GRU, Embedding, SpatialDropout1D
from tensorflow.keras.models import Sequential 

from ML_pipeline_vidisha import utils_vidisha # load the utility file from the saved folder 


# Function to train ML model
def train(model, x_train, y_train):
    batch_size = 32
    epochs = 50
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose='auto')
    return model


# Function to initiate model and training data
def fit(x_train, y_train):
    model = Sequential()
    model.add(Embedding(input_dim=utils_vidisha.top_words, output_dim=120))
    model.add(SpatialDropout1D(0.4))
    model.add(GRU(64, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    model = train(model, x_train, y_train)

    return model
