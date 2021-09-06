#This ML project is focused on prediction climate changes based on past data. This is done by shaping, processing, and splitting climate data from over the last 100 years, in order to build a model that will train on past data to provide accurate climate-temparature change predictions.

import pandas as pd
dataframe = pd.read_csv('data1.csv')  # Read in the dataset
dataframe.dropna(inplace=True)


#process data to make it more palatable for training
dataframe = dataframe.drop(['Year'], axis = 1)
df = dataframe.transpose()
df.head


import matplotlib.pyplot as plt
import seaborn as sns #for plotting

#graph = sns.lineplot(x = dataframe['Year'], dataframe['Value'])
graph = sns.lineplot(x = np.linspace(1, 142, 142), y = df.values.reshape(-1))

graph.set_title('Global Climate Changes over last Century')
graph.set_ylabel('Temp Difference')
graph.set_xlabel('Years after 1880')


#now we look at the temperature progession on a log scale, since temps have been increasing exponentially
dt = dataframe.apply(np.log)
graph = sns.lineplot(x = np.linspace(1, 142, 142), y = dt.values.reshape(-1))
graph.set_title('Global Climate Changes over last Century (Logarithmic)')
graph.set_ylabel('Temp Difference')
graph.set_xlabel('Years after 1880')


#Now we perform the preprocessing of the climate data to prep for training
from sklearn import preprocessing
scaler =  preprocessing.MinMaxScaler()
#climate_data = dataframe.values
cli_data = df.transpose()
cli_data = scaler.fit_transform(data)

# Use this function to create train/test split
def train_test_split(arr: np.array, split = 0.70):
    train_size = int(len(arr) * split)
    test_size = len(arr) - train_size
    US_train, US_test = arr[0:train_size,:], arr[train_size:len(arr),:]
    print("train.shape: ", US_train.shape)
    print("test.shape: ", US_test.shape)
    return (US_train, US_test)

# Call train_test_split on climate data
train, test = train_test_split(cli_data, split = 0.70)

# Generate a dataset where X[n] contains the readings for the 'time_step' previous days 
# and y contains the reading for today.
def create_dataset(dataset, time_steps=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_steps-1):
		a = dataset[i:(i+time_steps), 0]
		dataX.append(a)
		dataY.append(dataset[i + time_steps, 0])
	return np.array(dataX), np.array(dataY)

# Choose the number of time steps that the model "looks back"
time_steps = 2

# Create your training dataset.
X_train, y_train = create_dataset(train, time_steps)
## Create your test dataset.
X_test, y_test = create_dataset(test, time_steps)

# Reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential, Model
from tensorflow.keras.optimizers import * 
from keras.layers import Dense, Input, Embedding
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error

# Build model architecture here
input_layer = Input(shape=(1,time_steps))
#x = Embedding(, 10)(input_layer)
x = LSTM(10)(input_layer) 
out = Dense(1)(x)
model = Model(input_layer, out)


opt = Adam(learning_rate = 0.0075)
# Compile model
model.compile(loss='mean_squared_error', optimizer=opt, metrics = ['accuracy'])

# Model summary
model.summary()

# Fit model
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2)

# Plot the Model loss
def plot_losses(hist):
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'])
    plt.show()
    
plot_losses(history)

# Make predictions
trainPredict = model.predict(X_train)
testPredict = model.predict(X_test)
# Invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([y_train])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([y_train])

# Shift train predictions for plotting
trainPredictPlot = np.empty_like(cli_data)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[time_steps:len(trainPredict)+time_steps, :] = trainPredict
# Shift test predictions for plotting
testPredictPlot = np.empty_like(cli_data)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(time_steps*2)+1:len(cli_data)-1, :] = testPredict
# Plot baseline and predictions
plt.plot(scaler.inverse_transform(cli_data))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.title("Actual climate data vs Model predictions")
plt.ylabel('Temp increase per year')
plt.xlabel('Years after 1880')
plt.legend(['Actual data', 'Model train data prediction', 'Model test data prediction'])
plt.show()

