import pandas as pd
import time
import joblib
import argparse
from data_simulator import *

parser = argparse.ArgumentParser(description='Make predictions on simulated online data')
### --- These are command line inputs, you can just leave these as they are --- ###
parser.add_argument('--input_data',
                    default='../data/online_data.csv',
                    type=str,
                    help="The name of the file where the simulated live data is stored")

parser.add_argument('--refresh_output',
                    default=1,
                    type=int,
                    help="1 or 0: 1 to refresh outputs, 0 to carry on with previous ones")

parser.add_argument('--output_data',
                    default='../data/online_predictions.csv',
                    type=str,
                    help="The name of the file where the model predictions are saved")

parser.add_argument('--debug',
                    default=0,
                    type=int,
                    help="1 or 0: 1 to show each prediction as it's made, 0 to show none")

args = parser.parse_args()
### ---------------------------------------------------------------------------- ###

#  Load the saved model --> You should probably load your own model here
saved_model_file = '../xgboost_all_data.pkl'
rf_model = joblib.load(saved_model_file)

# Function that clears the simulated streaming data file so it starts again from time step 0
if args.refresh_output:    
    refresh_streaming_data()
    if os.path.exists(args.output_data):
        os.remove(args.output_data)

# Creates the streaming file with predictions if it doesn't already exist
if not os.path.exists(args.output_data):
    with open(args.output_data, 'a') as f:
            f.writelines('timestamp,value,prediction\n')

# Creates new features out of the value in the dataset
def add_features(data):
    """function to add some time series related features"""
    # create squared data
    data['data_squared'] = data['data'] ** 2
    data['data_tripled'] = data['data'] ** 3
    # create lag
    for i in range(100):
        data[f'data_lag_{i + 1}'] = data['data'].shift(i + 1)
        
    # get de-fragmented frame for performance reasons
    data = data.copy()
    
    return data

running_live = True
print('--- Model making sequential predictions. Press CTRL+C to stop ---')
while running_live:
    # Create the simulation of the live data. You must leave this as is
    stream_data()
    online_data = pd.read_csv(args.input_data)
    
    # Enrich data with a lot more features
    online_data_np = add_features(online_data).values

    ### This is a basic example of how you could make predictions on the latest data point
    ### with your saved model. You're welcome to re-write any of this code so that it works
    ### with your model

    # Make a prediction on the latest data point
    live_pred = rf_model.predict(online_data_np[[-1]])
    
    # See the predictions if you set --debug when calling this script
    print(live_pred[0]) if args.debug else ""

    # Write out your data to the streaming prediction file
    with open(args.output_data, 'a') as f:
        f.writelines(f'{online_data.shape[0]},{online_data.iloc[-1, 0]},{live_pred[0]}\n')

