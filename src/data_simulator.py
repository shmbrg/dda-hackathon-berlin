import pandas as pd
import argparse
import os
import time


# If our simulated streaming file exists and we want to start anew, delete the file
def refresh_streaming_data():
    output_data = '../data/online_data.csv'
    if os.path.exists(output_data):
        os.remove(output_data)

def stream_data():
    '''
    Function that simulates what a live stream of data would look like
    '''

    input_data = '../data/ADT_test.csv'
    output_data = '../data/online_data.csv'
    cycle_time = 1.0
    debug = False

    print(f' Data is saved in: {output_data}') if debug else ""

    input_data = pd.read_csv(input_data)
    input_len = input_data.shape[0]

    # Create the file if it doesn't exist
    if not os.path.exists(output_data):
        with open(output_data, 'a') as f:
                # Changed "value" to "data", because that is what has been used in the training dataset
                f.writelines('data\n')

    output_data_file = pd.read_csv(output_data)

    last_pos = output_data_file.shape[0]
    # For each of the rows in the test data, write each one out one at a time to a csv file
    # This simulates the live data
    time.sleep(cycle_time)
    with open(output_data, 'a') as f:
        f.writelines(f'{input_data.iloc[last_pos, 0]}\n')
    if debug:
        print(f'At pos {i} just wrote the value: {input_data.iloc[last_pos, 0]}')

    print(' --- Restarting data generation --- ') if debug else ""
        