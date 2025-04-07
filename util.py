
import pickle
import json
import numpy as np

locations = None
data_columns = None
model = None

def get_estimated_price(location,sqft,bhk,bath):
    try:
        loc_index = data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index>=0:
        x[loc_index] = 1

    return round(model.predict([x])[0],2)


def load_saved_artifacts():
    print("loading saved artifacts...start")
    global  data_columns
    global locations

    with open(r"C:\Users\User\Desktop\flask\server\artifacts\columns.json", "r") as f:
        data_columns = json.load(f)['data_columns']
        locations = data_columns[3:]  # first 3 columns are sqft, bath, bhk

    global model
    if model is None:
        with open(r'C:\Users\User\Desktop\flask\server\artifacts\banglore_house_price_prediction.pickle', 'rb') as f:
            model = pickle.load(f)
    print("loading saved artifacts...done")

def get_location_names():
    return locations

def get_data_columns():
    return data_columns

if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('1st Phase JP Nagar',1000, 3, 3))
    print(get_estimated_price('1st Phase JP Nagar', 1000, 2, 2))
    print(get_estimated_price('Kalhalli', 1000, 2, 2)) # other location
    print(get_estimated_price('Ejipura', 1000, 2, 2))  # other location