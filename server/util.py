import json
import pickle
import numpy as np
__locations=None
__data_columns=None
__model=None

def get_estimated_price(location,sqft,bhk,bath):
    try:
        loc_index=__data_columns.index(location.lower())
    except:
        loc_index=-1
        
    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
        
    return round(__model.predict([x])[0],2)

def get_location_names():
    load_saved_artifacts()
    return __locations

def load_saved_artifacts():
    print("Loading saved artifacts...start")
    global __data_columns
    global __locations
    global __model
    
    with open("./server/artifacts/columns.json",'r') as f:
        __data_columns=json.load(f)['data_columns'] #for dictionary
        __locations=__data_columns[3:]#starting from index 3 to end
    
    with open("./server/artifacts/banglore_home_prices_model.pickle",'rb') as f:# rb to read binary
        __model=pickle.load(f)
    print("Loading saved artifacts...done")
    
if __name__=='__main__':
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('vijayanagar',1000,2,2))