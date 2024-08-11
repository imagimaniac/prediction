

# %%

import pickle

def load_model():
    with open('car_price_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

def predict(features):
    model = load_model()
    # Preprocess features if necessary
    prediction = model.predict([features])
    return prediction[0]




#%%