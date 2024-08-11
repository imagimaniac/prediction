

# %%

from flask import Flask, request, jsonify
import model

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = [data['car_ID'], data['wheelbase'], data['carlength'], data['carwidth'], data['carheight'],data['curbweight'], data['enginesize'], data['boreratio'], data['stroke'], data['compressionratio'],data['horsepower'], data['peakrpm'], data['citympg'], data['highwaympg'], data['fueltype_diesel'],data['fueltype_gas'], data['aspiration_std'], data['aspiration_turbo'], data['doornumber_four'],data['doornumber_two'], data['carbody_convertible'], data['carbody_hardtop'],data['carbody_hatchback'], data['carbody_sedan'], data['carbody_wagon'], data['drivewheel_4wd'],data['drivewheel_fwd'], data['drivewheel_rwd'], data['enginelocation_front'], data['enginelocation_rear'], data['enginetype_dohc'], data['enginetype_dohcv'],data['enginetype_l'], data['enginetype_ohc'], data['enginetype_ohcf'], data['enginetype_ohcv'],data['enginetype_rotor'], data['cylindernumber_eight'], data['cylindernumber_five'],data['cylindernumber_four'], data['cylindernumber_six'], data['cylindernumber_three'],data['cylindernumber_twelve'], data['cylindernumber_two']]
    prediction = model.predict(features)
    return jsonify({'predicted_price': prediction})

if __name__ == '__main__':
    app.run(debug=True)





#%%

#%%