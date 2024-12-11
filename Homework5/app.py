import pickle
from flask import Flask, request, jsonify

# loading
with open('model1.bin', 'rb') as model_file:
    model = pickle.load(model_file)

with open('dv.bin', 'rb') as dv_file:
    dv = pickle.load(dv_file)

# flask app
app = Flask(__name__)

# defining route
@app.route('/predict', methods=['POST'])
def predict():
    # get client input
    client = request.get_json()

    # transforming
    X = dv.transform([client])

    # predicting
    y_pred = model.predict_proba(X)[0, 1]

    # return as json
    result = {'subscription_probability': y_pred}
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
