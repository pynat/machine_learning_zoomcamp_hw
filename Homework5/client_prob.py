import pickle

# loading dictionary vectorizer
with open('dv.bin', 'rb') as f_in:
    dv = pickle.load(f_in)

# loading model
with open('model1.bin', 'rb') as f_in:
    model = pickle.load(f_in)

# client data pred
client = {"job": "management", "duration": 400, "poutcome": "success"}

# transform
X = dv.transform([client])

# pred
pred_prob = model.predict_proba(X)[0, 1]

# print
print(f"Probability of subscription: {pred_prob:.3f}")