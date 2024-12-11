import requests

# url where server is running
url = "http://localhost:9696/predict"  
# client data
client = {"job": "student", "duration": 280, "poutcome": "failure"}  

# sending post request
response = requests.post(url, json=client)

# print
print(response.json())