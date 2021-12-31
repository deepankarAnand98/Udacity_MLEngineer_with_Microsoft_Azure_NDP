import requests
import json

# URL for the web service, should be similar to:
# 'http://8530a665-66f3-49c8-a953-b82a2d312917.eastus.azurecontainer.io/score'
scoring_uri = 'http://ce3b602e-c1f5-46c6-8294-6a6284752384.eastus2.azurecontainer.io/score'
# If the service is authenticated, set the key or token
key = '2Xvd3GBKlgujNvqi53yw47Emkl1wZtMm'

# Two sets of data to score, so we get two results back
data = {
  "data": [
                    {
                        "age": 30,
                        "job": "technician",
                        "marital": "married",
                        "education": "professional.course",
                        "default": "no",
                        "housing": "yes",
                        "loan": "no",
                        "contact": "telephone",
                        "month": "may",
                        "day_of_week": "mon",
                        "duration": 365,
                        "campaign": 3,
                        "pdays": 999,
                        "previous": 1,
                        "poutcome": "failure",
                        "emp.var.rate": -1.4,
                        "cons.price.idx": 92.893,
                        "cons.conf.idx": -46.2,
                        "euribor3m": 4.981,
                        "nr.employed": 5228.1
                    },
                    {
                        "age": 47,
                        "job": "housemaid",
                        "marital": "married",
                        "education": "basic.6y",
                        "default": "unknown",
                        "housing": "no",
                        "loan": "yes",
                        "contact": "cellular",
                        "month": "jul",
                        "day_of_week": "thu",
                        "duration": 148,
                        "campaign": 1,
                        "pdays": 999,
                        "previous": 0,
                        "poutcome": "nonexistent",
                        "emp.var.rate": -2.9,
                        "cons.price.idx": 92.469,
                        "cons.conf.idx": -33.6,
                        "euribor3m": 1.072,
                        "nr.employed": 5076.2,
                    }
                ]
    }
# Convert to JSON string
input_data = json.dumps(data)
with open("data.json", "w") as _f:
    _f.write(input_data)

# Set the content type
headers = {'Content-Type': 'application/json'}
# If authentication is enabled, set the authorization header
headers['Authorization'] = f'Bearer {key}'

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.json())


