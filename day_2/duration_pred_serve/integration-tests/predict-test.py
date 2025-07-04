import requests

url = "http://localhost:9696/predict"

trip = {
    "PULocationID": 100,
    "DOLocationID": 102,
    "trip_distance": 30,
}

response = requests.post(url, json=trip).json()
print(response)
