import requests

url = "https://ci-example-bgmr.onrender.com/predict"
payload = {
    "data": [
        [5.1, 3.5, 1.4, 0.2],
        [6.2, 3.4, 5.4, 2.3]
    ]
}

response = requests.post(url, json=payload)
print(response.json())