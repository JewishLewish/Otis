import requests

url = 'http://localhost:8000/api'
headers = {'Content-Type': 'application/json'}

# Replace this with your actual JSON data
json_data = {
    "content": "We are giving away free discord nitro! Just join at discord.gg/booster"
}

response = requests.post(url, json=json_data, headers=headers)

if response.status_code == 200:
    print("POST request successful!")
    print("Response:", response.text)
else:
    print("POST request failed with status code:", response.status_code)