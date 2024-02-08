import requests

# URL of the Flask server
url = "http://35.231.229.2:88/image"


# Path to the image file you want to send

image_path = "4.jpeg"

# Send the POST request with the image file
with open(image_path, "rb") as file:
    files = {"image": file}
    response = requests.post(url, files=files)

# Print the response from the server
print(response.text)
