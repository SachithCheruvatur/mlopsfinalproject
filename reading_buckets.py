import requests

# Define the URL of the test file
url = "https://storage.googleapis.com/mlopsfileprojectbucket/testfile.json"

# Send a GET request to retrieve the file content
response = requests.get(url)

# Check for successful response (status code 200)
if response.status_code == 200:
  # Get the content as a string
  content = response.text
  # Print the content
  print(content)
else:
  print(f"Error: Could not retrieve file. Status code: {response.status_code}")
