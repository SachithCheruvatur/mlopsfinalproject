# # streamlit_app/app.py

# import streamlit as st
# import requests

# st.title("RNN Autosuggest")

# # Input fields
# seeds = st.text_input("Enter Search Term", "sweater")
# #num_steps = st.number_input("Number of steps", min_value=1, max_value=100, value=30)
# num_steps = 100

# # Convert seeds input to list
# seeds_list = seeds.split(',')

# if st.button("Generate"):
#     return_list = []

#     # Prepare data for FastAPI request
#     data = {
#         "seeds": seeds_list,
#         "num_steps": num_steps
#     }
    
#     # FastAPI endpoint URL
#     url = "http://34.93.45.146:8000/generate-suggestions-2/"
    
#     # Send POST request to FastAPI
#     response = requests.post(url, json=data)
    
#     if response.status_code == 200:
#         suggestions = response.json().get("suggestions", [])
#         st.write("Suggestions:")
#         for suggestion in suggestions:
#             res_list = suggestion.splitlines()
#             return_list.append (res_list[0])
#         for i in return_list:
#             st.write(i)

   

    
#     else:
#         st.write("Error:", response.status_code)
#         st.write(response.json().get("detail", "Unknown error"))



# streamlit_app/app.py

# import streamlit as st
# import requests

# st.title("RNN Autosuggest")

# # Input fields
# seed = st.text_input("Enter Search Term", "Jeans")
# num_steps = 100  # Fixed number of steps as per your curl command

# if st.button("Generate"):
#     return_list = []
#     # Prepare data for FastAPI request
#     data = {
#         "seeds": [seed],  # Send a list with a single seed item
#         "num_steps": num_steps
#     }
    
#     # FastAPI endpoint URL
#     url = "http://34.93.45.146:8000/generate-suggestions-2/"
    
#     # Send POST request to FastAPI
#     response = requests.post(url, json=data)
    
#     if response.status_code == 200:
#         suggestions = response.json().get("suggestions", [])
#         st.write("Suggestions:")
#         for suggestion in suggestions:
#             res_list = suggestion.splitlines()
#             return_list.append (res_list[0])
#         for i in return_list:
#             st.write(i)
#     else:
#         st.write("Error:", response.status_code)
#         st.write(response.json().get("detail", "Unknown error"))


import streamlit as st
import requests
import json

st.title("RNN Autosuggest")

# Input fields
seed = st.text_input("Enter Search Term", "Jeans")
num_steps = 100  # Fixed number of steps as per your curl command

def fetch_and_search_json(words):

    # URL of the JSON file in the Google Cloud bucket
    json_url = "https://storage.googleapis.com/mlopsfileprojectbucket/Redacted_Catalog.json"
    
    # Fetch the JSON file
    response = requests.get(json_url)
    if response.status_code == 200:
        product_list = response.json()
    else:
        st.write("Error fetching JSON:", response.status_code)
        return []

    # Search for matching products
    matching_products = []
    for product in product_list:
        product_title = product.get("title", "").lower()
        if any(word.lower() in product_title for word in words):
            matching_products.append(product)
    
    return matching_products

if st.button("Generate"):
    
    return_list = []
    # Prepare data for FastAPI request
    data = {
        "seeds": [seed],  # Send a list with a single seed item
        "num_steps": num_steps
    }
    
    # FastAPI endpoint URL
    url = "http://34.93.45.146:8000/generate-suggestions-2/"
    
    # Send POST request to FastAPI
    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        suggestions = response.json().get("suggestions", [])
        st.write("Suggestions:")
        for suggestion in suggestions:
            res_list = suggestion.splitlines()
            return_list.append(res_list[0])
        for i in return_list:
            if st.button(i):  # Make each suggestion a button
                clicked_words = i.split()
                st.write(f"Clicked suggestion: {i}")
                st.write("Searching for products...")
                matching_products = fetch_and_search_json(clicked_words)
                st.write("Matching Products:")
                for product in matching_products:
                    st.write(f"Title: {product['title']}")
                    st.write(f"URL: {product['url']}")
                    st.image(product['media'])
                    st.write(f"Category: {product['category']}")
                    st.write(f"Description: {product['description']}")
                    st.write(f"Brand: {product['brand_name']}")
                    st.write(f"Brand Logo: {product['brand_logo']}")
                    st.write("----")
    else:
        st.write("Error:", response.status_code)
        st.write(response.json().get("detail", "Unknown error"))


