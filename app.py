# streamlit_app/app.py

import streamlit as st
import requests

st.title("RNN Autosuggest")

# Input fields
seeds = st.text_input("Enter Search Term", "sweater")
#num_steps = st.number_input("Number of steps", min_value=1, max_value=100, value=30)
num_steps = 100

# Convert seeds input to list
seeds_list = seeds.split(',')

if st.button("Generate"):
    return_list = []

    # Prepare data for FastAPI request
    data = {
        "seeds": seeds_list,
        "num_steps": num_steps
    }
    
    # FastAPI endpoint URL
    url = "http://34.93.45.146:8000/generate-suggestions/"
    
    # Send POST request to FastAPI
    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        suggestions = response.json().get("suggestions", [])
        st.write("Suggestions:")
        for suggestion in suggestions:
            res_list = suggestion.splitlines()
            return_list.append (res_list[0])
        st.write(return_list)

   

    
    else:
        st.write("Error:", response.status_code)
        st.write(response.json().get("detail", "Unknown error"))
