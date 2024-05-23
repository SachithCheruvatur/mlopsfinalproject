import streamlit as st
import requests
import json

st.title("RNN Autosuggest")

# Input fields
seed = st.text_input("Enter Search Term", "Jeans")
num_steps = 100  # Fixed number of steps as per your curl command

def fetch_and_search_json(words):
    # URL of the JSON file in the Google Cloud bucket
    #github_url = "https://raw.githubusercontent.com/SachithCheruvatur/mlopsfinalproject/main/extracted_data.json"
    json_url = "https://storage.googleapis.com/mlopsfileprojectbucket/Extracted%20Features.json"

    
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
    
    # Return a shortlist of matching products
    short_list = matching_products[:6] if len(matching_products) >= 6 else matching_products
    
    return short_list

if 'suggestions' not in st.session_state:
    st.session_state.suggestions = []

if 'clicked_suggestion' not in st.session_state:
    st.session_state.clicked_suggestion = None

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
        
        st.session_state.suggestions = return_list
        st.session_state.clicked_suggestion = None
    else:
        st.write("Error:", response.status_code)
        st.write(response.json().get("detail", "Unknown error"))

if st.session_state.suggestions:
    st.write("Suggestions:")
    
    # Create buttons for all suggestions
    for i, suggestion in enumerate(st.session_state.suggestions):
        if st.button(suggestion, key=i):
            st.session_state.clicked_suggestion = suggestion

if st.session_state.clicked_suggestion:
    clicked_words = st.session_state.clicked_suggestion.split()
    st.write(f"Clicked suggestion: {st.session_state.clicked_suggestion}")
    st.write("Searching for products...")
    matching_products = fetch_and_search_json(clicked_words)

    if matching_products:
        # Display products
        st.header("Matching Products")
        for product in matching_products:
            title = product['title']
            media_url = product['media']
            product_url = product['url']
            category = product['category']

            # Display product title with thumbnail and a link in two columns
            col1, col2 = st.columns([1, 4])
            with col1:
                st.image(media_url, width=50)
            with col2:
                st.markdown(f"<a href='{product_url}' style='color: white; text-decoration: none;font-size:20px;'>{title}</a>", unsafe_allow_html=True)

        # Create two columns layout
        col1, col2 = st.columns(2)

        # Display categories in the first column
        with col1:
            st.header("Categories")
            displayed_categories = set()
            for product in matching_products:
                category_name = product['category']
                category_url = product.get('category_url', '')

                if category_name not in displayed_categories:
                    st.markdown(f"<a href='{category_url}' style='color: white; text-decoration: none;font-size:20px;'>{category_name}</a>", unsafe_allow_html=True)
                    displayed_categories.add(category_name)

        unique_brand_set = []

        # Display brands in the second column
        with col2:
            st.header("Brands")
            num_columns = 3
            columns = st.columns(num_columns)
            tracking = 0

            for product in matching_products:
                if product['brand_name'] in unique_brand_set:
                    continue
                if product['brand_name'] not in unique_brand_set:
                    brand_slug = product['brand_name']
                    brand_logo = product['brand_logo']
                    brand_url = f"https://www.gofynd.com/products/?brand={brand_slug.lower().replace(' ', '-')}"

                    link = f"<a href='{brand_url}' target='_blank'><img src='{brand_logo}' width='100' /></a>"
                    
                    columns[tracking].markdown(link, unsafe_allow_html=True)
                    
                    tracking += 1
                    if tracking == num_columns:
                        tracking = 0
                    unique_brand_set.append(product['brand_name'])





