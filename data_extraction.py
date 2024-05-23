# Script for Extracting Features from Fynd Catalog
import json
import urllib.parse
import os

# File names and paths setup
input_file_name = "new_products.json"
output_file_name = "extracted_data.json"

# Get the current script directory
script_directory = os.path.dirname(os.path.realpath(__file__))
input_file_path = os.path.join(script_directory, input_file_name)
output_file_path = os.path.join(script_directory, output_file_name)

# Create a URL for a given category
def create_category_url(category):
    base_url = "https://www.gofynd.com/products/?q="
    encoded_category = urllib.parse.quote_plus(category)
    category_url = f"{base_url}{encoded_category}"
    return category_url

# Read the input JSON file
with open(input_file_path, 'r') as infile:
    new_data = json.load(infile)

# Read the existing output JSON file if it exists
if os.path.exists(output_file_path):
    with open(output_file_path, 'r') as outfile:
        extracted_data = json.load(outfile)
else:
    extracted_data = []

# Process each dictionary in the new data list
for item in new_data:
    # Extract values
    name = item.get("name", "")
    slug = item.get("slug", "")
    brand_name = item.get("brand", {}).get("name", "")  # Extract brand name
    brand_logo_url = item.get("brand", {}).get("logo", {}).get("url", "")  # Extract brand logo URL
    slug_with_prefix = f"https://www.gofynd.com/product/{slug}"
    media_url = item.get("medias", [{}])[0].get("url", "")
    categories = item.get("categories", [{}])[0].get("name", "")
    category_url = create_category_url(categories)
    description = item.get("attributes", {}).get("description", "")

    # Create a dictionary with extracted values
    result_dict = {
        "title": name,
        "url": slug_with_prefix,
        "media": media_url,
        "category": categories,
        "category_url": category_url,
        "description": description,
        "brand_name": brand_name,  # Add extracted brand name
        "brand_logo": brand_logo_url  # Add extracted brand logo URL
    }

    # Add the dictionary to the result list
    extracted_data.append(result_dict)

# Write the combined data to the output JSON file
with open(output_file_path, 'w') as outfile:
    json.dump(extracted_data, outfile, indent=2)
