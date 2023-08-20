import streamlit as st
import os
import pandas as pd

def display_personalised(articleType, usage, gender):
    if gender == "male":
        gender = "Men"
    else:
        gender = "Women"
    
    # Read the CSV file into a DataFrame
    data = pd.read_csv('./styles.csv')

    # These can be retrieved from the sessions so taking this and this will vary from user to user
    # Brands and colors based on browsing and purchasing history
    brands = ["Levis", "Nike", "Adidas", "Puma", "US Polo", "Peter"]
    colors = [ "Black", "White"]

    # Filter the rows based on productDisplayName, color, usage, brands
    filtered_data = data[ data['productDisplayName'].str.contains(articleType) 
                            & data['productDisplayName'].str.contains(usage) 
                            & data['gender'] == gender]

    # Display the filtered product details & give is as input in our model to search from that easily
    if not filtered_data.empty:
        if filtered_data.iloc[0]['gender'] == gender:
            first_element_id = filtered_data.iloc[0]['id']
            image_path = os.path.join("images", str(first_element_id) + ".jpg")
            
            if os.path.exists(image_path):
                st.image(image_path, width=300)
                return image_path
            else:
                print("Failed")
    else:
        return None