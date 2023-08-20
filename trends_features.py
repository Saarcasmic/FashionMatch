import os
import numpy as np
import requests
import pickle
import streamlit as st
import pandas as pd
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm


data = pd.read_csv('./styles.csv')

# saving the fetched file from pininterest for future reference
def save_uploaded_file(image_url):
    response = requests.get(image_url)
    if response.status_code == 200:
        # Extract the filename from the URL
        filename = os.path.basename(image_url)
        
        # Construct the full path to save the image
        save_path = os.path.join('uploads', filename)
        
        # Write the image data to the local file
        with open(save_path, "wb") as f:
            f.write(response.content)
        
        print(f"Image downloaded and saved as {save_path}")
        return 1
    else:
        print("Failed to download the image")
        return 0

# extracting features for the image
def feature_extraction(img_path,model):
    if img_path is not None:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        result = model.predict(preprocessed_img).flatten()
        normalized_result = result / norm(result)

        return normalized_result

# it will give the product by finding the nearestNeighbour 
def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices

# Loading the features of inverntory data one contains image embeddings and other contain the filenames
feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))



# It will show the recommendation
def recommendation(image_path, model, gender):
    features = feature_extraction(image_path, model)

    # Recommendation
    indices = recommend(features, feature_list)

    # Display recommended images
    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]
    
    

    # Printing the results here in filtered data you can check brand in brands array that is most searched brands, or colors
    # according to your dataset here i had filtered with gender 
    for i, col in enumerate(cols):  
        file_path = filenames[indices[0][i]]
        file_name = os.path.splitext(os.path.basename(file_path))[0]
                    
                    
        filtered_data = data[(data['id'] == int(file_name)) &
                                (data['gender'] == gender)]

        if not filtered_data.empty :
            st.image(filenames[indices[0][i]], width=300)
            
# converting to keywords according to the labels
def gendertokeyword(gender):
    if gender == "male":
        return "Men"
    else:
        return "Women"