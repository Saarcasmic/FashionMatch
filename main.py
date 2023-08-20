import streamlit as st
import os
import joblib
import pandas as pd
import requests
import tensorflow
from PIL import Image
from keras.layers import GlobalMaxPooling2D
from keras.applications.resnet50 import ResNet50,preprocess_input
from pinscrape import pinscrape
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from categoriesMapping import myDictionary
from trends_features import *
from personalised_results import *
from intent_output import*

# Creating the model
model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])
    
st.title('Fashion Recommender System')




# creating intent scrapword for finding latest trends
intent_scraptWord = ""
intent_scraptWord += 'Single'

# Text input for user to enter a sentence 
user_age = st.number_input("Enter age:")
user_gender = st.text_input("Enter gender:", "")
user_gender = user_gender.lower()
user_gender = gendertokeyword(user_gender)


if user_age < 7 :
    intent_scraptWord += "Kid "
elif user_age < 18:
    intent_scraptWord += "Teen "
elif user_age < 55:
    intent_scraptWord += "Young "
else:
    intent_scraptWord += "Old "
    
intent_scraptWord += user_gender
    


# Text input for user to enter a sentence
user_input = st.text_input("Enter Description / Occassion to let us know:", "")

user_occassion = intentOutput(user_input)


intent_scraptWord += " "

load_button = st.button('Submit')

if load_button :
    # checking for the intent, browsing history, purchasing history, brands and putting that product image in our model ResNet50
    if user_occassion == 'Diwali Festival Shopping' :
        url_path = display_personalised(myDictionary['Ethnic'][0], 'Ethnic', user_gender)
        intent_scraptWord += "Ethnic Wear" + user_input
        if url_path is not None:
            recommendation(url_path, model, user_gender)
        
    elif user_occassion == 'Party Outfit Suggestions':
        url_path = display_personalised(myDictionary['Casual'][0], 'Casual', user_gender)
        if url_path is not None:
            recommendation(url_path, model, user_gender)
        intent_scraptWord += "Party wear" + user_input
        
    elif user_occassion == 'Formal Event Attire':
        url_path = display_personalised(myDictionary['Formals'][0], 'Formal', user_gender)
        if url_path is not None:
            recommendation(url_path, model, user_gender) 
        intent_scraptWord += "Formals"+ user_input
        
    elif user_occassion == 'Gym Wear Recommendations':
        url_path = display_personalised(myDictionary['Sportswear'][1], 'Casual', user_gender)
        if url_path is not None:
            recommendation(url_path, model, user_gender)
        intent_scraptWord += "Gym Wear"
        
    elif user_occassion == 'Casual Everyday Outfits':
        url_path = display_personalised(myDictionary['Casual'][0], 'Casual', user_gender)
        if url_path is not None:
            recommendation(url_path, model, user_gender)
        intent_scraptWord += user_input
        
    elif user_occassion == 'Winter Clothing Shopping':
        url_path = display_personalised(myDictionary['Winter'][0], 'Casual', user_gender)
        if url_path is not None:
            recommendation(url_path, model, user_gender)
        intent_scraptWord = intent_scraptWord + "Casual Wear" + user_input
    else:
        if "beach" in user_input:
            intent_scraptWord += "bikinis for beach"
        else:
            intent_scraptWord += user_input
        
    
    intent_scraptWord += user_gender
    
    # finding the latest trends from the pininterest, this can also be find by instagram, twitter and all
    details = pinscrape.scraper.scrape(intent_scraptWord, "output", {}, 1, 3)

    if details["isDownloaded"]:
        # loading the images details and there will be 1 to 3 images scrapped from pininterest images relevancy is cross checked bby image labels to be more personalsied
        for image_url in details['url_list']:
            if save_uploaded_file(image_url):
                response = requests.get(image_url)
                display_image = Image.open(BytesIO(response.content))
                # st.image(display_image)
                recommendation(os.path.join("uploads",os.path.basename(image_url)), model, user_gender)
            
    else:
        print("\nNothing to download !!")
    