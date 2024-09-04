# FashionMatch

Welcome to the FashionMatch project! This project utilizes a ResNet 50 Model to recommend fashion outfits based on various techniques including Intent Classification, Web Scraping, and Convolutional Neural Networks.

## Project Overview

The project is divided into several sub-models to provide a comprehensive understanding of its functionality.

## Demo Of the Project

https://github.com/DharmaWarrior/Fashion-Outfit-Recommender/assets/97218268/d2427b60-5720-4bc6-b55f-fa2677b8efbf


### Flow Diagram

![Fashion Outfit Generator (2)](https://github.com/DharmaWarrior/Fashion-Outfit-Recommender/assets/97218268/49c631cc-29d9-4832-955c-647568c134ee)

## Intent Classification

Intent classification involves understanding the user's intent from text data.

1. **Data Collection & Preprocessing** (`intent_preprocess.py`): Raw text data is preprocessed by converting it to lowercase, removing punctuation, and eliminating stopwords using NLTK's built-in stopwords list. Preprocessed data is saved in `preprocessed_intent_data.csv`.

2. **Feature Extraction using TF-IDF** (`intent_featureExtraction.py`): Numerical features are created from preprocessed sentences using the TF-IDF vectorizer. The TF-IDF matrix is saved in `tfidf_matrix.pkl`.

3. **Training and Evaluation of the Classifier** (`train_testclassifier.py`): The dataset is split into training and testing sets. An SVM classifier is initialized, trained on the training data, evaluated on the test data, and the trained model is saved.

## Web Scraping

Web scraping is employed to gather the latest fashion trends.

1. **Scraping Latest Trends**: Utilizing the Pinscrape library, information from platforms like Pinterest, Instagram, and Twitter is scraped. The focus is on Pinterest, and the library is configured to scrape 1 to 3 pages of data.

2. **Downloading and Analyzing Images**: Image details are downloaded after scraping. The scraper retrieves 1 to 3 image URLs based on relevance. Images are cross-checked using labels to ensure personalization.

3. **Personalized Recommendations**: By extracting the latest trends and providing personalized recommendations based on images, users are kept updated on trends tailored to their preferences.

## Convolutional Neural Network (CNN)

In this project, we propose a model that uses Convolutional Neural Network and the Nearest neighbor-backed recommender. As shown in the figure Initially, the neural networks are trained and then 
an inventory is selected for generating recommendations and a database is created for the items in the inventory. The nearest neighbor’s algorithm is used to find the most relevant products based on the 
input image and recommendations are generated.

![work-model](https://github.com/DharmaWarrior/Fashion-Outfit-Recommender/assets/97218268/15fcbf91-a058-42f7-ab48-f33920c6a617)

## Training the neural networks

Once the data is pre-processed, the neural networks are trained, utilizing transfer learning 
from ResNet50. More additional layers are added in the last layers that replace the architecture and 
weights from ResNet50 in order to fine-tune the network model to serve the current issue. The figure
 shows the ResNet50 architecture.

![resnet](https://github.com/DharmaWarrior/Fashion-Outfit-Recommender/assets/97218268/7746fa1a-f9c1-40b9-9e2f-399c2e74776c)


## Getting the inventory

The images from Kaggle Fashion Product Images Dataset. The 
inventory is then run through the neural networks to classify and generate embeddings and the output 
is then used to generate recommendations. The Figure shows a sample set of inventory data

![inventry](https://github.com/DharmaWarrior/Fashion-Outfit-Recommender/assets/97218268/55aa72d7-5597-4007-890b-9bce707e488a)

### Dataset Link

[Kaggle Dataset Big size 15 GB](https://www.kaggle.com/paramaggarwal/fashion-product-images-dataset)

[Kaggle Dataset Small size 572 MB](https://www.kaggle.com/paramaggarwal/fashion-product-images-small)


## Recommendation generation

To generate recommendations, our proposed approach uses Sklearn Nearest Neighbors Oh Yeah. This allows us to find the nearest neighbors for the 
given input image. The similarity measure used in this Project is the Cosine Similarity measure. The top 5 
recommendations are extracted from the database and their images are displayed.

## Installation
The steps to run:
1. `git clone https://github.com/DharmaWarrior/Fashion-Outfit-Recommender` - git clone the repo and `cd` in the folder
2. `pip install -r requirements.txt` - Use pip to install the requirement
3. `streamlit run main.py` - To run the web server, simply execute streamlit with the main recommender app:

## Acknowledgments

This project was made possible with the support of [Pinscrape](https://github.com/rmcgibbo/pinscrape) for web scraping and the [ResNet50 model](https://keras.io/api/applications/resnet/#resnet50-function) for Convolutional Neural Networks.

Feel free to contribute, report issues, or provide suggestions to enhance this project!

