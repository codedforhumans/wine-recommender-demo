import os
import string
import re
from collections import Counter, OrderedDict
from operator import itemgetter

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.phrases import Phraser, Phrases
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

import gensim
from gensim.models import KeyedVectors

from backend.database import Database
# from database import Database


database = Database()

class Recommender():

    def __init__(self):
        print("__init__")
        self.model = Model()
        self.model_knn = database.get_model_knn()
        self.dict_of_tfidf_weightings = self.model.get_dict_of_tfidf_weightings()
        self.model_w2v = database.get_Word2Vec().wv
        self.kv_words = database.get_KeyedVectors().key_to_index
        self.kv_model = database.get_KeyedVectors()
        self.wine_review_min_count = database.get_wine_review_min_count()
        # pass

    def get_data_object(self):
        print("get_data_object")
        return self.data
    
    def clean_text(self, text):
            if text is not None:
                return text.lower()
            else:
                return text

    def run_recommender(self, descriptors_trial_raw):
        print("run_recommender")
        # Words Saruul really wanted to use
        # descriptors_trial = ['aesthetic', 'acidic', 'lively', 'tasty', 'champagne', 'victorian']

        default_wine_descriptor = "rich"

        def descriptor_str_to_list(raw_string):
            separated = re.split('[^a-zA-Z]\W', raw_string)
            strip_white = [phrase.strip() for phrase in separated]
            no_empty_str = list(filter(None, strip_white))
            result = no_empty_str
            return no_empty_str

        descriptor_list = descriptor_str_to_list(descriptors_trial_raw)

        features, used_descriptors = self.find_features(descriptor_list)

        descriptors_left = list(set(descriptors_trial_raw) - set(used_descriptors))
        pure_descriptors = [self.clean_text(text) for text in descriptors_left]
        if len(pure_descriptors) == 0:
            pure_descriptors = [default_wine_descriptor]

        wine_descriptors = self.find_closest_wine_descriptors(pure_descriptors)

        result = self.descriptors_to_best_match_wines(list_of_descriptors = wine_descriptors, number_of_suggestions = 5, features = features)
        return result
    
    def find_features(self, descriptors):
        """
        Returns tuple consisting of (feature dictionary, the list of strings used from original descriptors)
        """
        features = {}

        varieties = self.get_varieties_min_count()
        countries = self.get_countries_min_count()
        # countries_found = list(set(descriptors) & set(countries))
        countries_case_insensitive = [country.lower() for country in countries]
        countries_found = [phrase for phrase in descriptors if phrase.lower() in countries_case_insensitive]
        if len(countries_found) != 0:
            features["Country"] = countries_found

        descriptors_no_country = list(set(descriptors) - set(countries_found))
        # used_strings = []
        
        for string in descriptors_no_country:
            variety_temp = [x for x in varieties if re.search(string, x, re.IGNORECASE)]
            if len(variety_temp) != 0:
                # used_strings += [string]
                if 'Variety' not in features:
                    features["Variety"] = variety_temp
                else:
                    features["Variety"] += variety_temp
        used_strings_unique = []
        for key in features.keys():
            used_strings_unique += features[key]
        # Make sure no duplicate value
        used_strings_unique = list(set(used_strings_unique))
        return (features, used_strings_unique)


    def find_closest_wine_descriptors(self, word_list, wine_descriptor_list = database.get_level_3_mapping()):
        return [self.find_closest_word(word, wine_descriptor_list) for word in word_list if word in self.kv_model.key_to_index]
    
    def find_closest_word(self, word, word_list):
        model = self.kv_model
        # Check if the word exists in the model's vocabulary
        if word not in model.key_to_index:
            print("Word not found in the vocabulary.")

        # Calculate the similarity scores between the given word and the words in the list
        similarity_scores = {w: model.similarity(word, w) for w in word_list if w in self.kv_words} 
        # Find the word with the highest similarity score
        closest_word = max(similarity_scores, key=similarity_scores.get)

        return closest_word
    
    def descriptors_to_best_match_wines(self, list_of_descriptors, number_of_suggestions=10, features: dict = {}):
        weighted_review_terms = []
        for term in list_of_descriptors:
            if term not in self.dict_of_tfidf_weightings:
                if term not in database.descriptor_mapping["raw descriptor"]:
                    print('choose a different descriptor from', term)
                    continue
                else:
                    term = database.descriptor_mapping['normalized'][term]
            tfidf_weighting = self.dict_of_tfidf_weightings[term]
            word_vector = self.model_w2v.get_vector(term).reshape(1, 300)
            weighted_word_vector = tfidf_weighting * word_vector
            weighted_review_terms.append(weighted_word_vector)
        review_vector = sum(weighted_review_terms)

        # Helpers
        def feature_filter(feature_name, feature_descriptor: list, wine_reviews_mincount_df):
            """
            Returns DataFrame that subsets wine_reviews_mincount_df into the correct feature_descriptor (list) for the feature_name
            i.e. feature_filter("Variety", "Riesling", wine_reviews_mincount)
            """
            if feature_name not in wine_reviews_mincount_df.columns:
                print("Found no feature: " + str(feature_name))
                return wine_reviews_mincount_df
            else:
                return wine_reviews_mincount_df.loc[wine_reviews_mincount_df[feature_name].isin(feature_descriptor)]

            
        def get_feature_matching_indices(search_index, dataframe):
            """
            Returns a list of indices in search_index that are available in dataframe's indices
            """
            
            indices = dataframe.index
            return [ind for ind in search_index if ind in indices]


        def get_feature_unmatching_indices(search_index, dataframe):
            """
            Returns a list of indices NOT in search_index that are available in dataframe's indices - for edge cases when asked variety is limited in result count
            """
            indices = dataframe.index
            return [ind for ind in search_index if ind not in indices]

        def get_filtered_suggestions(features: dict, search_index, suggestion_count):
            # Filter by each feature_name
            curr_df = self.wine_review_min_count

            if len(features) > 0:
                for feature_name, feature_descriptor in features.items():
                    curr_df = feature_filter(feature_name, feature_descriptor, curr_df)
                
            # Find indices available in the curr_df indices
            avail = get_feature_matching_indices(search_index, curr_df)
            
            # Find indices NOT available in the curr_df indices
            # no_avail = get_feature_unmatching_indices(search_index, curr_df)
            
            result_indices = avail
            if len(result_indices) < suggestion_count:
                extra = suggestion_count - len(result_indices)
                result_indices = avail # + no_avail[:extra]
                return result_indices
            else:
                return avail[:suggestion_count]
        
        distance, indice = self.model_knn.kneighbors(review_vector, n_neighbors = 15041) # # 15041 is the limit of n_samples (number_of_suggestions + 1)

        # distance_list = distance[0].tolist()[1:]
        indice_list_raw = indice[0].tolist()[1:]


        def clean_features(features: dict):
            result_dict = features
            if 'Country' in result_dict.keys():
                cleaned_countries = [country.title() if not country.isupper() else country for country in result_dict['Country']]
                result_dict['Country'] = cleaned_countries
            return result_dict

        indice_list = get_filtered_suggestions(clean_features(features), indice_list_raw, number_of_suggestions)

        result = []
        n = 1
        
        for i in indice_list:
            wine_name = self.wine_review_min_count['Name'][i]
            wine_descriptors = self.wine_review_min_count['normalized_descriptors'][i]
            string_one = " ".join(['Suggestion', str(n), ':', wine_name]) # , 'with a cosine distance of', "{:.3f}".format(d)
            string_two = " ".join(['This wine has the following descriptors:'] + list(wine_descriptors))
            string_three = " "
            string = " ".join([string_one, string_two, string_three])
            result.append(string)
            n+=1
        return result
    
    def get_countries_min_count(self, min_count = 0):
        return list(self.wine_review_min_count['Country'].value_counts().loc[lambda x: x > min_count].reset_index()['Country'])
        # country_to_variety_dict = self.wine_review_min_count[["Country", "Variety"]].groupby('Country')["Variety"].unique().to_dict()
        # return country_to_variety_dict

    
    def get_varieties_min_count(self, min_count = 0):
        return list(self.wine_review_min_count['Variety'].value_counts().loc[lambda x: x > min_count].reset_index()['Variety'])
        # variety_to_country_dict = self.wine_review_min_count[["Country", "Variety"]].groupby('Variety')["Country"].unique().to_dict()
        # return variety_to_country_dict
    

class Model():

    # database = Database()
    # w2v = database.get_Word2Vec()
    # kv = database.get_KeyedVectors()
    # model_knn = database.get_model_knn()
    X = database.get_X()

    def __init__(self):
        pass

    # def get_model_model_knn(self):
    #     return self.model_knn
    
    def get_model_X(self):
        return self.X

    def get_dict_of_tfidf_weightings(self):
        dict_of_tfidf_weightings = dict(zip(self.get_model_X().get_feature_names_out(), self.get_model_X().idf_))
        return dict_of_tfidf_weightings
    
    # def get_model_Word2Vec(self):
    #     return self.w2v

    # def get_model_KeyedVectors(self):
    #     return self.kv

    # def get_KeyedVectors_words(self):
    #     return self.kv.key_to_index
    



