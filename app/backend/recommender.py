import os
import string
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

    def run_recommender(self, descriptors_trial_raw):
        print("run_recommender")
        # Words Saruul really wanted to use
        # descriptors_trial = ['aesthetic', 'acidic', 'lively', 'tasty', 'champagne', 'victorian']
        def clean_text(text):
            return text.lower()
        
        descriptors_trial = [clean_text(text) for text in descriptors_trial_raw]

        wine_descriptors = self.find_closest_wine_descriptors(descriptors_trial)

        result = self.descriptors_to_best_match_wines(list_of_descriptors=wine_descriptors, number_of_suggestions=5)
        return result

    def find_closest_wine_descriptors(self, word_list, wine_descriptor_list = database.get_level_3_mapping()):
        print("find_closest_wine_descriptors")
        return [self.find_closest_word(word, wine_descriptor_list) for word in word_list]
    
    def find_closest_word(self, word, word_list):
        print("find_closest_word")
        model = self.kv_model
        # Check if the word exists in the model's vocabulary
        if word not in model.key_to_index:
            return "Word not found in the vocabulary."

        # Calculate the similarity scores between the given word and the words in the list
        similarity_scores = {w: model.similarity(word, w) for w in word_list if w in self.kv_words} 
        # Find the word with the highest similarity score
        closest_word = max(similarity_scores, key=similarity_scores.get)

        return closest_word
    
    def descriptors_to_best_match_wines(self, list_of_descriptors, number_of_suggestions=10):
        print("descriptors_to_best_match_wines")
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
        
        distance, indice = self.model_knn.kneighbors(review_vector, n_neighbors=number_of_suggestions+1)
        distance_list = distance[0].tolist()[1:]
        indice_list = indice[0].tolist()[1:]

        result = []
        n = 1
        
        for d, i in zip(distance_list, indice_list):
            wine_name = self.wine_review_min_count['Name'][i]
            wine_descriptors = self.wine_review_min_count['normalized_descriptors'][i]
            string_one = " ".join(['Suggestion', str(n), ':', wine_name]) # , 'with a cosine distance of', "{:.3f}".format(d)
            string_two = " ".join(['This wine has the following descriptors:'] + list(wine_descriptors))
            string_three = " "
            string = " ".join([string_one, string_two, string_three])
            result.append(string)
            n+=1
        return result
    

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
    



