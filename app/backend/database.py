import os
import string
from operator import itemgetter
import time
from functools import lru_cache

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

base_location = os.getcwd() + r'/app/data/'

def get_wine_dataframe():
    print("get_wine_dataframe")
    start_time = time.time()
    
    i = 0
    for file in os.listdir(base_location):
        if file.endswith ('.csv'):
            file_location = base_location + '/' + str(file)
            # print(file)
            if i==0:
                wine_dataframe = pd.read_csv(file_location, encoding='latin-1')
                i+=1
            else:
                df_to_append = pd.read_csv(file_location, encoding='latin-1', low_memory=False)
                wine_dataframe = pd.concat([wine_dataframe, df_to_append], axis=0)
    print("--- %s seconds ---" % (time.time() - start_time))
    return wine_dataframe

import pickle
class Database:
    print("descriptor_mapping")
    start_time = time.time()
    descriptor_mapping = pd.read_csv(base_location + 'descriptor/descriptor_mapping.csv').set_index('raw descriptor').reset_index()
    print("--- %s seconds ---" % (time.time() - start_time))

    print("wine_dataframe")
    start_time = time.time()
    wine_dataframe = get_wine_dataframe()
    print("--- %s seconds ---" % (time.time() - start_time))

    print("w2v")
    start_time = time.time()
    w2v = gensim.models.word2vec.Word2Vec.load(base_location + 'bin/wine_word2vec_model.bin')
    print("--- %s seconds ---" % (time.time() - start_time))

    print("kv")
    start_time = time.time()
    kv = KeyedVectors.load_word2vec_format(base_location + 'misc/glove_6B_50d.txt', binary=False, no_header=True)
    print("--- %s seconds ---" % (time.time() - start_time))

    print("knn")
    start_time = time.time()
    knn = NearestNeighbors(n_neighbors=10, algorithm= 'brute', metric='cosine')
    print("--- %s seconds ---" % (time.time() - start_time))

    print("vectorizer")
    start_time = time.time()
    vectorizer = TfidfVectorizer()
    print("--- %s seconds ---" % (time.time() - start_time))

    
    def __init__(self):
        # self.wine_dataframe = self.add_wine_review_vectors(get_wine_dataframe())
        # pass
        print("init db")
        start_time = time.time()

        self.cache_directory = "cache"  # Directory to store cached data
        os.makedirs(self.cache_directory, exist_ok=True)
        self.descriptorized_review = self.load_cached_model("descriptorized_reviews")

        if self.descriptorized_review is None:
            self.descriptorized_review = self.get_descriptorized_reviews()
            self.save_cached_model("descriptorized_reviews", self.descriptorized_review)

        print("--- %s seconds ---" % (time.time() - start_time))
    
    def get_cache_filename(self, cache_key):
        print("get_cache_filename")
        start_time = time.time()
        # Generate a cache filename based on the cache key
        result = os.path.join(self.cache_directory, f"{cache_key}.pkl")
        print("--- %s seconds ---" % (time.time() - start_time))
        return result

    def load_cached_model(self, cache_key):
        print("load_cached_model")
        start_time = time.time()
        # Load the cached model from a file
        cache_filename = self.get_cache_filename(cache_key)
        if os.path.exists(cache_filename):
            with open(cache_filename, "rb") as f:
                print("--- %s seconds ---" % (time.time() - start_time))
                return pickle.load(f)
        print("--- %s seconds ---" % (time.time() - start_time))
        return None
    
    def save_cached_model(self, cache_key, model):
        print("save_cached_model")
        start_time = time.time()
        # Save the model to a file
        cache_filename = self.get_cache_filename(cache_key)
        with open(cache_filename, "wb") as f:
            pickle.dump(model, f)
        print("--- %s seconds ---" % (time.time() - start_time))

    def get_model_knn(self):
        print("get_model_knn")
        start_time = time.time()


        input_vectors_listed = self.get_input_vectors_listed()

        # Create a cache key based on the input vectors
        cache_key = "modelknn" # tuple(input_vectors_listed)

        # Check if the model is already cached
        cached_model = self.load_cached_model(cache_key)
        if cached_model is not None:
            print("Using cached model for", cache_key)
            model_knn = cached_model
        else:
            print("Saving cached model for", cache_key)
            model_knn = self.knn.fit(input_vectors_listed)
            self.save_cached_model(cache_key, model_knn)

        print("--- %s seconds ---" % (time.time() - start_time))
        return model_knn
    

    def get_Word2Vec(self):
        return self.w2v

    def get_KeyedVectors(self):
        return self.kv

    def get_KeyedVectors_words(self):
        return self.kv.key_to_index
    
    # @lru_cache
    # def get_model_knn(self):
    #     start_time = time.time()

    #     input_vectors_listed = self.get_input_vectors_listed()
    #     model_knn = self.knn.fit(input_vectors_listed)

    #     print("--- %s seconds ---" % (time.time() - start_time))
    #     return model_knn
    
    def get_X(self):
        print("get_X")
        start_time = time.time()
        X = self.vectorizer.fit(self.get_descriptorized_reviews())
        print("--- %s seconds ---" % (time.time() - start_time))
        return X

    def get_dict_of_tfidf_weightings(self):
        print("get_dict_of_tfidf_weightings")
        start_time = time.time()
        dict_of_tfidf_weightings = dict(zip(self.get_X().get_feature_names_out(), self.get_X().idf_))
        print("--- %s seconds ---" % (time.time() - start_time))
        return dict_of_tfidf_weightings
        # pass

    def add_wine_review_vectors(self, df) -> pd.DataFrame:
        """
        Returns:
            Dataframe of wine_dataframe + columns 'normalized_descriptors', 'review_vector', & 'descriptor_count'
        """
        print("add_wine_review_vectors")
        start_time = time.time()
        wine_dataframe = df[df['Description'].notna()].reset_index(drop=True)
        wine_review_vectors = []
        descriptorized_reviews = self.get_descriptorized_reviews()

        dict_of_tfidf_weightings = self.get_dict_of_tfidf_weightings()
        wine_word2vec_model = self.get_Word2Vec()

        for d in descriptorized_reviews:
            descriptor_count = 0
            weighted_review_terms = []
            terms = d.split(' ')
            for term in terms:
                if (term in dict_of_tfidf_weightings.keys()) & (wine_word2vec_model.wv.__contains__(term)):
                    tfidf_weighting = dict_of_tfidf_weightings[term]
                    word_vector = wine_word2vec_model.wv.get_vector(term).reshape(1, 300)
                    weighted_word_vector = tfidf_weighting * word_vector
                    weighted_review_terms.append(weighted_word_vector)
                    descriptor_count += 1
                else:
                    continue
            try:
                review_vector = sum(weighted_review_terms)/len(weighted_review_terms)
            except:
                review_vector = []
            vector_and_count = [terms, review_vector, descriptor_count]
            wine_review_vectors.append(vector_and_count)

        wine_dataframe['normalized_descriptors'] = list(map(itemgetter(0), wine_review_vectors))
        wine_dataframe['review_vector'] = list(map(itemgetter(1), wine_review_vectors))
        wine_dataframe['descriptor_count'] = list(map(itemgetter(2), wine_review_vectors))

        # Drop duplicates based on the 'Name' column
        wine_dataframe.drop_duplicates(subset=['Name'], inplace=True)
        print("--- %s seconds ---" % (time.time() - start_time))
        return wine_dataframe

    def get_raw_descriptors_available(self):
        raw_descriptors = list(self.descriptor_mapping["raw descriptor"].unique())
        return raw_descriptors

    def get_level_1_mapping(self):
        level_1_words = list(self.descriptor_mapping.level_1.unique())
        return level_1_words
    
    def get_level_2_mapping(self):
        level_2_words = list(self.descriptor_mapping.level_2.unique())
        return level_2_words
    
    def get_level_3_mapping(self) -> list:
        """
        Returns:
            List of Level 3 (most granular) descriptors of wine
        """
        level_3_words = list(self.descriptor_mapping.level_3.unique())
        return level_3_words
    
    def get_wine_dataframe_main(self):
        print("get_wine_dataframe_main")
        start_time = time.time()
        result = self.add_wine_review_vectors(self.wine_dataframe)
        print("--- %s seconds ---" % (time.time() - start_time))
        return result
    
    def get_wine_review_min_count(self, min_count = 5):
        print("get_wine_review_min_count")
        start_time = time.time()
        wine_dataframe_min = self.add_wine_review_vectors(self.wine_dataframe)
        wine_reviews_mincount = wine_dataframe_min.loc[wine_dataframe_min['descriptor_count'] > min_count]
        
        # Stripping whitespace
        str_cols = ["Variety", "Country"]
        for col in str_cols:
            wine_reviews_mincount[col] = wine_reviews_mincount[col].str.strip()
        
        wine_reviews_mincount = wine_reviews_mincount.reset_index(drop=True)
        print("--- %s seconds ---" % (time.time() - start_time))
        return wine_reviews_mincount
    
    # @cached_property
    def get_input_vectors_listed(self, word_appearance_min_count = 5):
        print("get_input_vectors_listed")
        start_time = time.time()
        wine_reviews_mincount = self.get_wine_review_min_count(word_appearance_min_count)
        input_vectors = list(wine_reviews_mincount['review_vector'])
        input_vectors_listed = [a.tolist() for a in input_vectors]
        input_vectors_listed = [a[0] for a in input_vectors_listed]
        print("--- %s seconds ---" % (time.time() - start_time))
        return input_vectors_listed
        
    def get_descriptorized_reviews(self):
        print("get_descriptorized_reviews")
        if self.descriptorized_review is not None:
            return self.descriptorized_review
        else:
            start_time = time.time()
            descriptorized_reviews = []
            wine_reviews = self.get_wine_reviews()
            ngram = self.get_ngrams()
            for review in wine_reviews:
                normalized_review = normalize_text(review)
                phrased_review = ngram[normalized_review]
                descriptors_only = [self.return_descriptor_from_mapping(word) for word in phrased_review]
                no_nones = [str(d) for d in descriptors_only if d is not None]
                descriptorized_review = ' '.join(no_nones)
                descriptorized_reviews.append(descriptorized_review)
            self.descriptorized_review = descriptorized_reviews
            print("--- %s seconds ---" % (time.time() - start_time))
            return descriptorized_reviews
    
    def get_ngrams(self):
        print("get_ngrams")
        start_time = time.time()
        normalized_sentences = []
        full_corpus = ' '.join(self.get_wine_reviews())
        sentences_tokenized = sent_tokenize(full_corpus)
        for s in sentences_tokenized:
            normalized_text = normalize_text(s)
            normalized_sentences.append(normalized_text)
        phrases = Phrases(normalized_sentences)
        phrases = Phrases(phrases[normalized_sentences])
        ngrams = Phraser(phrases)
        print("--- %s seconds ---" % (time.time() - start_time))
        return ngrams

    def get_wine_reviews(self):
        return list(self.wine_dataframe.loc[self.wine_dataframe['Description'].notna()]['Description'])
    
    def return_descriptor_from_mapping(self, word):
        print("return_descriptor_from_mapping")
        start_time = time.time()
        if word in list(self.descriptor_mapping["raw descriptor"]):
            index = self.descriptor_mapping.index[self.descriptor_mapping['raw descriptor'] == word].tolist()[0]
            descriptor_to_return = self.descriptor_mapping['level_3'][index]
            print("--- %s seconds ---" % (time.time() - start_time))
            return descriptor_to_return


def normalize_text(raw_text):
    print("normalize_text")
    start_time = time.time()
    sno = SnowballStemmer('english')
    punctuation_table = str.maketrans({key: None for key in string.punctuation})
    stop_words = set(stopwords.words('english')) 
    try:
        word_list = word_tokenize(raw_text)
        normalized_sentence = []
        for w in word_list:
            try:
                w = str(w)
                lower_case_word = str.lower(w)
                stemmed_word = sno.stem(lower_case_word)
                no_punctuation = stemmed_word.translate(punctuation_table)
                if len(no_punctuation) > 1 and no_punctuation not in stop_words:
                    normalized_sentence.append(no_punctuation)
            except:
                continue
        print("--- %s seconds ---" % (time.time() - start_time))
        return normalized_sentence
    except:
        print("--- %s seconds ---" % (time.time() - start_time))
        return ''