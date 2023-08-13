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

from gensim.models import KeyedVectors

base_location = r'/Users/saruul/Desktop/c4h/Wine/wine_demo/data/'

i = 0
for file in os.listdir(base_location):
    if file.endswith ('.csv'):
        file_location = base_location + '/' + str(file)
        print(file)
        if i==0:
            wine_dataframe = pd.read_csv(file_location, encoding='latin-1')
            i+=1
        else:
            df_to_append = pd.read_csv(file_location, encoding='latin-1', low_memory=False)
            wine_dataframe = pd.concat([wine_dataframe, df_to_append], axis=0)

wine_dataframe.drop_duplicates(subset=['Name'], inplace=True)

reviews_list = list(wine_dataframe['Description'])
reviews_list = [str(r) for r in reviews_list]
full_corpus = ' '.join(reviews_list)
sentences_tokenized = sent_tokenize(full_corpus)


stop_words = set(stopwords.words('english')) 

punctuation_table = str.maketrans({key: None for key in string.punctuation})
sno = SnowballStemmer('english')

def normalize_text(raw_text):
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
        return normalized_sentence
    except:
        return ''

# sentence_sample = sentences_tokenized[:10]
normalized_sentences = []
for s in sentences_tokenized:
    normalized_text = normalize_text(s)
    normalized_sentences.append(normalized_text)

phrases = Phrases(normalized_sentences)
phrases = Phrases(phrases[normalized_sentences])

ngrams = Phraser(phrases)

phrased_sentences = []
for sent in normalized_sentences:
    phrased_sentence = ngrams[sent]
    phrased_sentences.append(phrased_sentence)

full_list_words = [item for sublist in phrased_sentences for item in sublist]

word_counts = Counter(full_list_words)
sorted_counts = OrderedDict(word_counts.most_common(5000))
counter_df = pd.DataFrame.from_dict(sorted_counts, orient='index')
# top_5000_words = counter_df.head(5000)
counter_df.to_csv('top_5000_descriptors.csv')

descriptor_mapping = pd.read_csv(base_location + 'descriptor/descriptor_mapping.csv').set_index('raw descriptor')

def return_mapped_descriptor(word):
    if word in list(descriptor_mapping.index):
        normalized_word = descriptor_mapping['level_3'][word]
        return normalized_word
    else:
        return word

normalized_sentences = []
for sent in phrased_sentences:
    normalized_sentence = []
    for word in sent:
        normalized_word = return_mapped_descriptor(word)
        normalized_sentence += [str(normalized_word)]
    normalized_sentences += [normalized_sentence]

wine_word2vec_model = Word2Vec(normalized_sentences, vector_size=300, min_count=5, epochs=15)

wine_word2vec_model.save(base_location + 'bin/wine_word2vec_model.bin')

wine_word2vec_model.wv.most_similar(positive='peach', topn=10)

wine_reviews = list(wine_dataframe['Description'])

def return_descriptor_from_mapping(word):
    if word in list(descriptor_mapping.index):
        descriptor_to_return = descriptor_mapping['level_3'][word]
        return descriptor_to_return

descriptorized_reviews = []
for review in wine_reviews:
    normalized_review = normalize_text(review)
    phrased_review = ngrams[normalized_review]
    descriptors_only = [return_descriptor_from_mapping(word) for word in phrased_review]
    no_nones = [str(d) for d in descriptors_only if d is not None]
    descriptorized_review = ' '.join(no_nones)
    descriptorized_reviews.append(descriptorized_review)

vectorizer = TfidfVectorizer()
X = vectorizer.fit(descriptorized_reviews)

dict_of_tfidf_weightings = dict(zip(X.get_feature_names(), X.idf_))

wine_review_vectors = []
for d in descriptorized_reviews:
    descriptor_count = 0
    weighted_review_terms = []
    terms = d.split(' ')
    for term in terms:
        if (term in dict_of_tfidf_weightings.keys()) & (wine_word2vec_model.wv.__contains__(term)):
            tfidf_weighting = dict_of_tfidf_weightings[term]
            word_vector = wine_word2vec_model.wv.get_vector(term).reshape(1, 300)
            # print(str(term) + ": " + str(word_vector))
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

wine_dataframe.reset_index(inplace=True)

most_common_reds = ['Pinot Noir', 'Cabernet Sauvignon', 'Bordeaux-style Red Blend', 'Syrah', 'Merlot',
                         'Sangiovese', 'Zinfandel', 'Tempranillo', 'Nebbiolo', 'Portuguese Red', 'Malbec',
                         'Rhone-style Red Blend', 'Cabernet Franc', 'Gamay']

# first, let's eliminate any review with fewer than 5 descriptors from our dataset
wine_reviews_mincount = wine_dataframe.loc[wine_dataframe['descriptor_count'] > 5]
wine_reviews_mincount.reset_index(inplace=True)

variety_mapping = {'Shiraz': 'Syrah', 'Pinot Gris': 'Pinot Grigio', 'Pinot Grigio/Gris': 'Pinot Grigio', 
                   'Garnacha, Grenache': 'Grenache', 'Garnacha': 'Grenache', 'CarmenÃ¨re': 'Carmenere',
                    'GrÃ¼ner Veltliner': 'Gruner Veltliner', 'TorrontÃ©s': 'Torrontes', 
                   'RhÃ´ne-style Red Blend': 'Rhone-style Red Blend', 'AlbariÃ±o': 'Albarino',
                  'GewÃ¼rztraminer': 'Gewurztraminer', 'RhÃ´ne-style White Blend': 'Rhone-style White Blend'}

def consolidate_varieties(variety_name):
    if variety_name in variety_mapping:
        return variety_mapping[variety_name]
    else:
        return variety_name

wine_reviews_clean = wine_reviews_mincount.copy()
wine_reviews_clean['Variety'] = wine_reviews_clean['Variety'].apply(consolidate_varieties)

def subset_wine_vectors(list_of_varieties):
    wine_variety_vectors = []
    for v in list_of_varieties:
        one_var_only = wine_reviews_clean.loc[wine_reviews_clean['Variety'] == v]
        review_arrays = one_var_only['review_vector'].apply(lambda x: x[0])
        average_variety_vec = np.average(review_arrays)
        wine_variety_vector = [v, average_variety_vec]
        wine_variety_vectors.append(wine_variety_vector)
    return wine_variety_vectors

def pca_wine_variety(list_of_varieties):
    wine_var_vectors = subset_wine_vectors(list_of_varieties)
    pca = PCA(n_components=2)
    pca.fit([w[1] for w in wine_var_vectors])  
    pca_dataset = pca.fit_transform([w[1] for w in wine_var_vectors])
    pca_dataframe = pd.DataFrame(pca_dataset, columns=['pca_1', 'pca_2'])
    pca_dataframe.index = [w[0] for w in wine_var_vectors]
    # print(pca_dataframe)
    return pca_dataframe

# pca_wine_variety(['Pinot Noir'])
pca_wine_variety(['Pinot Noir', 'Cabernet Sauvignon', 'Bordeaux-style Red Blend','Syrah',
                 'Merlot', 'Sangiovese', 'Zinfandel', 'Tempranillo', 'Nebbiolo',
                 'Portuguese Red', 'Malbec', 'Rhone-style Red Blend', 'Cabernet Franc',
                 'Gamay'])

# most_common_whites = ['Chardonnay', 'Sauvignon Blanc', 'Riesling', 'Pinot Grigio', 
#                         'Gruner Veltliner', 'Viognier', 'Chenin Blanc', 'Albarino', 'Pinot Blanc', 'Verdejo',
#                         'Torrontes', 'Vermentino', 'Melon', 'Gewurztraminer', 'Rhone-style White Blend']
most_common_whites = ['Chardonnay', 'Sauvignon Blanc', 'Riesling', 'Pinot Grigio', 
                        'Gruner Veltliner', 'Viognier']

pca_w_dataframe = pca_wine_variety(most_common_whites)

most_common_reds = ['Pinot Noir', 'Cabernet Sauvignon', 'Bordeaux-style Red Blend', 'Syrah', 'Merlot',
                         'Sangiovese', 'Zinfandel', 'Tempranillo', 'Nebbiolo', 'Portuguese Red', 'Malbec',
                         'Rhone-style Red Blend', 'Cabernet Franc', 'Gamay']

pca_r_dataframe = pca_wine_variety(most_common_reds)

for i, txt in enumerate(pca_r_dataframe.index):
    plt.annotate(txt, (list(pca_r_dataframe['pca_1'])[i]+0.2, list(pca_r_dataframe['pca_2'])[i]-0.1), fontsize=12)

input_vectors = list(wine_reviews_mincount['review_vector'])
input_vectors_listed = [a.tolist() for a in input_vectors]
input_vectors_listed = [a[0] for a in input_vectors_listed]

knn = NearestNeighbors(n_neighbors=10, algorithm= 'brute', metric='cosine')
model_knn = knn.fit(input_vectors_listed)

# # name_test = "Point & Line 2016 John Sebastiano Vineyard Reserve Pinot Noir (Sta. Rita Hills)"
# name_test = "Aspen Peak 2019 Viognier (Grand Valley)"
# wine_test_vector = wine_reviews_mincount.loc[wine_reviews_mincount['Name'] == name_test]['review_vector'].tolist()[0]
# distance, indice = model_knn.kneighbors(wine_test_vector, n_neighbors=9)
# distance_list = distance[0].tolist()[1:]
# indice_list = indice[0].tolist()[1:]

# # main_wine = wine_reviews_mincount.loc[wine_reviews_mincount['Name'] == name_test]

# # print('Wine to match:', name_test)
# # print('The original wine has the following descriptors:', list(main_wine['normalized_descriptors'])[0])
# # print('_________')

# n = 1
# for d, i in zip(distance_list, indice_list):
#     wine_name = wine_reviews_mincount['Name'][i]
#     wine_descriptors = wine_reviews_mincount['normalized_descriptors'][i]
#     # print('Suggestion', str(n), ':', wine_name, 'with a cosine distance of', "{:.3f}".format(d))
#     # print('This wine has the following descriptors:', wine_descriptors)
#     # print('')
#     n+=1

def descriptors_to_best_match_wines(list_of_descriptors, number_of_suggestions=10):
    weighted_review_terms = []
    for term in list_of_descriptors:
        if term not in dict_of_tfidf_weightings:
            if term not in descriptor_mapping.index:
                print('choose a different descriptor from', term)
                continue
            else:
                term = descriptor_mapping['normalized'][term]
        tfidf_weighting = dict_of_tfidf_weightings[term]
        word_vector = wine_word2vec_model.wv.get_vector(term).reshape(1, 300)
        weighted_word_vector = tfidf_weighting * word_vector
        weighted_review_terms.append(weighted_word_vector)
    review_vector = sum(weighted_review_terms)
    
    distance, indice = model_knn.kneighbors(review_vector, n_neighbors=number_of_suggestions+1)
    distance_list = distance[0].tolist()[1:]
    indice_list = indice[0].tolist()[1:]

    result = ""
    n = 1
    for d, i in zip(distance_list, indice_list):
        wine_name = wine_reviews_mincount['Name'][i]
        wine_descriptors = wine_reviews_mincount['normalized_descriptors'][i]
        print("wine_descriptors")
        print(wine_descriptors)
        string_one = "".join(['Suggestion', str(n), ':', wine_name, 'with a cosine distance of', "{:.3f}".format(d)])
        string_two = "".join(['This wine has the following descriptors:'] + list(wine_descriptors))
        string_three = ""
        result += "\n".join([string_one, string_two, string_three])
        n+=1
    return result

        
    
# descriptors = ['complex', 'high_acid', 'fresh', 'grass', 'lime']
# descriptors_to_best_match_wines(list_of_descriptors=descriptors, number_of_suggestions=5)




level_3_words = list(descriptor_mapping.level_3.unique())


# Load the pre-trained word embeddings model
model = KeyedVectors.load_word2vec_format(base_location + 'misc/glove_6B_50d.txt', binary=False, no_header=True)
available_words = model.key_to_index

def find_closest_word(word, word_list):
    # Check if the word exists in the model's vocabulary
    if word not in model.key_to_index:
        return "Word not found in the vocabulary."

    # Calculate the similarity scores between the given word and the words in the list
    similarity_scores = {w: model.similarity(word, w) for w in word_list if w in available_words}
    # Find the word with the highest similarity score
    closest_word = max(similarity_scores, key=similarity_scores.get)

    return closest_word

def find_closest_wine_descriptors(word_list, wine_descriptor_list = level_3_words):
    return [find_closest_word(word, wine_descriptor_list) for word in word_list]

def run_recommender(descriptors_trial):
    # Words Saruul really wanted to use
    # descriptors_trial = ['aesthetic', 'acidic', 'lively', 'tasty', 'champagne', 'victorian']

    wine_descriptors = find_closest_wine_descriptors(descriptors_trial)

    result = descriptors_to_best_match_wines(list_of_descriptors=wine_descriptors, number_of_suggestions=5)
    return result
