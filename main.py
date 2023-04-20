#Work flow

#1. Collecting data
#2. Data Preprocessing
#3. Feature Extraction
#4. cosine similarity algorithm

#import dependencies

import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies_data = pd.read_csv("C:/Users/shris/Downloads/movies.csv")

#print dimensions of the dataset
#print(movies_data.shape)
#it has 24 columns, we will use most relevant faetures only
#genre, keyword, ...


#print features of the data set
#print(movies_data.columns)


#select features that are of importance while recommendation
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']


#fill null values with empty string
for feature in selected_features :
    movies_data[feature] = movies_data[feature].fillna('')

#combining all selected features
combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] +  ' ' + movies_data['cast'] + ' ' + movies_data['director']

#converting text data to feature vectors
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
#print(feature_vectors)


#cosine similarity
similarity = cosine_similarity(feature_vectors)
#this will return a mXm matrix which a similarity of each value in feature_vector with every other value


movie_name = input('ENTER YOUR FAVOURITE MOVIE NAME')

#list of all movies
list_of_all_movies = movies_data['title'].to_list()
#print(list_of_all_movies)

#finding the close match for the movie
find_closematch = difflib.get_close_matches(movie_name, list_of_all_movies)
#print(find_closematch)

close_match = find_closematch[0]
print(close_match)

index_of_movie = movies_data[movies_data.title == close_match]['index'].values[0]
print(index_of_movie)

similarity_score = list(enumerate(similarity[index_of_movie]))

# sorting the movies based on their similarity score
sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True)
#print(sorted_similar_movies)


# print the name of similar movies based on the index
print('Movies suggested for you : \n')

i = 1
for movie in sorted_similar_movies:
  index = movie[0]
  title_from_index = movies_data[movies_data.index==index]['title'].values[0]
  if (i<11):
    print(i, '.',title_from_index)
    i+=1




