import pandas as pd
import numpy as np

df_credits=pd.read_csv("tmdb_5000_credits.csv")
df_movies=pd.read_csv("tmdb_5000_movies.csv")

df= df_movies.merge(df_credits,on='title')
#decide the columns that will matter and remove the ones that don't
df[['movie_id','title','overview','genres','keywords','cast','crew']]

#check for missing data, remove duplicates
print(df.isnull().sum())
print(df.duplicated().sum())

#change format of each dictionary in genre
import ast
def convert(obj):
    k=[]
    for i in ast.literal_eval(obj):
        k.append(i['name'])
    return k
def convert_3(obj):
    k=[]
    for i in ast.literal_eval(obj):
        k.append(i['name'].replace(" ",""))
    return k[0:3]

df['genres']=df['genres'].apply(convert)
print(df["genres"])
df['keywords']=df['keywords'].apply(convert)
df['cast']=df['cast'].apply(convert_3)
print(df['cast'])

def director(obj):
    k=[]
    for i in ast.literal_eval(obj):
        if i['job']=="Director":
            k.append(i['name'].replace(" ",""))
            break
    return k
df['crew']=df['crew'].apply(director)
print(df['crew'])



#split the words in overview using comma
df['overview']=df['overview'].apply(lambda x:str(x))
df['overview']=df['overview'].apply(lambda x:x.split())

df['tags']=df['overview'] + df['genres']+df['keywords']+df['cast']+df['crew']

#create a new dataframe

new_df=df[['movie_id','title','tags']]
new_df["tags"]=new_df['tags'].apply(lambda x:" ".join(x))
new_df["tags"]=new_df['tags'].apply(lambda x:x.lower())

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words='english')

#convert to numpy array
vectors=cv.fit_transform(new_df['tags']).toarray()

#gets the vector of most frequently used words among those 5000 words
cv.get_feature_names_out()

#apply stemming to prevent similar words
import nltk
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

def stem(text):
    y=[]

    for  i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

new_df['tags']=new_df['tags'].apply(stem)

from sklearn.metrics.pairwise import cosine_similarity


similarity=cosine_similarity(vectors)

def recommend(movie):
    movie_index=new_df[new_df['title']==movie].index[0]
    dist=similarity[movie_index]
    movies=sorted(list(enumerate(dist)),reverse=True,key=lambda x:x[1])[1:5]

    for i in movies:
        print(new_df.iloc[i[0]].title)


recommend(movie)

