import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
import warnings; warnings.simplefilter('ignore')
from nltk.corpus import stopwords
nltk.download('stopwords')

class ProductRecommendation:
    def __init__(self):
        self.model = SentenceTransformer('paraphrase-distilroberta-base-v1')
        self.df = None
        self.des_embeddings = None

    def preprocess_data(self, df):
        # Remove NaN values
        df['brand'].fillna(value='', inplace=True)

        # Removing duplicates
        df['name2'] = df['product_name'].str.lower().str.replace(" ", "")
        df['brand2'] = df['brand'].str.lower().str.replace(" ", '')
        df = df.drop_duplicates(subset=['name2', 'brand2', 'retail_price'], keep='first')

        # Product category
        df['product_category_tree'] = df['product_category_tree'].str.replace(">>", ",")

        # Remove stopwords
        def remove_stopwords(text):
            stop_words = set(stopwords.words('english'))
            words = text.split()
            filtered_words = [word for word in words if word.lower() not in stop_words]
            return ' '.join(filtered_words)

        df['description'] = df['description'].astype(str).apply(remove_stopwords)

        def extract_keys(row):
            if isinstance(row, list):  # Check if the row is a list
                return [item["key"] for item in row if isinstance(item, dict) and "key" in item]
            else:
                return []

        # Apply the function to create a new column 'key_list'
        df['key_list'] = df['product_specifications'].apply(extract_keys)

        # Combine text columns into 'soup'
        text_columns = ['product_category_tree', 'description', 'product_name', 'brand', 'key_list']
        df['soup'] = df[text_columns].astype(str).apply(' '.join, axis=1)

        self.df = df
        self.df['index'] = range(0, df.shape[0])
        self.df = self.df.reset_index(drop=True)

        descriptions = df['soup'].tolist()
        self.des_embeddings = []
        for i, des in enumerate(descriptions):
            self.des_embeddings.append(self.model.encode(des))

    def fit(self, data_df):
        self.preprocess_data(data_df)

    def recommend(self, query):
        if self.des_embeddings is None:
            raise Exception("Data is not yet preprocessed. Call fit method first.")

        # Compute cosine-similarities with all embeddings
        query_embedd = self.model.encode(query)
        cosine_scores = util.pytorch_cos_sim(query_embedd, self.des_embeddings)
        top5_matches = torch.argsort(cosine_scores, dim=-1, descending=True).tolist()[0][1:8]
        return top5_matches

    def predict(self, product_name):
        if self.df is None:
            raise Exception("Data is not yet preprocessed. Call fit method first.")

        query_show_des = self.df.loc[self.df['product_name'] == product_name]['soup'].to_list()[0]
        recommended_results = self.recommend(query_show_des)

        recommendations = []
        for index in recommended_results:
            recommendations.append(self.df.iloc[index, [3, 6, 7, 13]])

        return recommendations

