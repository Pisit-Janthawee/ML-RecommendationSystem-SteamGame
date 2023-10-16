
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.feature_extraction.text import CountVectorizer

from text_normalization import TextNormalizer


class Pipeline():
    def __init__(self, df):
        self.df = df.copy()
        self.steam_content_cols = ['steam_appid', 'name',
                                   'developer', 'genres',
                                   'steamspy_tags',
                                   'short_description',
                                   'header_image']

    def text_normalization(self):
        tn = TextNormalizer()
        for col in self.steam_content_cols:
            if col in ['name',
                       'developer', 'genres',
                       'steamspy_tags',
                       'short_description']:
                self.df[col] = self.df[col].apply(tn.text_normalization)
        self.df['short_description'] = self.df['short_description'].apply(
            tn.remove_html)

    def vectorizer(self):
        vectorizer = CountVectorizer(
            lowercase=True, stop_words='english')
        matrix = vectorizer.fit_transform(self.df['all'])
        return matrix

    def dropRows(self):
        self.df.dropna(subset=['developer', 'publisher'], inplace=True)
        self.df = self.df.drop(
            self.df[self.df['total_ratings'] <= 10].index).reset_index(drop=True)
        self.df = self.df.drop(
            self.df[self.df['english'] == 0].index).reset_index(drop=True)

    def similarity(self):
        similarity_vector = cosine_similarity(
            self.vectorizer(), self.vectorizer())
        return similarity_vector

    def feature_engineering(self):
        self.df['total_ratings'] = self.df['positive_ratings'] - \
            self.df['negative_ratings']

    def feature_extraction(self, ):
        self.df['genres&tags'] = self.df['genres'] + \
            ' ' + self.df['steamspy_tags']

    def feature_combining(self):
        tn = TextNormalizer()
        self.df['genres&tags'] = self.df['genres'] + \
            ' ' + self.df['steamspy_tags']

        self.df['genres&tags'] = self.df['genres&tags'].apply(
            tn.del_duplicate_tag)
        self.df['all'] = self.df['name'] + ' ' + \
            self.df['developer'] + ' ' + self.df['genres&tags']

    def preprocessing(self):
        self.feature_engineering()
        self.feature_extraction()
        self.dropRows()
        self.feature_combining()
        return self.similarity()
