import pandas as pd
import matplotlib.pyplot as plt

import requests
from PIL import Image
from io import BytesIO
import time


class RecommendationSystem:
    def __init__(self, df, similarity_vectorizer):
        self.df = df
        self.similarity_vectorizer = similarity_vectorizer

    def nearest_name(self, name: str):
        result = self.df[self.df['name'].str.startswith(name[:-1])]
        close = self.get_close_matches(name)
        if len(result) != 0:
            return result.index[0]
        elif close:
            return close[0]
        else:
            return None

    def get_close_matches(self, name: str):
        result = self.df.loc[self.df['name'].apply(
            lambda x: x.lower()).str.contains(name.lower())]

        if len(result) != 0:
            return result
        return []

    def recommendations(self, name: str, limit=10):

        ind = self.nearest_name(name=name)
        cos_score = list(enumerate(self.similarity_vectorizer[ind]))

        cos_score = sorted(cos_score, key=lambda x: x[1], reverse=True)
        cos_score = cos_score[0:limit]
        ten_ind = [i[0] for i in cos_score]
        df = self.df[['name', 'developer', 'genres',
                      'steamspy_tags',
                      'header_image']].iloc[ten_ind]
        score = [i[1] for i in cos_score]
        score = [round(num, 4)*100 for num in score]

        df = pd.DataFrame({'name': df['name'],
                           'developer': df['developer'],
                           'genres': df['genres'],
                           'steamspy_tags': df['steamspy_tags'],
                           'header_image': df['header_image'],
                           'match_score': score
                           })

        fig, axs = plt.subplots(2, 5, figsize=(20, 8))
        axs = axs.flatten()

        for index, row in df.iterrows():
            response = requests.get(row['header_image'])

            img = Image.open(BytesIO(response.content))
            # to get the image, we have to wait module taking time to retrieve the images from the URLs.
            time.sleep(0.5)
            if index < len(axs):
                axs[index].imshow(img)
                axs[index].set_title(r'$\bf{' + df['name'].loc[index] + '}$' +
                                     f"\nMatch score: {df['match_score'].loc[index]} %\nDeveloper: {df['developer'].loc[index]}\nGenres: {df['genres'].loc[index]}\nTags: {df['steamspy_tags'].loc[index]}")

        plt.tight_layout()
        plt.show()
        image_path = "output.png"
        plt.savefig(image_path, format="png")
        average_match_score = df['match_score'].mean()
        print(f'Average Match Score: {average_match_score:.4f}')
        return df[['name', 'developer', 'genres',
                   'steamspy_tags', 'match_score']], image_path
