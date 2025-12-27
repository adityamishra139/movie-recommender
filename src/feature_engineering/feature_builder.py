import pandas as pd
import ast

class FeatureBuilder:
    def __init__(self):
        pass

    def convert(self, obj):
        """Extracts names from JSON list string."""
        L = []
        try:
            for i in ast.literal_eval(obj):
                L.append(i['name'])
        except (ValueError, TypeError):
            pass
        return L

    def convert3(self, obj):
        """Extracts top 3 names from JSON list string."""
        L = []
        try:
            counter = 0
            for i in ast.literal_eval(obj):
                if counter != 3:
                    L.append(i['name'])
                    counter += 1
                else:
                    break
        except (ValueError, TypeError):
            pass
        return L

    def fetch_director(self, obj):
        """Extracts director name from crew JSON list string."""
        L = []
        try:
            for i in ast.literal_eval(obj):
                if i['job'] == 'Director':
                    L.append(i['name'])
                    break
        except (ValueError, TypeError):
            pass
        return L

    def collapse(self, L):
        """Removes spaces from list of strings."""
        L1 = []
        for i in L:
            L1.append(i.replace(" ", ""))
        return L1

    def create_soup(self, x):
        """Combines keywords, cast, director, and genres into a single string."""
        return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + ' '.join(x['director']) + ' ' + ' '.join(x['genres'])

    def build_features(self, df):
        """
        Builds features from the raw dataframe.
        Args:
            df (pd.DataFrame): The raw dataframe containing movies and credits.
        Returns:
            pd.DataFrame: The processed dataframe with new features.
        """
        # Create a copy to avoid SettingWithCopyWarning
        df = df.copy()

        # Parse JSON columns
        if 'genres' in df.columns:
            df['genres'] = df['genres'].apply(self.convert)
        if 'keywords' in df.columns:
            df['keywords'] = df['keywords'].apply(self.convert)
        if 'cast' in df.columns:
            df['cast'] = df['cast'].apply(self.convert3)
        if 'crew' in df.columns:
            df['crew'] = df['crew'].apply(self.fetch_director)
            
        # Rename crew to director for clarity since we extracted director
        # But commonly we just keep the extracted list in a 'director' column
        df['director'] = df['crew']

        # Select relevant columns
        features = ['genres', 'keywords', 'cast', 'director']
        
        # Collapse spaces
        for feature in features:
            df[feature] = df[feature].apply(self.collapse)

        # Rename columns from merge
        if 'title_x' in df.columns:
            df.rename(columns={'title_x': 'title'}, inplace=True)
        if 'title_y' in df.columns:
            df.drop(columns=['title_y'], inplace=True)

        # Create soup
        # Ensure we don't have NaNs that break join
        for feature in features:
             df[feature] = df[feature].apply(lambda x: x if isinstance(x, list) else [])

        df['soup'] = df.apply(self.create_soup, axis=1)
        
        return df
