class Recommender:
    def __init__(self, df, similarity_matrix):
        """
        Args:
            df (pd.DataFrame): DataFrame with movie titles and indices
            similarity_matrix (np.ndarray): Cosine similarity matrix
        """
        self.df = df.reset_index(drop=True)
        self.similarity = similarity_matrix

    def recommend(self, movie_name, top_n=5):
        """
        Recommends similar movies based on cosine similarity.
        Args:
            movie_name (str): Name of the movie
            top_n (int): Number of recommendations
        Returns:
            list: Recommended movie titles
        """
        # Find movie index
        if movie_name not in self.df['title'].values:
            # Fallback or strict error? 
            # Trying to handle potential case sensitivity or extra spaces might be good, 
            # but for now strict matching as per user code.
            raise ValueError(f"Movie '{movie_name}' not found in dataset.")

        movie_index = self.df[self.df['title'] == movie_name].index[0]

        # Fetch similarity scores
        distances = self.similarity[movie_index]

        # Sort movies by similarity score
        movie_list = sorted(
            list(enumerate(distances)),
            reverse=True,
            key=lambda x: x[1]
        )

        # Exclude the movie itself and return top N
        recommendations = [
            self.df.iloc[i[0]].title
            for i in movie_list[1: top_n + 1]
        ]

        return recommendations

    def search_movies(self, query):
        """
        Searches for movies matching the query (case-insensitive substring).
        Args:
            query (str): Search query
        Returns:
            list: List of matching movie titles
        """
        query = query.lower().strip()
        titles = self.df['title'].tolist()
        
        # 1. Exact match (case-insensitive)
        exact_matches = [t for t in titles if t.lower() == query]
        if exact_matches:
            return exact_matches
            
        # 2. Substring match
        partial_matches = [t for t in titles if query in t.lower()]
        
        return partial_matches
