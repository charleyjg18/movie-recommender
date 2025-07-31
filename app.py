from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

app = Flask(__name__)

class MovieRecommendationSystem:
    """Complete self-contained movie recommendation system"""
    
    def __init__(self):
        self.models = {}
        self.collaborative_data = None
        self.load_models()
        self.setup_collaborative()
    
    def load_models(self):
        """Load basic models"""
        print("🚀 Loading Movie Recommendation Models...")
        
        # Content-based models
        try:
            self.models['movies'] = pickle.load(open('model.pkl', 'rb'))
            self.models['similarity'] = pickle.load(open('similarity.pkl', 'rb'))
            print("✅ Content-based model loaded")
        except Exception as e:
            print(f"❌ Error loading content-based model: {e}")
            self.models['movies'] = None
            self.models['similarity'] = None
    
    def setup_collaborative(self):
        """Setup collaborative filtering from scratch"""
        try:
            # Load ratings data
            if os.path.exists('synthetic_ratings.csv'):
                ratings_df = pd.read_csv('synthetic_ratings.csv')
                print(f"✅ Loaded {len(ratings_df)} ratings")
                
                # Create user-movie matrix
                user_movie_matrix = ratings_df.pivot_table(
                    index='user_id', 
                    columns='movie_id', 
                    values='rating',
                    fill_value=0
                )
                
                # Calculate similarities
                user_matrix = user_movie_matrix.values
                user_similarity = cosine_similarity(user_matrix)
                
                item_matrix = user_movie_matrix.T.values
                item_similarity = cosine_similarity(item_matrix)
                
                # Simple SVD
                binary_matrix = (user_movie_matrix > 0).astype(int)
                svd_model = TruncatedSVD(n_components=min(50, min(binary_matrix.shape)-1), random_state=42)
                user_factors = svd_model.fit_transform(binary_matrix)
                item_factors = svd_model.components_.T
                
                # Store all collaborative data
                self.collaborative_data = {
                    'ratings_df': ratings_df,
                    'user_movie_matrix': user_movie_matrix,
                    'user_similarity': user_similarity,
                    'item_similarity': item_similarity,
                    'user_factors': user_factors,
                    'item_factors': item_factors,
                    'svd_model': svd_model
                }
                
                print("✅ Collaborative filtering setup complete")
            else:
                print("⚠️ No synthetic_ratings.csv found")
                self.collaborative_data = None
                
        except Exception as e:
            print(f"❌ Error setting up collaborative filtering: {e}")
            import traceback
            traceback.print_exc()
            self.collaborative_data = None
    
    def get_available_methods(self):
        """Check which methods are available"""
        return {
            'content_based': (self.models.get('movies') is not None and 
                            self.models.get('similarity') is not None),
            'collaborative': self.collaborative_data is not None,
            'hybrid': (self.models.get('movies') is not None and 
                      self.collaborative_data is not None)
        }
    
    def recommend_content_based(self, movie_title, n_recommendations=5):
        """Content-based recommendations"""
        if not self.get_available_methods()['content_based']:
            return [], []
        
        try:
            movies_df = self.models['movies']
            similarity = self.models['similarity']
            
            movie_indices = movies_df[movies_df['title'] == movie_title].index
            if len(movie_indices) == 0:
                return [], []
            
            index = movie_indices[0]
            distances = sorted(list(enumerate(similarity[index])), 
                             reverse=True, key=lambda x: x[1])
            
            recommended_movies = []
            recommended_posters = []
            
            for i in distances[1:n_recommendations+1]:
                movie_name = movies_df.iloc[i[0]]['title']
                recommended_movies.append(movie_name)
                poster_url = f"https://via.placeholder.com/400x600/667eea/ffffff?text={movie_name.replace(' ', '+')[:20]}"
                recommended_posters.append(poster_url)
            
            return recommended_movies, recommended_posters
            
        except Exception as e:
            print(f"Error in content-based recommendation: {e}")
            return [], []
    
    def predict_user_based(self, user_id, movie_id, k=5):
        """User-based collaborative filtering prediction"""
        if not self.collaborative_data:
            return 3.0
        
        user_movie_matrix = self.collaborative_data['user_movie_matrix']
        user_similarity = self.collaborative_data['user_similarity']
        
        if user_id not in user_movie_matrix.index or movie_id not in user_movie_matrix.columns:
            return 3.0
        
        user_idx = list(user_movie_matrix.index).index(user_id)
        movie_idx = list(user_movie_matrix.columns).index(movie_id)
        
        user_sims = user_similarity[user_idx]
        movie_ratings = user_movie_matrix.iloc[:, movie_idx]
        
        rated_users = movie_ratings > 0
        if rated_users.sum() == 0:
            return 3.0
        
        valid_sims = user_sims[rated_users]
        valid_ratings = movie_ratings[rated_users]
        
        if len(valid_sims) > k:
            top_k_indices = np.argsort(valid_sims)[-k:]
            valid_sims = valid_sims[top_k_indices]
            valid_ratings = valid_ratings.iloc[top_k_indices]
        
        if valid_sims.sum() == 0:
            return 3.0
        
        prediction = np.average(valid_ratings, weights=valid_sims)
        return max(1.0, min(5.0, prediction))
    
    def predict_item_based(self, user_id, movie_id, k=5):
        """Item-based collaborative filtering prediction"""
        if not self.collaborative_data:
            return 3.0
        
        user_movie_matrix = self.collaborative_data['user_movie_matrix']
        item_similarity = self.collaborative_data['item_similarity']
        
        if user_id not in user_movie_matrix.index or movie_id not in user_movie_matrix.columns:
            return 3.0
        
        user_ratings = user_movie_matrix.loc[user_id]
        movie_idx = list(user_movie_matrix.columns).index(movie_id)
        
        rated_movies = user_ratings > 0
        if rated_movies.sum() == 0:
            return 3.0
        
        movie_sims = item_similarity[movie_idx]
        valid_sims = movie_sims[rated_movies]
        valid_ratings = user_ratings[rated_movies]
        
        if len(valid_sims) > k:
            top_k_indices = np.argsort(valid_sims)[-k:]
            valid_sims = valid_sims[top_k_indices]
            valid_ratings = valid_ratings.iloc[top_k_indices]
        
        if valid_sims.sum() == 0:
            return 3.0
        
        prediction = np.average(valid_ratings, weights=valid_sims)
        return max(1.0, min(5.0, prediction))
    
    def predict_svd(self, user_id, movie_id):
        """SVD matrix factorization prediction"""
        if not self.collaborative_data:
            return 3.0
        
        user_movie_matrix = self.collaborative_data['user_movie_matrix']
        user_factors = self.collaborative_data['user_factors']
        item_factors = self.collaborative_data['item_factors']
        
        if user_id not in user_movie_matrix.index or movie_id not in user_movie_matrix.columns:
            return 3.0
        
        user_idx = list(user_movie_matrix.index).index(user_id)
        movie_idx = list(user_movie_matrix.columns).index(movie_id)
        
        prediction = np.dot(user_factors[user_idx], item_factors[movie_idx])
        scaled_prediction = 1 + 4 * max(0, min(1, prediction))
        
        return scaled_prediction
    
    def recommend_collaborative(self, user_id, method='user_based', n_recommendations=5):
        """Collaborative filtering recommendations"""
        if not self.get_available_methods()['collaborative']:
            return [], []
        
        try:
            user_movie_matrix = self.collaborative_data['user_movie_matrix']
            movies_df = self.models['movies']
            
            if user_id not in user_movie_matrix.index:
                return [], []
            
            user_ratings = user_movie_matrix.loc[user_id]
            unrated_movies = user_ratings[user_ratings == 0].index
            
            predictions = []
            
            for movie_id in unrated_movies:
                if method == 'user_based':
                    pred_rating = self.predict_user_based(user_id, movie_id)
                elif method == 'item_based':
                    pred_rating = self.predict_item_based(user_id, movie_id)
                elif method == 'svd':
                    pred_rating = self.predict_svd(user_id, movie_id)
                else:
                    pred_rating = 3.0
                
                if movie_id in movies_df.index:
                    movie_title = movies_df.loc[movie_id, 'title']
                    predictions.append((movie_title, pred_rating))
            
            predictions.sort(key=lambda x: x[1], reverse=True)
            
            recommended_movies = []
            recommended_posters = []
            
            for movie_title, pred_rating in predictions[:n_recommendations]:
                recommended_movies.append(movie_title)
                poster_url = f"https://via.placeholder.com/400x600/74b9ff/ffffff?text={movie_title.replace(' ', '+')[:20]}"
                recommended_posters.append(poster_url)
            
            return recommended_movies, recommended_posters
            
        except Exception as e:
            print(f"Error in collaborative recommendation: {e}")
            import traceback
            traceback.print_exc()
            return [], []
    
    def recommend_hybrid(self, movie_title=None, user_id=None, n_recommendations=5):
        """Hybrid recommendations"""
        if not self.get_available_methods()['hybrid']:
            return [], []
        
        try:
            all_recommendations = {}
            
            # Content-based recommendations
            if movie_title:
                content_movies, _ = self.recommend_content_based(movie_title, n_recommendations * 2)
                for i, movie in enumerate(content_movies):
                    score = (len(content_movies) - i) / len(content_movies)
                    all_recommendations[movie] = 0.6 * score
            
            # Collaborative recommendations
            if user_id:
                collab_movies, _ = self.recommend_collaborative(user_id, n_recommendations=n_recommendations * 2)
                for i, movie in enumerate(collab_movies):
                    score = (len(collab_movies) - i) / len(collab_movies)
                    if movie in all_recommendations:
                        all_recommendations[movie] += 0.4 * score
                    else:
                        all_recommendations[movie] = 0.4 * score
            
            # Sort and create result
            if all_recommendations:
                sorted_recommendations = sorted(all_recommendations.items(), key=lambda x: x[1], reverse=True)
                
                recommended_movies = []
                recommended_posters = []
                
                for movie_name, score in sorted_recommendations[:n_recommendations]:
                    recommended_movies.append(movie_name)
                    poster_url = f"https://via.placeholder.com/400x600/ff6b6b/ffffff?text={movie_name.replace(' ', '+')[:20]}"
                    recommended_posters.append(poster_url)
                
                return recommended_movies, recommended_posters
            
            return [], []
            
        except Exception as e:
            print(f"Error in hybrid recommendation: {e}")
            return [], []
    
    def get_movie_list(self):
        """Get list of available movies"""
        if self.models.get('movies') is not None:
            return self.models['movies']['title'].tolist()
        return []
    
    def get_user_range(self):
        """Get valid user ID range"""
        if self.collaborative_data and 'ratings_df' in self.collaborative_data:
            ratings_df = self.collaborative_data['ratings_df']
            return int(ratings_df['user_id'].min()), int(ratings_df['user_id'].max())
        return 1, 100

# Initialize the recommendation system
rec_system = MovieRecommendationSystem()

def get_template_context():
    """Get template context"""
    return {
        'movie_list': rec_system.get_movie_list(),
        'available_methods': rec_system.get_available_methods(),
        'min_user_id': rec_system.get_user_range()[0],
        'max_user_id': rec_system.get_user_range()[1]
    }

@app.route('/')
def index():
    try:
        context = get_template_context()
        return render_template('index.html', **context)
    except Exception as e:
        print(f"Error in index route: {e}")
        safe_context = {
            'movie_list': [],
            'available_methods': {'content_based': False, 'collaborative': False, 'hybrid': False},
            'min_user_id': 1,
            'max_user_id': 100,
            'error': "Error loading application"
        }
        return render_template('index.html', **safe_context)

@app.route('/recommend', methods=['POST'])
def recommend_movies():
    try:
        movie = request.form.get('movie', '').strip()
        user_id_str = request.form.get('user_id', '').strip()
        method = request.form.get('method', 'content_based')
        collab_method = request.form.get('collab_method', 'user_based')
        
        user_id = None
        if user_id_str:
            try:
                user_id = int(user_id_str)
            except ValueError:
                user_id = None
        
        context = get_template_context()
        
        if not context['available_methods'].get(method, False):
            context['error'] = f"{method.replace('_', ' ').title()} method is not available"
            return render_template('index.html', **context)
        
        # Generate recommendations
        if method == 'content_based':
            if not movie:
                context['error'] = "Please select a movie"
                return render_template('index.html', **context)
            
            recommended_movies, recommended_posters = rec_system.recommend_content_based(movie)
            display_title = f"'{movie}'"
            method_description = "Content-based recommendations"
            
        elif method == 'collaborative':
            if not user_id:
                context['error'] = "Please enter a user ID"
                return render_template('index.html', **context)
            
            recommended_movies, recommended_posters = rec_system.recommend_collaborative(user_id, method=collab_method)
            display_title = f"User {user_id}"
            method_description = f"Collaborative filtering ({collab_method.replace('_', '-')})"
            
        elif method == 'hybrid':
            if not movie and not user_id:
                context['error'] = "Please provide movie and/or user ID"
                return render_template('index.html', **context)
            
            recommended_movies, recommended_posters = rec_system.recommend_hybrid(movie, user_id)
            
            title_parts = []
            if movie:
                title_parts.append(f"'{movie}'")
            if user_id:
                title_parts.append(f"User {user_id}")
            display_title = " + ".join(title_parts)
            method_description = "Hybrid recommendations"
        
        if not recommended_movies:
            context['error'] = "No recommendations found"
            return render_template('index.html', **context)
        
        recommendations = list(zip(recommended_movies, recommended_posters))
        
        return render_template('recommend.html',
                             movie=display_title,
                             recommendations=recommendations,
                             method=method,
                             method_description=method_description,
                             total_recommendations=len(recommendations))
    
    except Exception as e:
        print(f"Error in recommend route: {e}")
        import traceback
        traceback.print_exc()
        context = get_template_context()
        context['error'] = "An error occurred"
        return render_template('index.html', **context)

@app.route('/api/status')
def api_status():
    return jsonify({
        'status': 'online',
        'available_methods': rec_system.get_available_methods(),
        'total_movies': len(rec_system.get_movie_list()),
        'user_range': rec_system.get_user_range()
    })

if __name__ == '__main__':
    print("🚀 Starting Self-Contained Movie Recommendation System...")
    print("=" * 60)
    
    print(f"📊 Total movies: {len(rec_system.get_movie_list())}")
    
    available_methods = rec_system.get_available_methods()
    for method, available in available_methods.items():
        status = "✅" if available else "❌"
        method_name = method.replace('_', ' ').title()
        print(f"{status} {method_name}")
    
    min_user, max_user = rec_system.get_user_range()
    print(f"👥 User ID range: {min_user}-{max_user}")
    
    print("\n📍 Open: http://localhost:5000")
    print("=" * 60)
    print("🎬 Ready to serve recommendations!")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
