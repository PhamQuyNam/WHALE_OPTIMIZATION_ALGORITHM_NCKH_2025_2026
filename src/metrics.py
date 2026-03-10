import numpy as np
from sklearn.metrics import mean_squared_error
from collections import defaultdict

def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_mae(y_true, y_pred):
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

def rmse(model, data):
    se = 0
    for u, i, r in data:
        u, i = int(u), int(i)
        pred = model.P[u] @ model.Q[i]
        se += (r - pred)**2
    return np.sqrt(se / len(data))

def get_top_k_recommendations(model, user_id, n_items, k=10):
    """Generate top-K item recommendations for a given user"""
    # Predict ratings for all items for this user
    user_factors = model.P[user_id]
    all_predictions = np.dot(user_factors, model.Q.T)
    
    # Get indices of top K highest predicted ratings
    top_k_items = np.argsort(all_predictions)[::-1][:k]
    return top_k_items

def evaluate_ranking_metrics(model, train_data, test_data, n_users, n_items, k=10):
    """Calculate Precision@K, Recall@K, and NDCG@K"""
    # Build ground truth from test data (items user interacted with/rated highly)
    # For implicit feedback, any interaction is positive. For explicit (1-5), maybe threshold >= 4
    # Here we assume test_data format is [u, i, r]
    ground_truth = defaultdict(set)
    for u, i, r in test_data:
        # We consider a rating of >= 4.0 as a "relevant" or "liked" item
        if r >= 4.0: 
            ground_truth[int(u)].add(int(i))
            
    # Also we need to filter out items seen in training to truly recommend *new* items
    train_seen = defaultdict(set)
    if train_data is not None:
        for u, i, r in train_data:
            train_seen[int(u)].add(int(i))

    precisions = []
    recalls = []
    ndcgs = []

    # Evaluate for each user in the test set who has relevant items
    for u in ground_truth.keys():
        relevant_items = ground_truth[u]
        if len(relevant_items) == 0:
            continue
            
        # Predict ratings for ALL items for this user
        user_pred = np.dot(model.P[u], model.Q.T)
        
        # Mask out items the user has already seen in training
        seen_items = list(train_seen[u])
        user_pred[seen_items] = -np.inf 
        
        # Get Top-K recommendations
        top_k_items = np.argsort(user_pred)[::-1][:k]
        
        # Calculate Hits
        hits = [1 if item in relevant_items else 0 for item in top_k_items]
        
        # Precision@K
        precision = sum(hits) / k
        precisions.append(precision)
        
        # Recall@K
        recall = sum(hits) / min(len(relevant_items), k) # Or just / len(relevant_items) depending on definition
        recalls.append(recall)
        
        # NDCG@K
        idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(relevant_items), k))])
        dcg = sum([hits[i] / np.log2(i + 2) for i in range(k)])
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcgs.append(ndcg)

    return {
        f"Precision@{k}": np.mean(precisions),
        f"Recall@{k}": np.mean(recalls),
        f"NDCG@{k}": np.mean(ndcgs)
    }


