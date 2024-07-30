from flask import Flask, request, jsonify
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pymysql.cursors
import random
import os

db = pymysql.connect(host=os.getenv('DB_HOST'),
    port=os.getenv('DB_PORT'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD'),
    db=os.getenv('DB_NAME'),
    charset='utf8mb4',
    cursorclass=pymysql.cursors.DictCursor)

app = Flask(__name__)

@app.route('/recommend', methods=['GET'])
def recommend():
    product_id = request.args.get('product_id', type=int)
    if not product_id:
        return jsonify({"error": "Product ID is required"}), 400
    
    cursor = db.cursor()
    cursor.execute("SELECT * FROM products p JOIN reviews r ON r.product_id = p.id")
    results = cursor.fetchall()
    
    df = pd.DataFrame(results)

    if df.empty or df[df['product_id'] == product_id].empty:
        cursor.execute("SELECT * FROM products")
        all_products = cursor.fetchall()
        all_products_ids = [p['id'] for p in all_products]
        random_recommendations = random.sample(all_products_ids, min(3, len(all_products_ids)))
        return jsonify(random_recommendations)
        
    
    user_item_matrix = df.pivot_table(index='user', columns='product_id', values='rating')

    item_similarity = cosine_similarity(user_item_matrix.fillna(0).T)
    item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

    def get_recommendations(product, item_similarity_df, top_n=3):
        similar_scores = item_similarity_df[product].sort_values(ascending=False)
        top_products = similar_scores.iloc[1:top_n+1].index
        return top_products.tolist()

    recommendations = get_recommendations(product_id, item_similarity_df)
    
    return jsonify(recommendations)


@app.route('/')
def index():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run(debug=True)