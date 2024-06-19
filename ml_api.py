from flask import Flask, request, jsonify, render_template_string
import tensorflow as tf
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the trained model and scaler
model = tf.keras.models.load_model(r'C:\Users\lenovo\Downloads\Machine Learning\ML_RecomenderSystem\script ml api\model\recomender_model.h5')
with open(r'C:\Users\lenovo\Downloads\Machine Learning\ML_RecomenderSystem\script ml api\model\X_train_scaled.pkl', 'rb') as f:
    X_train_scaled = pickle.load(f)

# Load dataset
df = pd.read_csv(r'C:\Users\lenovo\Downloads\Machine Learning\ML_RecomenderSystem\dataset\data_file.csv')

# Define features
FEATURES = ['Product_ID', 'Product_Description', 'Product_Category', 'Product_Line', 'Raw_Material']

# Function to get product index
def get_product_index(user_input):
    try:
        if user_input.isdigit():  # Check if input is a number
            product_id = int(user_input)
            if product_id in df['Product_ID'].values:
                return df.loc[df['Product_ID'] == product_id].index[0]
        for feature in FEATURES[1:]:
            if user_input in df[feature].values:
                return df.loc[df[feature] == user_input].index[0]
    except Exception as e:
        print(f"Error finding product index: {e}")
    return None

@app.route('/')
def index():
    return render_template_string('''
    <form action="/find_product" method="post">
        <label for="product_input">Enter Product Information:</label><br>
        <input type="text" id="product_input" name="product_input" required><br><br>
        <input type="submit" value="Find Product">
    </form>
    ''')

@app.route('/find_product', methods=['POST'])
def find_product():
    user_input = request.form['product_input']
    index = get_product_index(user_input)

    if index is not None:
        product_features = X_train_scaled[index].reshape(1, -1)
        product_data = df.iloc[index].to_dict()

        # Convert int64 to int for JSON serialization
        for key in product_data:
            if isinstance(product_data[key], np.int64):
                product_data[key] = int(product_data[key])

        # Find similar products
        similarities = cosine_similarity(product_features, X_train_scaled).flatten()
        similar_indices = similarities.argsort()[-11:-1][::-1]
        
        recommended_products = []
        for idx in similar_indices:
            recommended_product = df.iloc[idx].to_dict()
            for key in recommended_product:
                if isinstance(recommended_product[key], np.int64):
                    recommended_product[key] = int(recommended_product[key])
            recommended_product['Similarity Score'] = similarities[idx]
            recommended_products.append(recommended_product)
        
        return jsonify({
            "product_data": product_data,
            "recommended_products": recommended_products
        }), 200

    else:
        return jsonify({"status": "ERROR", "message": "Product not found"}), 404

if __name__ == '__main__':
    app.run(debug=True)
