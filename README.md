## MODISTA Machine-Learning

## What's MODISTA?
Presenting "MODISTA", a mobile application that provides a variety of current trend clothing

## Our Model
Our application provides convenience in making purchases and a good clothing recommendation system in accordance with current trends using machine learning for the accuracy of the recommendation system

![image](https://github.com/Modista-Team/Machine-Learning/assets/170929755/a53e76ce-4573-4b86-8640-40d415b22663)
![image](https://github.com/Modista-Team/Machine-Learning/assets/170929755/31444bbc-a1c9-4481-aa94-c1b585e64d82)

Our recommendation system is trained with 600+ data in csv format, achieving roughly 77% accuracy and 0.9647179684741083 R^2 Score.

![image](https://github.com/Modista-Team/Machine-Learning/assets/170929755/3e42f95d-7204-4449-9e56-38295b7865be)

## Recomendation System
### Dataset
https://www.kaggle.com/datasets/ishanshrivastava28/sales-transaction-dataset-with-product-details (Historical User CSV Dataset)

### Requirements
To run the notebook and utilize the model, the following dependencies are required:
- Numpy
- Pandas
- Scikit-learn
- Tensorflow
- Matplotlib
- pickle

To run the ml_api and utilize the model, the following dependencies are required:
- flask
- numpy
- pandas
- scikit-learn
- tensorflo
- numpy

### Flask API
User can input the product they want to search
![image](https://github.com/Modista-Team/Machine-Learning/assets/170929755/4232ccad-5ae2-4652-9751-2aeac9171d68)
![image](https://github.com/Modista-Team/Machine-Learning/assets/170929755/cfbfd93a-2503-438d-961b-2625f1053950)

**Generates description and top10 product recommendations:**
Description of the product you want to find
![image](https://github.com/Modista-Team/Machine-Learning/assets/170929755/d76cba97-2d4c-46fa-87b8-17b49b7fd0d7)

Top10 product recommendations
recommended_products": [
    {
      "Customer_ID": 15,
      "Date": 20230812,
      "Latitude": -6.729447,
      "Longitude": 40.743075,
      "Product_Category": "Womenswear",
      "Product_Description": "Ties",
      "Product_ID": 341,
      "Product_Line": "Shoes",
      "Quantity": 4,
      "Raw_Material": "Leather",
      "Region": "Wells",
      "Sales_Revenue": 370.84,
      "Similarity Score": 0.9226180600416957,
      "Unit_Price": 92.71
    },
    {
      "Customer_ID": 2,
      "Date": 20230815,
      "Latitude": -46.421333,
      "Longitude": -49.010338,
      "Product_Category": "Accessories",
      "Product_Description": "Knitwear",
      "Product_ID": 271,
      "Product_Line": "Shoes",
      "Quantity": 3,
      "Raw_Material": "Cashmere",
      "Region": "Wakefield",
      "Sales_Revenue": 243.57,
      "Similarity Score": 0.9155307689627981,
      "Unit_Price": 81.19
    },
    {
      "Customer_ID": 92,
      "Date": 20240117,
      "Latitude": 81.7841385,
      "Longitude": 163.623306,
      "Product_Category": "Womenswear",
      "Product_Description": "Formal Shirts",
      "Product_ID": 277,
      "Product_Line": "Trousers",
      "Quantity": 5,
      "Raw_Material": "Fabrics",
      "Region": "Wells",
      "Sales_Revenue": 368.35,
      "Similarity Score": 0.8659540513828182,
      "Unit_Price": 73.67
    },
    {
      "Customer_ID": 79,
      "Date": 20240306,
      "Latitude": 81.4370785,
      "Longitude": 165.783352,
      "Product_Category": "Sports",
      "Product_Description": "Formal Shirts",
      "Product_ID": 367,
      "Product_Line": "Leathers",
      "Quantity": 2,
      "Raw_Material": "Fabrics",
      "Region": "Worcester",
      "Sales_Revenue": 133.64,
      "Similarity Score": 0.8558351364273996,
      "Unit_Price": 66.82
    },
    {
      "Customer_ID": 18,
      "Date": 20231118,
      "Latitude": 40.4179215,
      "Longitude": 114.493308,
      "Product_Category": "Sports",
      "Product_Description": "Knitwear",
      "Product_ID": 389,
      "Product_Line": "Shoes",
      "Quantity": 5,
      "Raw_Material": "Wool",
      "Region": "Wakefield",
      "Sales_Revenue": 447.75,
      "Similarity Score": 0.8487451012346819,
      "Unit_Price": 89.55
    },
    {
      "Customer_ID": 41,
      "Date": 20240208,
      "Latitude": -12.9005065,
      "Longitude": -107.977783,
      "Product_Category": "Menswear",
      "Product_Description": "Casual Shirts",
      "Product_ID": 260,
      "Product_Line": "Shoes",
      "Quantity": 10,
      "Raw_Material": "Polyester",
      "Region": "Truro",
      "Sales_Revenue": 300.1,
      "Similarity Score": 0.8437671365718726,
      "Unit_Price": 30.01
    },
    {
      "Customer_ID": 56,
      "Date": 20230612,
      "Latitude": 58.681951,
      "Longitude": -59.702081,
      "Product_Category": "Sports",
      "Product_Description": "Polo Shirts",
      "Product_ID": 351,
      "Product_Line": "Tops",
      "Quantity": 4,
      "Raw_Material": "Fabrics",
      "Region": "Wells",
      "Sales_Revenue": 238.28,
      "Similarity Score": 0.8302604724025341,
      "Unit_Price": 59.57
    },
    {
      "Customer_ID": 92,
      "Date": 20231127,
      "Latitude": -65.0342225,
      "Longitude": -28.349124,
      "Product_Category": "Sports",
      "Product_Description": "Shorts",
      "Product_ID": 274,
      "Product_Line": "Leathers",
      "Quantity": 10,
      "Raw_Material": "Cashmere",
      "Region": "York",
      "Sales_Revenue": 382.7,
      "Similarity Score": 0.8266819145658888,
      "Unit_Price": 38.27
    },
    {
      "Customer_ID": 10,
      "Date": 20240205,
      "Latitude": -31.7520845,
      "Longitude": 33.600044,
      "Product_Category": "Accessories",
      "Product_Description": "Belts",
      "Product_ID": 380,
      "Product_Line": "Shoes",
      "Quantity": 5,
      "Raw_Material": "Cotton",
      "Region": "York",
      "Sales_Revenue": 285.8,
      "Similarity Score": 0.8040985964724375,
      "Unit_Price": 57.16
    },
    {
      "Customer_ID": 32,
      "Date": 20240420,
      "Latitude": 42.0442685,
      "Longitude": -34.542402,
      "Product_Category": "Accessories",
      "Product_Description": "Coats",
      "Product_ID": 262,
      "Product_Line": "Tops",
      "Quantity": 5,
      "Raw_Material": "Cotton",
      "Region": "Wells",
      "Sales_Revenue": 69.1,
      "Similarity Score": 0.8009695328201338,
      "Unit_Price": 13.82
    }
  ]
}



