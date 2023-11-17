import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
@st.cache_resource  
def load_model():
    # Load the trained model
    with open('./models/xgb_classifier.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
        return model

# Load the column names
columns = ['age', 'annual_income', 'num_bank_acc', 'num_credit_card',
       'interest_rate', 'delay_from_due_date', 'outstanding_debt',
       'credit_util_ratio', 'credit_history_age', 'installment_per_month',
       'amount_invested_monthly', 'monthly_balance',
       'payment_of_min_amount_Yes']

# Load the model
model = load_model()

@st.cache_resource  
def predict_credit_score(data):
    # Preprocess the input data as needed for your regression model
    # Make sure the order of features matches the order used during training
    
    data['payment_of_min_amount_Yes'] = data['payment_of_min_amount_Yes'].map({True: 1, False: 0})

    # Ensure that the test data has the same columns as your training data
    data = data.reindex(columns=columns, fill_value=0)

    # Make predictions
    prediction = model.predict(data)

     # Get feature importances
    feature_importances = model.feature_importances_

    return prediction, feature_importances
 
st.title('Credit Score Prediction App')

# Load image from a file path or URL
image_path = "img.jpg"  # Replace with the path to your image file or URL
st.image(image_path, use_column_width=True)


# Create sliders for numeric input
st.sidebar.header('Input Features')
age = st.sidebar.number_input('Age', min_value=18, max_value=100, value=21, step=1)
income = st.sidebar.number_input('Annual Income ($)', min_value=10000.0, max_value=1000000000.0, value=10000.0, step=5000)
num_of_bank_acc = st.sidebar.number_input('Number of Bank Accounts Owned', min_value=1 ,max_value=12, value=1, step=1)
num_credit_card = st.sidebar.number_input('Number of Credit Cards Owned', min_value=1 ,max_value=12, value=1, step=1)
interest = st.sidebar.number_input('Credit Cards Interest rate (%)',min_value=1.0, max_value=33.0, value=1.0, step=1.0)
delay_from_due_date = st.sidebar.number_input('Days Delayed Since Due Date for Payment', min_value=0, max_value=90, value=1, step=5)
outstanding_debt = st.sidebar.number_input('Outstanding Debt ($)', min_value=0.0, max_value=100000.0, value=0.0, step=500.0)
credit_util = st.sidebar.number_input('Credit Card Utilization (%))', min_value=0.0, max_value=100.0, value=0.0, step=5.0)
credit_history = st.sidebar.number_input('Credit History Age (ie. 1.1 = 1 year 1 month)', min_value=0.1, max_value=100.12, value=0.1)
installment_per_month = st.sidebar.number_input('Credit Card Installment per Month ($)', min_value=0.0, max_value=100000.0, value=0.0 , step=100.0)
amount_invested_month = st.sidebar.number_input('Amount Invested Per Month ($)', min_value=0.0, max_value=100000.0, value=0.0, step=100.0)
monthly_balance = st.sidebar.number_input('Monthly Balance ($)', min_value=0.0, max_value=3000.0, value=0.0, step=100.0)
payment_of_min_amount = st.sidebar.selectbox('Credit Card Payment in minimum?', ['Yes', 'No'])

# Create a DataFrame from the user input
user_data = pd.DataFrame({
    'age': [age],  # Ensure 'age' is cast to int
    'annual_income': [income],
    'num_bank_acc': [num_of_bank_acc],  # Ensure 'num_bank_acc' is cast to int
    'num_credit_card': [num_credit_card],  # Ensure 'num_credit_card' is cast to int
    'interest_rate': [interest],
    'delay_from_due_date': [delay_from_due_date],  # Ensure 'delay_from_due_date' is cast to int
    'outstanding_debt': [outstanding_debt],
    'credit_util_ratio': [credit_util],
    'credit_history_age': [credit_history],
    'installment_per_month': [installment_per_month],
    'amount_invested_monthly': [amount_invested_month],
    'monthly_balance': [monthly_balance],
    'payment_of_min_amount_Yes': [payment_of_min_amount == 'Yes'],
})

# Initialize prediction variable
prediction = None

# Add a "Predict" button
if st.button('Predict'):
    prediction, feature_importances = predict_credit_score(user_data)

    # Display the prediction
    if prediction is not None:
        class_labels = {0: 'Poor Credit Score', 1: 'Average Credit Score', 2: 'Good Credit Score'}
        predicted_class = class_labels.get(prediction[0], 'Unknown')
        
        # Apply different styles or colors based on the predicted label
        if predicted_class == 'Good Credit Score':
            style = "color: green; font-size: 24px;"
        elif predicted_class == 'Average Credit Score':
            style = "color: yellow; font-size: 24px;"
        elif predicted_class == 'Poor Credit Score':
            style = "color: red; font-size: 24px;"

        # Display prediction with customized style
        st.write(f"<h3 style='{style}'>Model's Prediction : {predicted_class}</h3>", unsafe_allow_html=True)

        # Display user input features as a table
        st.subheader('User Input Features for Prediction')
        st.write(user_data)









    
    
    
