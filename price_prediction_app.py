import streamlit as st
import numpy as np
import pandas as pd
import pickle



# Streamlit app code
def main():



    # Load the saved model
    model = pickle.load(open('Price_Predict.pkl', "rb"))  # rb = read binary


    # Load the scaling parameters
    scaler = pickle.load(open('Scaler.pkl', "rb"))   # rb = read binary





    # Create a list for all unique value present in categorical columns in dataframe
    Brand = ['Maruti', 'Hyundai', 'Datsun', 'Honda', 'Tata', 'Chevrolet', 'Toyota', 'Jaguar',
         'Mercedes-Benz', 'Audi', 'Skoda', 'Jeep', 'BMW', 'Mahindra', 'Ford', 'Nissan',
         'Renault', 'Fiat', 'Volkswagen', 'Volvo', 'Mitsubishi', 'Land', 'Daewoo', 'MG',
         'Force', 'Isuzu', 'OpelCorsa', 'Ambassador', 'Kia']

    Year = [1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003,
        2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]

    Fuel_Type = ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric']

    Seller_Type = ['Individual', 'Dealer', 'Trustmark Dealer']

    Transmission = ["Manual", "Automatic"]

    Owner_Type = ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car']


    # Function to preprocess and scale user input
    def preprocess_input(user_input):

        # Create a dataframe from the user input
        input_df = pd.DataFrame(user_input)


        # Encode the categorical columns
        encoded_df = pd.get_dummies(data = input_df, columns = [i for i in input_df.select_dtypes(include = 'object')], dtype= 'int')

        # Ensure all columns are present in the encoded dataframe
        missing_cols = set(scaler.feature_names_in_) - set(encoded_df.columns)
        for col in missing_cols:
            encoded_df[col] = 0
        

        # Ensure all columns are present and in the same order as they are present in actual preprocessed dataframe for scaling
        encoded_df = encoded_df.reindex(columns=scaler.feature_names_in_)


        # Scale the data using the saved scaler
        scaled_data = scaler.transform(encoded_df)

        return scaled_data


    # Function to predict car price
    def predict_price(scaled_data):
        # Make predictions using the pre-trained model
        predicted_price = np.round(model.predict(scaled_data), 2)
        return predicted_price




    st.title("Used Car Price Prediction")

    # Collect user input using sidebar widgets
    brand = st.sidebar.selectbox('Select Brand', Brand)
    year = st.sidebar.selectbox('Select Year of Launch', Year)
    km_driven = st.sidebar.number_input('Kilometre Driven')
    fuel_type = st.sidebar.selectbox('Select Fuel Type', Fuel_Type)
    seller_type = st.sidebar.selectbox('Select Seller Type', Seller_Type)
    transmission = st.sidebar.selectbox('Select Transmission', Transmission)
    owner_type = st.sidebar.selectbox('Select Owner Type', Owner_Type)

    # Preprocess and scale user input
    user_input = {
        'brand': [brand],
        'year': [year],
        'km_driven': [km_driven],
        'fuel': [fuel_type],
        'seller_type': [seller_type],
        'transmission': [transmission],
        'owner': [owner_type]
    }
    scaled_features = preprocess_input(user_input)

    # # Decode the encoded features for display
    # decoded_features = decode_features(user_input)

    # Display the selected features
    st.subheader('Selected Features:')
    input_data = {
        'brand': brand,
        'year': year,
        'km_driven': km_driven,
        'fuel': fuel_type,
        'seller_type': seller_type,
        'transmission': transmission,
        'owner': owner_type
    }
    st.write('INPUT DATA', input_data)
    if st.button("Predict"):
        # Predict the car price
        predicted_price = predict_price(scaled_features)

        # Display the predicted price
        st.subheader('Predicted Car Price:')
        #st.write(predicted_price)
        st.write("Predicted Selling Price :","  ","₹","  ",predicted_price)

# Run the app
if __name__ == '__main__':
    main()