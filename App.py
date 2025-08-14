import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations to avoid TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Suppress TensorFlow INFO and WARNING logs

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import GridSearchCV
from warnings import filterwarnings

# Suppress unnecessary warnings
filterwarnings("ignore")
# Set Seaborn theme for consistent plot styling
sns.set_theme(style="darkgrid")

# Set the app title
st.title("Zomato Bangalore Restaurants Analysis and Prediction")

# Load the dataset
try:
    df = pd.read_csv("zomato.csv")
    st.write("Dataset loaded successfully.")
except FileNotFoundError:
    st.error("Error: 'zomato.csv' not found in the project folder. Please download it from https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants and place it in 'C:\\Users\\manar\\Downloads\\Final model'.")
    st.stop()

# --- Data Preprocessing ---
# Remove unnecessary columns
del df['url']
del df['address']
del df['phone']
# Rename columns for clarity
df.rename(columns={'approx_cost(for two people)': 'average_cost', 'listed_in(city)': 'locality', 'listed_in(type)': 'restaurant_type'}, inplace=True)
# Fill missing values using forward fill
df.fillna(method="ffill", inplace=True)
# Extract rating from 'rate' column (e.g., '4.1/5' -> '4.1')
df['rating'] = df['rate'].str[0:3]
# Replace special values in rating
df['rating'] = df['rating'].replace(['NEW'], '1.0')
df['rating'] = df['rating'].replace(['-'], '0.0')
# Convert rating to numeric, filling invalid values with 0.0
df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0.0)
# Create target column: 1 (Good) if rating >= 3.75, else 0 (Poor)
df['target'] = np.where(df['rating'] >= 3.75, '1', '0')
# Drop original 'rate' column
df = df.drop('rate', axis=1)
# Convert target to numeric
df['target'] = pd.to_numeric(df['target'])
# Store original categorical values for EDA to avoid KeyError: 10
original_cols = {
    'locality': df['locality'].copy(),
    'rest_type': df['rest_type'].copy(),
    'average_cost': df['average_cost'].copy(),
    'cuisines': df['cuisines'].copy(),
    'dish_liked': df['dish_liked'].copy()
}
# Encode categorical columns using LabelEncoder
les = {}
for col in ['online_order', 'location', 'book_table', 'rest_type', 'dish_liked', 'cuisines', 'average_cost', 'reviews_list', 'menu_item', 'restaurant_type', 'locality']:
    les[col] = LabelEncoder()
    df[col] = les[col].fit_transform(df[col].astype(str))
# Normalize columns to [0, 1] range
def normalize_col(col_name):
    return (df[col_name] - df[col_name].min()) / (df[col_name].max() - df[col_name].min())
for col in ['online_order', 'location', 'book_table', 'votes', 'rest_type', 'dish_liked', 'cuisines', 'average_cost', 'reviews_list', 'menu_item', 'restaurant_type', 'locality', 'rating']:
    df[col] = normalize_col(col)
# Drop 'name' column as it's not used
df.drop(['name'], axis=1, inplace=True)

# Function to train models on a subset of data
@st.cache_resource
def train_models(df_subset):
    # Prepare features (X) and target (Y) from subset
    X = df_subset.drop('target', axis=1).values
    Y = df_subset['target'].values
    # Split data into training (70%) and testing (30%)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
    models_dict = {}
    
    # Logistic Regression
    LL = LogisticRegression(solver='liblinear', max_iter=1000, random_state=31)
    LL.fit(X_train, Y_train)
    Y_pred_lr = LL.predict(X_test)
    score_lr = round(accuracy_score(Y_pred_lr, Y_test) * 100, 2)
    models_dict['Logistic Regression'] = {'model': LL, 'accuracy': score_lr}
    
    # KNN Classifier
    kclf = KNeighborsClassifier(n_neighbors=31, leaf_size=30)
    kclf.fit(X_train, Y_train)
    Y_pred_kclf = kclf.predict(X_test)
    score_kclf = round(accuracy_score(Y_pred_kclf, Y_test) * 100, 2)
    models_dict['KNN Classifier'] = {'model': kclf, 'accuracy': score_kclf}
    
    # SVM
    svm = SVC(C=8.0, kernel='rbf', degree=3, gamma='scale', coef0=0.01, shrinking=True, probability=True, tol=0.1, cache_size=300, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovo')
    svm.fit(X_train, Y_train)
    Y_pred_svm = svm.predict(X_test)
    score_svm = round(accuracy_score(Y_pred_svm, Y_test) * 100, 2)
    models_dict['Support Vector Machine'] = {'model': svm, 'accuracy': score_svm}
    
    # Gaussian Naive Bayes
    gnb = GaussianNB(var_smoothing=1e-50)
    gnb.fit(X_train, Y_train)
    Y_pred_gnb = gnb.predict(X_test)
    score_gnb = round(accuracy_score(Y_pred_gnb, Y_test) * 100, 2)
    models_dict['Gaussian Naive Bayes'] = {'model': gnb, 'accuracy': score_gnb}
    
    # Random Forest with GridSearch
    param_grid = {
        'max_depth': [5, 10],
        'n_estimators': [10, 20],
        'max_features': ['sqrt'],
        'criterion': ['gini']
    }
    RFclf = RandomForestClassifier()
    grid = GridSearchCV(estimator=RFclf, param_grid=param_grid, cv=4, n_jobs=2, verbose=0)
    grid_result = grid.fit(X_train, Y_train)
    model_rf_grid = grid_result.best_estimator_
    Y_pred_RFclf = model_rf_grid.predict(X_test)
    score_RFclf = round(accuracy_score(Y_pred_RFclf, Y_test) * 100, 2)
    models_dict['Random Forest with GridSearch'] = {'model': model_rf_grid, 'accuracy': score_RFclf}
    
    # Random Forest
    max_accuracy = 0
    best_x = 0
    for x in range(10):
        rf = RandomForestClassifier(random_state=x)
        rf.fit(X_train, Y_train)
        Y_pred_rf = rf.predict(X_test)
        current_accuracy = round(accuracy_score(Y_pred_rf, Y_test) * 100, 2)
        if current_accuracy > max_accuracy:
            max_accuracy = current_accuracy
            best_x = x
    rf = RandomForestClassifier(random_state=best_x)
    rf.fit(X_train, Y_train)
    Y_pred_rf = rf.predict(X_test)
    score_rf = round(accuracy_score(Y_pred_rf, Y_test) * 100, 2)
    models_dict['Random Forest'] = {'model': rf, 'accuracy': score_rf}
    
    # XGBoost
    xgb = XGBClassifier(learning_rate=0.001, n_estimators=500, subsample=1.0, max_depth=10)
    xgb.fit(X_train, Y_train)
    Y_pred_xgb = xgb.predict(X_test)
    score_xgb = round(accuracy_score(Y_pred_xgb, Y_test) * 100, 2)
    models_dict['XGBoost'] = {'model': xgb, 'accuracy': score_xgb}
    
    # Decision Tree
    tclf = DecisionTreeClassifier(max_depth=10, min_samples_split=2, min_samples_leaf=1, random_state=46)
    tclf.fit(X_train, Y_train)
    Y_pred_tclf = tclf.predict(X_test)
    score_tclf = round(accuracy_score(Y_pred_tclf, Y_test) * 100, 2)
    models_dict['Decision Tree'] = {'model': tclf, 'accuracy': score_tclf}
    
    # Neural Network
    model_nn = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model_nn.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    history = model_nn.fit(X_train, Y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
    test_loss, test_accuracy = model_nn.evaluate(X_test, Y_test, verbose=0)
    models_dict['Neural Network'] = {'model': model_nn, 'accuracy': test_accuracy * 100}
    
    # Select the best model based on accuracy
    best_model_name = max(models_dict, key=lambda x: models_dict[x]['accuracy'])
    best_model = models_dict[best_model_name]['model']
    return models_dict, best_model_name, best_model

# Function to display EDA visualizations
def display_eda():
    st.header("Exploratory Data Analysis")
    st.write("This page displays visualizations to explore the Zomato Bangalore Restaurants dataset.")
    
    # Online Order and Table Booking Distribution
    st.subheader("Online Order and Table Booking Distribution")
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    (ax1, ax2) = ax
    labels = ["Yes", "No"]
    values = df['online_order'].value_counts().tolist()[:2]
    ax1.pie(x=values, labels=labels, autopct="%1.1f%%", colors=['#AAb3ff', '#CC80FF'], shadow=True, startangle=45, explode=[0.1, 0.1])
    ax1.set_title("ONLINE ORDER", fontdict={'fontsize': 12}, fontweight='bold')
    values = df['book_table'].value_counts().tolist()[:2]
    ax2.pie(x=values, labels=labels, autopct="%1.1f%%", colors=['#AAb3ff', '#CC80FF'], shadow=True, startangle=45, explode=[0.1, 0.1])
    ax2.set_title("TABLE BOOKING", fontdict={'fontsize': 12}, fontweight='bold')
    st.pyplot(fig)
    
    # Restaurant Type Distribution
    st.subheader("Restaurant Type Distribution")
    fig = plt.figure(figsize=(10, 5))
    round(df["restaurant_type"].value_counts() / df.shape[0] * 100, 2).plot.pie(autopct='%1.1f%%', colors=['#AAb3ff', '#CC80FF', '#DD00AA', '#c4ff4d', '#339933', '#FF0099', '#FF9933'], explode=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1))
    plt.title('Restaurant Type', fontdict={'fontsize': 12}, fontweight='bold')
    st.pyplot(fig)
    
    # Location of Restaurants
    st.subheader("Location of Restaurants")
    fig = plt.figure(figsize=(10, 5))
    locality_counts = original_cols['locality'].value_counts()[:10]
    sns.barplot(x=locality_counts.values, y=locality_counts.index, palette="twilight_shifted")
    plt.title("Location of the Restaurant", fontdict={'fontsize': 12}, fontweight='bold')
    plt.xlabel("Number of Restaurants")
    st.pyplot(fig)
    
    # Restaurant Type vs Performance
    st.subheader("Restaurant Type vs Performance")
    fig = plt.figure(figsize=(10, 5))
    sns.countplot(x=df['restaurant_type'], hue='target', data=df, palette="twilight_shifted", saturation=2, dodge=True)
    plt.title('Restaurant Type vs Performance', fontdict={'fontsize': 12}, fontweight='bold')
    st.pyplot(fig)
    
    # Online Order vs Performance
    st.subheader("Online Order vs Performance")
    fig = plt.figure(figsize=(10, 5))
    sns.countplot(x=df['online_order'], hue='target', data=df, palette="twilight_shifted", saturation=2, dodge=True)
    plt.title('Online Order', fontdict={'fontsize': 12}, fontweight='bold')
    st.pyplot(fig)
    
    # Table Booking vs Performance
    st.subheader("Table Booking vs Performance")
    fig = plt.figure(figsize=(10, 5))
    sns.countplot(x=df['book_table'], hue='target', data=df, palette="twilight_shifted", saturation=2, dodge=True)
    plt.title('Table Booking', fontdict={'fontsize': 12}, fontweight='bold')
    st.pyplot(fig)
    
    # Most Common Restaurant Types
    st.subheader("Most Common Restaurant Types")
    fig = plt.figure(figsize=(10, 5))
    rest_type_counts = original_cols['rest_type'].value_counts()[:10]
    sns.barplot(x=rest_type_counts.values, y=rest_type_counts.index, palette="twilight_shifted")
    plt.title("Most Common Restaurant Types", fontdict={'fontsize': 12}, fontweight='bold')
    plt.xlabel("Number of Outlets")
    st.pyplot(fig)
    
    # Average Cost for Meal for Two
    st.subheader("Average Cost for Meal for Two")
    fig = plt.figure(figsize=(10, 5))
    average_cost_counts = original_cols['average_cost'].value_counts()[:10]
    sns.barplot(x=average_cost_counts.values, y=average_cost_counts.index, palette="twilight_shifted")
    plt.title('Average Cost for Meal for Two People', fontdict={'fontsize': 12}, fontweight='bold')
    st.pyplot(fig)
    
    # Most Common Cuisines
    st.subheader("Most Common Cuisines")
    fig = plt.figure(figsize=(10, 5))
    cuisines_counts = original_cols['cuisines'].value_counts()[:10]
    sns.barplot(x=cuisines_counts.values, y=cuisines_counts.index, palette="twilight_shifted")
    plt.title("Most Common Cuisines", fontdict={'fontsize': 12}, fontweight='bold')
    plt.xlabel("Number of Cuisines")
    st.pyplot(fig)
    
    # Most Liked Dishes
    st.subheader("Most Liked Dishes")
    fig = plt.figure(figsize=(10, 5))
    dish_liked_counts = original_cols['dish_liked'].value_counts()[:10]
    sns.barplot(x=dish_liked_counts.values, y=dish_liked_counts.index, palette="twilight_shifted")
    plt.title("Most Liked Dishes", fontdict={'fontsize': 12}, fontweight='bold')
    plt.xlabel("Number of Outlets")
    st.pyplot(fig)
    
    # Restaurant Type vs Online Order
    st.subheader("Restaurant Type vs Online Order")
    fig = plt.figure(figsize=(10, 5))
    sns.countplot(x=df['restaurant_type'], hue='online_order', data=df, palette="twilight_shifted", saturation=2, dodge=True)
    plt.title('Restaurant Type vs Online Order', fontdict={'fontsize': 12}, fontweight='bold')
    st.pyplot(fig)
    
    # Restaurant Type vs Table Booking
    st.subheader("Restaurant Type vs Table Booking")
    fig = plt.figure(figsize=(10, 5))
    sns.countplot(x=df['restaurant_type'], hue='book_table', data=df, palette="twilight_shifted", saturation=2, dodge=True)
    plt.title('Restaurant Type vs Table Booking', fontdict={'fontsize': 12}, fontweight='bold')
    st.pyplot(fig)
    
    # Location vs Online Order
    st.subheader("Location vs Online Order")
    fig = plt.figure(figsize=(10, 5))
    g = sns.countplot(x=df['locality'], hue='online_order', data=df, palette="twilight_shifted", saturation=2, dodge=True)
    g.set_xticklabels(g.get_xticklabels(), rotation=90, ha="right")
    plt.title('Effect of Location on Online Ordering', fontdict={'fontsize': 12}, fontweight='bold')
    st.pyplot(fig)
    
    # Location vs Table Booking
    st.subheader("Location vs Table Booking")
    fig = plt.figure(figsize=(10, 5))
    g = sns.countplot(x=df['locality'], hue='book_table', data=df, palette="twilight_shifted", saturation=2, dodge=True)
    g.set_xticklabels(g.get_xticklabels(), rotation=90, ha="right")
    plt.title('Effect of Location on Table Booking', fontdict={'fontsize': 12}, fontweight='bold')
    st.pyplot(fig)

# Function to display prediction interface
def display_prediction(models_dict, best_model_name, best_model):
    st.header("Predict Restaurant Performance")
    st.write("Enter the details of a new restaurant to predict its performance (0 = Poor, 1 = Good) using the best model: " + best_model_name)
    
    # Input fields
    online_order = st.selectbox("Online Order", ["Yes", "No"])
    book_table = st.selectbox("Book Table", ["Yes", "No"])
    votes = st.number_input("Votes", min_value=0, value=0)
    location = st.selectbox("Location", les['location'].classes_)
    rest_type = st.selectbox("Restaurant Type", les['rest_type'].classes_)
    dish_liked = st.text_input("Dishes Liked", "Biryani")
    cuisines = st.text_input("Cuisines", "North Indian")
    average_cost = st.text_input("Average Cost for Two", "300")
    reviews_list = st.text_input("Reviews List", "[]")
    menu_item = st.text_input("Menu Item", "[]")
    restaurant_type = st.selectbox("Restaurant Type (Listed)", les['restaurant_type'].classes_)
    locality = st.selectbox("Locality", les['locality'].classes_)
    rating = st.number_input("Rating", min_value=0.0, max_value=5.0, value=3.0)
    
    # Prediction button
    if st.button("Predict"):
        new_data = pd.DataFrame({
            'online_order': [online_order],
            'book_table': [book_table],
            'votes': [votes],
            'location': [location],
            'rest_type': [rest_type],
            'dish_liked': [dish_liked],
            'cuisines': [cuisines],
            'average_cost': [average_cost],
            'reviews_list': [reviews_list],
            'menu_item': [menu_item],
            'restaurant_type': [restaurant_type],
            'locality': [locality],
            'rating': [rating]
        })
        # Encode categorical inputs
        for col in ['online_order', 'book_table', 'location', 'rest_type', 'dish_liked', 'cuisines', 'average_cost', 'reviews_list', 'menu_item', 'restaurant_type', 'locality']:
            if col in les:
                try:
                    new_data[col] = les[col].transform(new_data[col].astype(str))
                except ValueError:
                    new_data[col] = les[col].transform([les[col].classes_[0]])[0]
        # Normalize inputs
        for col in ['online_order', 'book_table', 'votes', 'location', 'rest_type', 'dish_liked', 'cuisines', 'average_cost', 'reviews_list', 'menu_item', 'restaurant_type', 'locality', 'rating']:
            new_data[col] = (new_data[col] - df[col].min()) / (df[col].max() - df[col].min())
        # Make prediction
        if best_model_name == 'Neural Network':
            prediction = (best_model.predict(new_data.values, verbose=0) > 0.5).astype(int)
        else:
            prediction = best_model.predict(new_data.values)
        st.write(f"Predicted Performance: {'Good (1)' if prediction[0] == 1 else 'Poor (0)'}")


# Create a subset for model training
df_subset = df.sample(n=5000, random_state=42)
# Train models and get results
models_dict, best_model_name, best_model = train_models(df_subset)

# Create sidebar for page navigation
page = st.sidebar.selectbox("Select Page", ["EDA", "Prediction"])

# Display the selected page
if page == "EDA":
    display_eda()
else:
    display_prediction(models_dict, best_model_name, best_model)