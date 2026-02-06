import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('C:/Users/Shankar/jupyter notes/LR PROJECT.csv')
    df['date'] = pd.to_datetime(df['date'])
    # Fill missing values as in notebook
    df['likes'] = df['likes'].fillna(df['likes'].median())
    df['comments'] = df['comments'].fillna(df['comments'].median())
    df['watch_time_minutes'] = df['watch_time_minutes'].fillna(df['watch_time_minutes'].median())
    df['engagement rate'] = (df['likes'] + df['comments']) / df['views']
    return df

def preprocess(df):
    X = df.drop(columns=['ad_revenue_usd', 'video_id', 'date'])
    y = df['ad_revenue_usd']
    cat_cols = ['category', 'device', 'country']
    num_cols = X.select_dtypes(include='number').columns
    preprocessor = ColumnTransformer([
        ('num', 'passthrough', num_cols),
        ('cat', OneHotEncoder(drop='first'), cat_cols)
    ])
    X_encoded = preprocessor.fit_transform(X)
    feature_names = list(num_cols) + list(preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols))
    X_encoded_df = pd.DataFrame(X_encoded, columns=feature_names)
    return X_encoded_df, y, preprocessor, feature_names

def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

def main():
    st.title('Content Monetization Revenue Predictor')
    df = load_data()
    X_encoded_df, y, preprocessor, feature_names = preprocess(df)
    model = train_model(X_encoded_df, y)

    st.header('Predict Revenue from User Input')
    # User input form
    with st.form('input_form'):
        views = st.number_input('Views', min_value=0, value=int(df["views"].median()))
        likes = st.number_input('Likes', min_value=0, value=int(df["likes"].median()))
        comments = st.number_input('Comments', min_value=0, value=int(df["comments"].median()))
        watch_time = st.number_input('Watch Time (minutes)', min_value=0.0, value=float(df["watch_time_minutes"].median()))
        video_length = st.number_input('Video Length (minutes)', min_value=0.0, value=float(df["video_length_minutes"].median()))
        subscribers = st.number_input('Subscribers', min_value=0, value=int(df["subscribers"].median()))
        category = st.selectbox('Category', df['category'].unique())
        device = st.selectbox('Device', df['device'].unique())
        country = st.selectbox('Country', df['country'].unique())
        submitted = st.form_submit_button('Predict')
    if submitted:
        input_df = pd.DataFrame([{
            'views': views,
            'likes': likes,
            'comments': comments,
            'watch_time_minutes': watch_time,
            'video_length_minutes': video_length,
            'subscribers': subscribers,
            'category': category,
            'device': device,
            'country': country,
            'engagement rate': (likes + comments) / views if views > 0 else 0
        }])
        input_encoded = preprocessor.transform(input_df)
        pred = model.predict(input_encoded)
        st.success(f'Predicted Ad Revenue (USD): {pred[0]:.2f}')

    st.header('Basic Visual Analytics')
    st.subheader('Revenue Distribution')
    fig, ax = plt.subplots()
    sns.histplot(df['ad_revenue_usd'], bins=30, ax=ax)
    st.pyplot(fig)

    st.subheader('Correlation Heatmap')
    fig2, ax2 = plt.subplots()
    sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, ax=ax2)
    st.pyplot(fig2)

    st.header('Model Insights')
    # Calculate model metrics
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    y_pred = model.predict(X_encoded_df)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    st.markdown(f"**R² Score:** {r2:.4f}")
    st.markdown(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
    st.markdown(f"**Mean Absolute Error (MAE):** {mae:.2f}")
    st.write('Linear Regression Coefficients:')
    coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': model.coef_})
    st.dataframe(coef_df.sort_values('Coefficient', key=abs, ascending=False))

if __name__ == '__main__':
    main()
