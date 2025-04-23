import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

WEIGHTS = {
    'Director': 0.09,
    'Genre': 0.10,
    'Cast 1': 0.15,
    'Cast 2': 0.11,
    'Cast 3': 0.08,
    'Cast 4': 0.05,
    'Music Director': 0.07,
    'Lead Singer': 0.04,
    'Teaser Views': 0.09,
    'Trailer Views': 0.11,
    'Best hits in Songs': 0.07,
    'Poster Views': 0.03,
    'Critics Review': 0.09,
    'IMDB Rating': 0.08
}

# --- Utility Functions ---
def get_weighted_mean_box_office(df, column, value, weight):
    weekend = df.loc[df[column] == value, 'Opening Weekend Collection'].dropna().tolist()
    total = df.loc[df[column] == value, 'Box Office Collection India'].dropna().tolist()

    weekend_pred = np.mean(weekend) * weight if weekend else 0
    total_pred = np.mean(total) * weight if total else 0

    return weekend_pred, total_pred

def predict_box_office_step1(df, selections, category):
    weekend, total = [], []

    for col, val in selections.items():
        if val:
            w_pred, t_pred = get_weighted_mean_box_office(df, col, val, WEIGHTS[col])
            weekend.append(w_pred)
            total.append(t_pred)

    weekend_result = round(sum(weekend), 0)
    total_result = round(sum(total), 0)

    if category in ["Religious/Political", "Political"]:
        weekend_result += 15
        total_result += 30

    return weekend_result, total_result

def predict_box_office_step2(weekend, total, views, category):
    for view_type in ['Teaser Views', 'Trailer Views', 'Best hits in Songs', 'Poster Views']:
        weekend += weekend * (views[view_type] / 100) * WEIGHTS[view_type]
        total += total * (views[view_type] / 100) * WEIGHTS[view_type]

    if category in ["Religious/Political", "Political"]:
        weekend += 20
        total += 30

    return round(weekend, 0), round(total, 0)

def predict_box_office_step3(weekend, total, imdb_rating, critics_review):
    weekend += weekend * (imdb_rating / 10) * WEIGHTS['IMDB Rating']
    total += total * (imdb_rating / 10) * WEIGHTS['IMDB Rating']

    weekend += weekend * (critics_review / 10) * WEIGHTS['Critics Review']
    total += total * (critics_review / 10) * WEIGHTS['Critics Review']

    return round(weekend, 0), round(total, 0)

# --- Streamlit UI ---
st.title("🎬 Box Office Prediction App")

# ✅ Automatically load Excel file from the project folder (GitHub)
excel_file = "Movie Collections 2023.xlsx" 
df = pd.read_excel(excel_file, engine="openpyxl")
df.columns = df.columns.str.strip()

st.subheader("Data Preview")
st.dataframe(df.head())

st.sidebar.header("Select Movie Attributes")
selections = {
    'Director': st.sidebar.selectbox("Director", [''] + sorted(df['Director'].dropna().unique().tolist())),
    'Genre': st.sidebar.selectbox("Genre", [''] + sorted(df['Genre'].dropna().unique().tolist())),
    'Music Director': st.sidebar.selectbox("Music Director", [''] + sorted(df['Music Director'].dropna().unique().tolist())),
    'Lead Singer': st.sidebar.selectbox("Lead Singer", [''] + sorted(df['Lead Singer'].dropna().unique().tolist())),
}

# Collect all unique casts from Cast 1-4
cast_columns = ['Cast 1', 'Cast 2', 'Cast 3', 'Cast 4']
all_casts = pd.unique(df[cast_columns].values.ravel('K'))
all_casts = [cast for cast in all_casts if pd.notna(cast)]
all_casts_sorted = sorted(set(all_casts))

selections['Cast 1'] = st.sidebar.selectbox("Cast 1", [''] + all_casts_sorted)
selections['Cast 2'] = st.sidebar.selectbox("Cast 2", [''] + all_casts_sorted)
selections['Cast 3'] = st.sidebar.selectbox("Cast 3", [''] + all_casts_sorted)
selections['Cast 4'] = st.sidebar.selectbox("Cast 4", [''] + all_casts_sorted)

category = st.sidebar.radio("Movie Category", ["None", "Religious/Political", "Political"])

if st.sidebar.button("Predict Step 1"):
    step1_wknd, step1_total = predict_box_office_step1(df, selections, category)
    st.session_state['step1'] = (step1_wknd, step1_total)
    st.success(f"Step 1 - Weekend: {step1_wknd} Cr | Total: {step1_total} Cr")

if 'step1' in st.session_state:
    st.sidebar.header("Media Views")
    views = {
        'Teaser Views': st.sidebar.slider("Teaser Views (%)", 0, 100, 50),
        'Trailer Views': st.sidebar.slider("Trailer Views (%)", 0, 100, 50),
        'Best hits in Songs': st.sidebar.slider("Best Hits (%)", 0, 100, 50),
        'Poster Views': st.sidebar.slider("Poster Views (%)", 0, 100, 50)
    }

    if st.sidebar.button("Predict Step 2"):
        step2_wknd, step2_total = predict_box_office_step2(
            st.session_state['step1'][0],
            st.session_state['step1'][1],
            views,
            category
        )
        st.session_state['step2'] = (step2_wknd, step2_total)
        st.success(f"Step 2 - Weekend: {step1_wknd} Cr | Total: {step2_total} Cr")

if 'step2' in st.session_state:
    imdb = st.sidebar.slider("IMDB Rating", 0.0, 10.0, 6.5)
    critics = st.sidebar.slider("Critics Review", 0.0, 10.0, 5.0)

    if st.sidebar.button("Predict Step 3"):
        step3_wknd, step3_total = predict_box_office_step3(
            st.session_state['step2'][0],
            st.session_state['step2'][1],
            imdb,
            critics
        )

        st.success(f"Final Prediction - Weekend: {step1_wknd} Cr | Total: {step3_total} Cr")

        # Visualization
        st.subheader("Prediction Breakdown")
        labels = ['Step 1', 'Step 2', 'Final']
        weekend_values = [st.session_state['step1'][0], st.session_state['step2'][0], step3_wknd]
        total_values = [st.session_state['step1'][1], st.session_state['step2'][1], step3_total]

        fig1, ax1 = plt.subplots()
        ax1.bar(labels, weekend_values)
        ax1.set_title("Weekend Prediction")
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots()
        ax2.bar(labels, total_values)
        ax2.set_title("Total Box Office Prediction")
        st.pyplot(fig2)

        if st.button("Download Prediction"):
            result_df = pd.DataFrame({
                'Stage': labels,
                'Weekend Prediction': weekend_values,
                'Total Prediction': total_values
            })
            result_df.to_csv("prediction_output.csv", index=False)
            st.download_button("Download CSV", "prediction_output.csv", file_name="prediction.csv")
