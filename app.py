import streamlit as st
import pandas as pd
import os

# Load processed data
df = pd.read_csv("students_with_recommendations.csv")

# Create dynamic recommendations
subject_cols = [
    'math_score', 'history_score', 'physics_score',
    'chemistry_score', 'biology_score', 'english_score', 'geography_score'
]

def dynamic_recommendations(row):
    weak_subjects = [subj for subj in subject_cols if row[subj] < 50]
    resources = [f"Improve {subj.split('_')[0].capitalize()} Skills" for subj in weak_subjects]
    return resources if resources else ["You're doing great in all subjects! Keep going!"]

df['personalized_materials'] = df.apply(dynamic_recommendations, axis=1)

# Streamlit UI
st.title("📚  AI-Driven Personalized Learning Experience Platform")

student_index = st.number_input("Enter Student Row Number", min_value=0, max_value=len(df)-1, value=0)
student_data = df.iloc[student_index]

st.subheader("🎯 Student Engagement Profile")
st.write(student_data[['quiz_score', 'time_spent', 'engagement_score']])

st.subheader("📘 Cluster-Based Recommendations")
st.write(student_data['recommended_materials'])

st.subheader("📗 Personalized Subject-Based Tips")
st.write(student_data['personalized_materials'])

# Feedback section
st.subheader("🗣️ Feedback")
feedback = st.radio("Was this recommendation helpful?", ("Yes", "No"))
if feedback == "Yes":
    st.success("Glad it helped!")
else:
    st.warning("Thanks for your feedback!")

# Log feedback
with open("feedback_log.csv", "a") as f:
    f.write(f"{student_index},{feedback}\n")
