#!/usr/bin/env python
# coding: utf-8
# In[6]:
import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
# Charger les données
final_df_cupid = pd.read_csv('hearthack_final_df.csv')
# Préparer l'interface utilisateur
st.title(":flèche_de_cupidon: Application de Recommandation de Match Idéal")
st.header("Entrez vos informations pour trouver votre match parfait !")
# Entrées utilisateur
age = st.slider("Âge", 18, 100, 28)
user_status = st.selectbox("Statut", ["Single", "In a relationship"])
user_sex = st.selectbox("Sexe", ["Male", "Female"])
user_orientation = st.selectbox("Orientation", ["Straight", "Gay", "Bisexual"])
user_body_type = st.selectbox("Type de corps", ["Fit", "Average", "Athletic", "Overweight"])
user_diet = st.selectbox("Régime alimentaire", ["Anything", "Vegetarian", "Vegan"])
user_education = st.selectbox("Niveau d'éducation", ["High school", "College", "Wildcodeschool", "Graduate"])
height = st.slider("Taille (en cm)", 140, 220, 180)
user_job = st.selectbox("Métier", ["Other", "Computer / hardware / software", "Art / music / writing", "Sales / marketing / biz dev"])
user_pets = st.selectbox("Animaux de compagnie", ["Likes pets", "Does not like pets"])
user_smokes = st.selectbox("Fume", ["No", "Yes"])
# Convertir les entrées en format compatible avec le modèle
status_dict = {"Single": 0, "In a relationship": 1}
sex_dict = {"Male": 1, "Female": 0}
orientation_dict = {"Straight": "straight", "Gay": "gay", "Bisexual": "bisexual"}
body_type_dict = {"Fit": 2, "Average": 1, "Athletic": 3, "Overweight": 4}
diet_dict = {"Anything": 0, "Vegetarian": 1, "Vegan": 2}
education_dict = {"High school": 1, "College": 2, "Wildcodeschool": 0, "Graduate": 3}
job_dict = {"Other": 0, "Computer / hardware / software": 12, "Art / music / writing": 1, "Sales / marketing / biz dev": 2}
pets_dict = {"Likes pets": 1, "Does not like pets": 0}
smokes_dict = {"No": 0, "Yes": 1}
user_input = {
    'age': age,
    'status': status_dict[user_status],
    'sex': sex_dict[user_sex],
    'orientation': orientation_dict[user_orientation],
    'body_type': body_type_dict[user_body_type],
    'diet': diet_dict[user_diet],
    'education': education_dict[user_education],
    'height': height,
    'job': job_dict[user_job],
    'pets': pets_dict[user_pets],
    'smokes': smokes_dict[user_smokes]
}
user_input_df = pd.DataFrame([user_input])
user_input_df = pd.get_dummies(user_input_df, columns=['orientation'])
required_orientation_columns = ['orientation_bisexual', 'orientation_gay', 'orientation_straight']
for column in required_orientation_columns:
    if column not in user_input_df.columns:
        user_input_df[column] = 0
# Fonction pour trouver le match idéal
def find_ideal_match():
    # Filtrer les données en fonction des préférences utilisateur
    if user_input_df['sex'].values[0] == 1:  # Homme
        if user_input_df['orientation_straight'].values[0] == 1:
            filtered_df = final_df_cupid[
                ((final_df_cupid['sex'] == 0) & (final_df_cupid['orientation_straight'] == 1)) |
                ((final_df_cupid['sex'] == 0) & (final_df_cupid['orientation_bisexual'] == 1))
            ]
        elif user_input_df['orientation_gay'].values[0] == 1:
            filtered_df = final_df_cupid[
                ((final_df_cupid['sex'] == 1) & (final_df_cupid['orientation_gay'] == 1)) |
                ((final_df_cupid['sex'] == 1) & (final_df_cupid['orientation_bisexual'] == 1))
            ]
        elif user_input_df['orientation_bisexual'].values[0] == 1:
            filtered_df = final_df_cupid[
                (final_df_cupid['sex'] == 0) |
                (final_df_cupid['sex'] == 1)
            ]
    else:  # Femme
        if user_input_df['orientation_straight'].values[0] == 1:
            filtered_df = final_df_cupid[
                ((final_df_cupid['sex'] == 1) & (final_df_cupid['orientation_straight'] == 1)) |
                ((final_df_cupid['sex'] == 1) & (final_df_cupid['orientation_bisexual'] == 1))
            ]
        elif user_input_df['orientation_gay'].values[0] == 1:
            filtered_df = final_df_cupid[
                ((final_df_cupid['sex'] == 0) & (final_df_cupid['orientation_gay'] == 1)) |
                ((final_df_cupid['sex'] == 0) & (final_df_cupid['orientation_bisexual'] == 1))
            ]
        elif user_input_df['orientation_bisexual'].values[0] == 1:
            filtered_df = final_df_cupid[
                (final_df_cupid['sex'] == 0) |
                (final_df_cupid['sex'] == 1)
            ]
    # Aligner les colonnes avec le dataset
    for column in filtered_df.columns:
        if column not in user_input_df.columns and column != 'name':
            user_input_df[column] = 0
    user_input_encoded = user_input_df[filtered_df.drop(columns=['Lien_photo', 'name', 'question_1', 'question_2', 'question_3']).columns]
    # Instancier et ajuster le modèle KNN
    knn_model = NearestNeighbors(n_neighbors=2)
    features = filtered_df.drop(columns=['Lien_photo', 'name', 'question_1', 'question_2', 'question_3'])
    knn_model.fit(features)
    # Trouver le match le plus proche
    distances, indices = knn_model.kneighbors(user_input_encoded)
    nearest_neighbor_index = indices[0][1]
    nearest_neighbor_info = filtered_df.iloc[nearest_neighbor_index]
    # Afficher les résultats
    st.subheader("Votre match idéal :")
    st.write(f"Nom : {nearest_neighbor_info['name']}")
    st.write(f"Âge : {nearest_neighbor_info['age']}")
    st.write(f"Sexe : {'Homme' if nearest_neighbor_info['sex'] == 1 else 'Femme'}")
    st.write(f"Orientation : {'Straight' if nearest_neighbor_info['orientation_straight'] == 1 else ('Gay' if nearest_neighbor_info['orientation_gay'] == 1 else 'Bisexual')}")
    st.write(f"Type de corps : {nearest_neighbor_info['body_type']}")
    st.write(f"Régime alimentaire : {nearest_neighbor_info['diet']}")
    st.write(f"Éducation : {nearest_neighbor_info['education']}")
    st.write(f"Taille : {nearest_neighbor_info['height']}")
    st.write(f"Métier : {nearest_neighbor_info['job']}")
    st.write(f"Animaux de compagnie : {'Aime les animaux' if nearest_neighbor_info['pets'] == 1 else 'Naime pas les animaux'}")
    st.write(f"Fume : {'Non' if nearest_neighbor_info['smokes'] == 0 else 'Oui'}")
    st.image(f"{nearest_neighbor_info['Lien_photo']}")
# Ajouter un bouton pour générer le match idéal
if st.button("Trouve ton match idéal"):
    find_ideal_match()









