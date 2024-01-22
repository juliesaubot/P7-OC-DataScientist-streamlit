# Projet 7 Openclassrooms

## Objectifs du projet
Ce projet a pour but de déployer un modèle via une API (FAST API) dans le Web via Heroku en utilisant un dashboard (Streamlit) pour présenter le travail de modélisation. Il comporte aussi un test unitaire à l'aide de Pytest. Le code source de l'API contient entre autres un calculateur de score pour une demande de crédit bancaire qui retourne au Dashboard la probabilité que le client puisse le rembourser, et qui indique donc si le crédit est accordé ou non. Cette partie est l'API qui a été build and deploy on Heroku.

## Découpage des dossiers
•	fichier main.py : fichier python de l'API FAST API
•	fichier dashboard.py :
•	fichier test_app.py : fichier contenant les tests unitaires avec Pytest
•	dossier data :
o	fichier X_test_reduit.csv contient les données client de test scalées
o	fichier X_test_reduit_unsc.csv contient les données client test réduit non scalés qui va servir a créer le fichier df_streamlit en rajoutant les informations de prediction
o	fichier df_streamlit.csv contient les données client non scalées et la réponse de la banque
•	dossier mlflow_model :
o	model.pkl : contient le model de prédiction lightgbm choisi et entrainé sur les données d'entrainement test
•	fichier logo.png : logo de la société
•	fichier requirements.txt : stocke des informations sur toutes les bibliothèques, modules et packages utilisés lors du développement du projet