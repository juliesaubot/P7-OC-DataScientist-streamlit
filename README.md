# Projet 7 Openclassrooms

## Objectifs du projet
Ce projet a pour but de déployer un modèle via une API (FAST API) dans le Web via Heroku en utilisant un dashboard (Streamlit) pour présenter le travail de modélisation. Il comporte aussi un test unitaire à l'aide de Pytest. Le code source de l'API contient entre autres un calculateur de score pour une demande de crédit bancaire qui retourne au Dashboard la probabilité que le client puisse le rembourser, et qui indique donc si le crédit est accordé ou non. Cette partie est l'API qui a été build and deploy on Heroku.

## Découpage des dossiers
•	fichier main.py : fichier python de l'API FAST API

•	Procfile.py : Le Procfile contient le code qui donne les commandes pour indiquer quels fichiers doivent être exécutés par l’application lors de son ouverture. Ouvrez le fichier que vous avez créé et tapez cette ligne de code.

•	Le fichier requirements.txt : contient la liste des packages et des dépendances nécessaires à l’exécution de l’application Web. Vous trouverez ci-dessous un exemple de la façon dont vous devez remplir ce fichier.

•	Le fichier setup.sh : contient le script shell requis pour configurer l’environnement shell pour nos besoins. 

dossier data :

o	fichier X_test_reduit.csv contient les données client de test scalées

o	fichier X_test_reduit_unsc.csv contient les données client test réduit non scalés qui va servir a créer le fichier df_streamlit en rajoutant les informations de prediction

o	fichier df_streamlit.csv contient les données client non scalées et la réponse de la banque


•	model.pkl : contient le model de prédiction lightgbm choisi et entrainé sur les données d'entrainement test

•	fichier logo.png : logo de la société

•	fichier requirements.txt : stocke des informations sur toutes les bibliothèques, modules et packages utilisés lors du développement du projet

## Lien utiles

URL github du code de l'API : https://github.com/juliesaubot/Saubot_Julie_FASTAPI_102023

URL API déployé dans le Cloud : https://saubot-julie-fastapi-102023-5cca48d2a8d1.herokuapp.com/

URL dashboard déployé dans le Cloud : https://saubot-julie-dashboard-102023-00ee4bc4d00a.herokuapp.com/ 