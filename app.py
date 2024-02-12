import streamlit as st
from PIL import Image
import pandas as pd 
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from io import BytesIO
import pickle 
import seaborn as sns

import requests
import pandas as pd
import numpy as np
import shap

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config("Crédit banque", layout = "wide")
titre, image = st.columns([2,1])

with titre:
    st.title("Société 'Prêt à dépenser'")
    st.subheader('Réponse à la demande de crédit')
    st.write("""###### ***Dashboard réalisé par Julie Saubot pour le projet 7 du parcours de Datascientist Openclassrooms***""" )
with image:
    img = Image.open('logo.png')
    st.image(img, width = 250)

#Chargement des données du dataframe df pour liste des identifiants et matrice X (données imputées et scalées) pour shap
@st.cache_data(persist=True)
def load_data():
    #df = pd.read_csv('data/df_streamlit.gz')
    df = pd.read_csv('data/df_streamlit.csv')
    #X = pd.read_csv("data/X_test_reduit.gz", index_col = 0 )
    X = pd.read_csv("data/X_test_reduit_100.csv", index_col = 0 )
    return df, X

df, X = load_data()

#Chargement du modèle pour shap
def load_model():
    with open("model.pkl", 'rb') as pickle_in:
        model = pickle.load(pickle_in)
    print('Type du fichier :', type(model))
    return model

model=load_model()

#Liste de features de df:
list_feature_names = df.columns.to_list()
list_feature_names.remove('Demande_credit')
list_feature_names.remove('predict')
list_feature_names.remove('predict_proba')
list_feature_names.remove('SK_ID_CURR')

#Sélection du client
response = requests.get("https://saubot-julie-fastapi-102023-5cca48d2a8d1.herokuapp.com/predict")
#response = requests.get("http://127.0.0.1:8000/predict")
print(response)
if response:
    list_client_id = response.json()['list_client_id']
    list_client_id = sorted(list_client_id)
else:
    print("erreur web : ", response)

with st.sidebar:
    st.write("************************************************************************************")
    st.header("Client :")
    # Sélection du client
    id_filter = st.selectbox("""#### Sélectionnez l'identifiant de la personne :""", 
    list_client_id, 
    index=None, 
    placeholder='Cliquer pour sélectionner')
    st.write("")
    st.write("")
    st.write("************************************************************************************")

if id_filter==None:
    st.write("""#### ***En attente d'un numéro de client***""")
else:
    #Calcul de la réponse par rapport à un identifiant de client
    id = {"sk_id" : str(id_filter)}
    sk_id = id["sk_id"]
    response2 = requests.get(url = 'https://saubot-julie-fastapi-102023-5cca48d2a8d1.herokuapp.com/predict_get/', params = id)
    #response2 = requests.get("http://127.0.0.1:8000/predict_get/", params = id)
    if response2 :
        temp = response2.json()
        result = str(temp["Réponse"])
        #Scores du client
        seuil = str(((1-0.414)*100))
        proba = float(temp["Proba_client"])
        pourcentage = round(((1-proba)*100),2)
        pourc_str = str(pourcentage)


    #Affichage données client, score, jauge, feature importance locale dans colonnes :
    outer_cols = st.columns([1,1], gap = 'large')


    #Colonne 1 comporte 2 colonnes
    with outer_cols[0]:
        col1, col2 = st.columns([1,1], gap = 'large')

        with col1:
            
            #Affichage client :
            st.write("************************************************************************************")
            st.metric(label = """##### Numéro du client""", value = id_filter)
            

        with col2:
            #Affichage score et seuil
            st.write("************************************************************************************")
            st.metric(label = """##### Score du client :""", value = pourcentage)
            st.metric(label="""##### Valeur seuil :""", value=seuil)
            
            
        st.write('<p style="font-size:24px; color:grey;">Si le pourcentage du client dépasse le seuil indiqué, le client ne sera pas à risque, et il pourra obtenir son prêt.</p>',
                        unsafe_allow_html=True)
        st.write("************************************************************************************")
        
        #Jauge
        fig = go.Figure(go.Indicator(domain={'row': 0, 'column': 0},
        value=pourcentage,
        mode="gauge+number+delta",
        title={'text': "Visualisation du score du client", 'font_color':'black'},
        delta={'reference': float(seuil), "increasing": {"color": "green"}, 'decreasing': {"color": "red"}},
        gauge={'axis': {'range': [None, 100]},
        'steps': [{'range': [0, 100], 'color': "lightgray"}],
        'bar': {'color': "black"},
        'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': float(seuil)}}))
        
        fig.update_layout(paper_bgcolor="white", height=350)
        st.plotly_chart(fig, use_container_width=True)
            
        
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        
        #Affichage résultats du modèle de prédiction:             
        with st.expander("""##### Cliquer pour obtenir les résultats du modèle de prédiction"""):
            st.metric(label = "Probabilité calculée par le modèle", value = proba)
            st.metric(label='# et seuil de probabilité : ', value=0.414)
            st.write('<p style="font-size:24px; color:grey;">Si le score du client est en-dessous du seuil indiqué par le modèle de prédiction, le client ne sera pas à risque, et il pourra obtenir son prêt.</p>',
                        unsafe_allow_html=True)
            
            
        white_space = st.columns([1]) 

       
    # 2e colonne : réponse banque et informations du client
    with outer_cols[1]:
        st.write("************************************************************************************")

        st.subheader("Réponse de la banque ")
        
        if result == "Non" :
                st.write('<p style="font-size:26px; color:red;">Demande de prêt refusée, désolé(e)</p>', unsafe_allow_html=True)
        if result == "Oui" :
                st.write('<p style="font-size:26px; color:green;">Félicitations! Demande de prêt accordée !</p>', unsafe_allow_html=True)
        if result == "Erreur" :
                st.write('<p style="font-size:26px; color:orange;">Erreur</p>', unsafe_allow_html=True)
        st.write("")
        st.write("")
        st.write("************************************************************************************")
        st.write("")
        st.write("")
        
        #Informations du client
        st.write("""#### Informations du client numéro : """ + str(id_filter))
        req = requests.get(url = 'https://saubot-julie-fastapi-102023-5cca48d2a8d1.herokuapp.com/data_customer/', params = id)
        #req = requests.get("http://127.0.0.1:8000/data_customer/", params = id)
        print(req)
        info = req.json()['data']
        df_info = pd.DataFrame(info)
        df_info = df_info.set_index('SK_ID_CURR')
        df_info_tr = df_info.T
        #df_info_tr.columns = [str(id_filter)]
        st.dataframe(df_info_tr)

    

    st.write("************************************************************************************")
    #Features    
    st.header("Features")

    #Feature importance locale:
    shap.initjs()
    explainer = shap.Explainer(model.best_estimator_)
    shap_val = explainer.shap_values(X)
    ind = int(df.index[df["SK_ID_CURR"] == int(id_filter)].values)

    st.subheader("1. Feature importance locale pour le client numéro : "+str(id_filter))
    st.write('<p style="font-size:22px; color:white;">Importance des features pour le calcul du score du client :</p>', unsafe_allow_html=True)

    fig, ax = plt.subplots()
    X_approx = np.round(X,3)
    fig = shap.force_plot(base_value=explainer.expected_value[0],
                    shap_values=shap_val[0][ind,:],
                    features=X_approx.loc[[int(id_filter)]],
                    feature_names=list_feature_names,
                    matplotlib=True)
    plt.gcf().set_size_inches(30,6)
    plt.tight_layout()
    st.pyplot(fig)

    #Graphique de distribution de feature, avec et sans classe
    st.subheader("2. Distribution des features ")

    col1, col2 = st.columns([1, 2], gap='large') 
    with col1: #Configuration du graphique
        feat_select = st.selectbox("Sélectionnez une feature :", list_feature_names)
        
        classe = st.radio(
            "Choisir une classe :",
            ["Pas de classe",
                "Classe O : clients dont la demande est acceptée",
                "Classe 1 : clients dont la demande est refusée",
                ]
        )
        
        fig, ax = plt.subplots()
        n_bins = st.number_input(
            label='Choisir un nombre de bins',
            value=20
            )
        
        if classe == "Pas de classe" :
            t = 2
        if classe == "Classe O : clients dont la demande est acceptée":
            t = 0
            demande = "acceptée"
        if classe == "Classe 1 : clients dont la demande est refusée" :
            t = 1
            demande = "refusée"


    with col2: #Graphique de distribution
        
        if t == 2 :
            st.write("""#### Distribution de la feature sans classe : """ + feat_select)
            sns.histplot(df[feat_select], bins=n_bins)
            x_cust = df.loc[df['SK_ID_CURR'] == int(id_filter), feat_select].item()
            plt.axvline(x=x_cust, color="red", label="Position du client")
            plt.legend()
            buf = BytesIO()
            fig.savefig(buf, format="png")
            st.image(buf,  width=500)
        else:
            st.write("""#### Distribution de la feature : """ + feat_select + """ pour la classe des clients dont la demande est : """+demande)
            df_select = df[df['predict'] == t]
            sns.histplot(df_select [feat_select], bins=n_bins)
            if int(id_filter) in df_select['SK_ID_CURR'].to_list() :
                x_cust = df.loc[df['SK_ID_CURR'] == int(id_filter), feat_select].item()
                plt.axvline(x=x_cust, color="red", label="Position du client")
                plt.legend()
            buf = BytesIO()
            fig.savefig(buf, format="png")
            st.image(buf,  width=500)


    #Feature importance globale
    st.subheader("3. Feature importance globale")
    st.write('<p style="font-size:22px; color:white;">Importance des features pour la construction du modèle de prédiction :</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,4,1]) #2 colonnes vides pour resserrer le graphique

    with col2:
        shap.initjs()
        
        fig, ax = plt.subplots(figsize=(10, 10))
        shap.summary_plot(shap_val, features=X, feature_names=list_feature_names)
            
        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.image(buf,  width=600)


    #Analyse bivariée
    st.subheader('4. Analyse bivariée')
    col1, col2 = st.columns([1,3])
    with col1: #Configuration du graphique
        feat_select_plot1 = st.selectbox("""#### Sélectionnez une feature pour x :""", list_feature_names, key=1, index=3)
        feat_select_plot2 = st.selectbox("""#### Sélectionnez une feature pour y :""", list_feature_names, key=2)


    with col2: #Graphique d'analyse
        df_acc = df[df['Demande_credit']=="Accordée"]
        df_ref = df[df['Demande_credit']=="Refusée"]
        
        
        fig = go.Figure(data=go.Scatter(
            x=df_acc[ feat_select_plot1],
            y=df_acc[feat_select_plot2],
            mode="markers",
            name="Demande accordée",
            legendwidth=20,
            marker=dict(color="LightSkyBlue", size=8)))
        
        fig.add_trace(go.Scatter(
            x=df_ref[feat_select_plot1],
            y=df_ref[feat_select_plot2],
            mode="markers",
            name="Demande refusée",
            legendwidth=20,
            marker=dict(color="salmon", size=8)))
            
            
        fig.update_layout(
            xaxis_title=feat_select_plot1,
            yaxis_title=feat_select_plot2)
        
        
        
        # valeurs pour le client selectionné 
        var_x=df.loc[df["SK_ID_CURR"]== int(id_filter), feat_select_plot1]
        var_y=df.loc[df["SK_ID_CURR"]== int(id_filter), feat_select_plot2]
        var_z=df.loc[df["SK_ID_CURR"]== int(id_filter), "Demande_credit"]

        #j'ajoute la position du client sur le scatter plot
        if var_z.values[0] == "Refusée" : 
           fig.add_trace(go.Scatter(
            x = var_x,
            y = var_y,
            mode="markers",
            name="Client",
            legendwidth=20,
            marker_symbol = 'diamond',
            marker=dict(color="salmon", size=12),
            marker_line=dict(width=2, color="DarkSlateGrey")
            )
        )
        if var_z.values[0] == "Accordée":
          fig.add_trace(go.Scatter(
            x = var_x,
            y = var_y,
            mode="markers",
            name="Client",
            legendwidth=20,
            marker_symbol = 'diamond',
            marker=dict(color="LightSkyBlue", size=12),
            marker_line=dict(width=2, color="DarkSlateGrey")
            )
        )


        fig.update_layout(legend=dict(font=dict(size= 20)), height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col1: # on ecrit sur la colonne 1 les informations du client

        st.write("************************************************************************************")
        st.write('<p style="font-size:18px; color:white;">Valeurs du client :</p>', unsafe_allow_html=True)

        st.write(feat_select_plot1, "= ", var_x.values[0])
        st.write(feat_select_plot2, "= ", var_y.values[0])
        st.write("La demande de credit du client est : ", var_z.values[0])

#streamlit run app.py