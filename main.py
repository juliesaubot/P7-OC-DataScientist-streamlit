from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import pickle
import json
import requests
from pydantic import BaseModel
import uvicorn 

app = FastAPI()

@app.get("/")
def hello():
    return JSONResponse({'message': "Hello World!"})

#Load data
#Fichier à prédire X test qui a subi un préprocessing comme le fichier d'entraînement
#X_test = pd.read_csv("data/X_test_reduit.gz", index_col = 0)
X_test = pd.read_csv("data/X_test_reduit_100.csv", index_col = 0)

#on importe le dataframe à predire X test sans scaling pour le dashboard streamlit
#X_test_unscal = pd.read_csv("data/X_test_reduit_unsc.gz", index_col = 0)
X_test_unscal = pd.read_csv("data/X_test_reduit_unsc_100.csv", index_col = 0)

#on enregistre les numéros clients
num_client = X_test.index.unique()
print(num_client)

#Présentation du DF à prédire
print('Dataframe de données à prédire :')
print('X_test shape : ', X_test.shape)

#on charge le modèle final
with open("model.pkl", 'rb') as pickle_in:
    print("utilisation du model lightgbm")
    grid_lgbm = pickle.load(pickle_in)
    
#Application du modèle lgbm avec seuil calculé précédemment et détermination de la probabilité
y_predict_prob = grid_lgbm.predict_proba(X_test)[:,1]

#Seuil métier optimal : 0.414
y_predict = (grid_lgbm.predict_proba(X_test)[:,1] >= 0.414).astype(int)


#Ajout de la probabilité et de la prédiction au dataframe X test non scalé 
df = X_test_unscal.copy()
df = df.reset_index()
df['predict_proba'] = y_predict_prob
df['predict'] = y_predict
df['Demande_credit'] = ''
df.loc[df['predict'] == 0, 'Demande_credit'] = 'Accordée'
df.loc[df['predict'] == 1, 'Demande_credit'] = 'Refusée'

#Vérifications 
print("Vérification que le modèle prédit les targets dans les mêmes proportions que dans le notebook")
print(df['predict'].value_counts(normalize=True))

print("Résultats dans le notebook :")
print("0    0.61895")
print("1    0.38105")

#dataframe reduit pour faire un test de deploiement sur Heroku
df.to_csv('data/df_streamlit.csv',index=False)

#Sauvegarde de df pour streamlit
#df.to_csv('data/df_streamlit.gz', compression='gzip',index=False)


@app.get('/predict')
async def predict():
    """
    Returns
    liste des clients dans le fichier
    """
    return JSONResponse({"model": "lgbmc10_GridCV","list_client_id" : list(num_client.astype(str))})


@app.get("/predict_get/{sk_id}")
async def predict_get(sk_id: str):
    """
    Parameters
    sk_id : numero de client
    Returns
    JSON de la reponse de la prediction si le credit a été accepté ou non
    et de la probabilité du client
    """
    id = int(sk_id)
    result = df.loc[df["SK_ID_CURR"] == id, 'Demande_credit']
    proba_str = str(df.loc[df['SK_ID_CURR'] == id, 'predict_proba'].values[0])
    if result.values == "Refusée":
        return JSONResponse({"Réponse" : "Non", "Proba_client" : proba_str})
    if result.values == "Accordée":
        return JSONResponse({"Réponse" : "Oui", "Proba_client" : proba_str})
    else:
        return JSONResponse({"Réponse" : "Erreur"})

@app.post("/data_customer/{sk_id}")
async def data_customer(sk_id: str):
    """
    Parameters 
    sk_id : numero de client

    Returns 
    toutes les valeurs des variables du client choisi : information client

    """
    id = int(sk_id)
    # Get the personal data customer (pd.Series)
    X_cust = df.loc[df["SK_ID_CURR"] == id, :]
    #Convert the pd.Series (df row) of customer's data to JSON
    X_cust_json = json.loads(X_cust.to_json())
    # Return the cleaned data
    return JSONResponse({'status': 'ok','data': X_cust_json})


if __name__ == '__main__':
    uvicorn.run("main:app",host="127.0.0.1", port=8000,debug=True)







