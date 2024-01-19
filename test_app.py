import pytest
from main import app
from fastapi.testclient import TestClient

client = TestClient(app)



def test_hello():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World!"}



def test_predict():
    """
    fonction de test de la fonction predict de main.py
    On verifie si en sortie on a bien un status code de 200
    et un json de la forme demandé

    """

    url_endpoint = '/predict'
    response = client.get(url_endpoint)

    assert response.status_code == 200
    assert response.json() == {"model": "", "list_client_id": []}


def test_predict_get():

    url_endpoint = '/predict_get/100016'

    idx_client = 100016

    response = client.get(url_endpoint)

    assert response.status_code == 200

    data_response = response.json()

    #verifier qu'on a bien les champs Reponse et Proba_client
    assert 'Réponse' in data_response 
    assert 'Proba_client' in data_response

    #verifier qu'on a en retour les valeurs attendues
    assert data_response['Réponse'] == "Non"
    assert data_response['Proba_client'] == '0.4959097257760469'



