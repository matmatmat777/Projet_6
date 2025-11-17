
# Seattle Energy Consumption – Machine Learning API
Prédiction de la consommation énergétique annuelle d’un bâtiment (kWh/year)
Model : Lasso Regression • Déploiement : BentoML + Docker

## 1) Description du projet
Ce projet vise à prédire la consommation énergétique annuelle d’un bâtiment à Seattle à partir de ses caractéristiques (surface, type, âge, localisation, etc.).
L’objectif final est de fournir :

un modèle ML fiable (Lasso Regression),

une API REST permettant de faire des prédictions en temps réel,

un déploiement Dockerisé pour un usage reproductible.

## 2) Structure du projet
├── seattle_api/
│   ├── service.py             # API BentoML
│   ├── schemas.py             # Schéma Pydantic utilisé pour valider les entrées
│   ├── save_model.py          # Script pour sauvegarder le modèle dans BentoML
│   ├── bentofile.yaml         # Configuration du Bento (build)
│   ├── README.md              # Ce document
│   ├── requirements.txt
│   ├── client_example.py      # Exemple de requête Python
│   └── /models                # (géré automatiquement par BentoML)
└── notebooks/
    ├── projet_6_exploration.ipynb
    └── projet_6_modelisation.ipynb

Schéma du pipeline API:

                    ┌────────────────────────┐
                    │      Client (JSON)     │
                    └────────────┬───────────┘
                                 │
                                 ▼
                 ┌─────────────────────────────────┐
                 │      Pydantic (schemas.py)      │
                 │ - validation                    │
                 │ - typage                        │
                 │ - contraintes                   │
                 └─────────────────┬───────────────┘
                                   │
                                   ▼
                      ┌─────────────────────────┐
                      │  DataFrame pandas       │
                      │ - conversion dict -> df │
                      └─────────────┬───────────┘
                                    │
                                    ▼
                   ┌────────────────────────────────┐
                   │ Réordonnancement des features  │
                   │ df = df[self.feature_names]    │
                   └─────────────┬──────────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────────┐
                  │ Pipeline sklearn (dans le modèle)│
                  │ - imputation                     │
                  │ - scaling                        │
                  │ - encoding                       │
                  │ - modèle Lasso                   │
                  └──────────────────┬───────────────┘
                                     │
                                     ▼
                        ┌────────────────────┐
                        │   y_pred (float)   │
                        └──────────┬─────────┘
                                   │
                    (clamp négatif ▼)
                                   ▼
                     ┌────────────────────┐
                     │  y = max(0, pred)  │
                     └──────────┬─────────┘
                                │
                                ▼
                     ┌─────────────────────────┐
                     │    Réponse JSON API     │
                     └─────────────────────────┘

## 3) Installation
```
conda create -n p6 python=3.10
conda activate p6
pip install -r requirements.txt
bentoml --version
```

## 4) Entraînement du modèle & sauvegarde BentoML
Une fois le notebook exécuté et le modèle best_gb obtenu, on le sauvegarde :
  ```
  python save_model.py
  bentoml models list
  ```
exemple :
  ```
  energy_consumption_model:mf5qc3gcecopaoen
  ```

## 5) L’API BentoML : service.py
Elle attend un JSON validé par Pydantic (BuildingData) et retourne :

la prédiction (toujours ≥ 0),

les unités,

les données envoyées.

Lancer l’API en local :
  ```
  bentoml serve service:EnergyConsumptionService --reload

  ```
  Interface Swagger auto-générée :
   http://localhost:3000

## 6) Exemple d’appel API
```
import requests

url = "http://localhost:3000/predict"

example_payload = {
    "input_data": {
        "payload": {
            "PropertyGFATotal": 88434.0,
            "PropertyGFABuilding(s)": 88434.0,
            "PropertyGFAParking": 0.0,
            "NumberofBuildings": 1,
            "NumberofFloors": 12,
            "BuildingAge": 89,
            "PrimaryPropertyType": "Hotel",
            "LargestPropertyUseType": "Hotel",
            "CouncilDistrictCode": 7,
            "Neighborhood": "DOWNTOWN",
            "Latitude": 47.6129,
            "Longitude": -122.336,
            "HasParking": 0,
            "ParkingRatio": 0.0,
            "AreaPerFloor": 7369.5,
            "BuildingDensity": 0.000011,
            "GeoCluster": 2
        }
    }
}

response = requests.post(url, json=example_payload)
print(response.json())

```

## 7) Construction du Bento (build)
```
bentoml build
```
exemple d'output:
```
Successfully built Bento(tag="energy_consumption_service:n3hn5bwcdwctkoen")
```
## 8) Dockerisation & exécution
Build image Docker:
```
bentoml containerize energy_consumption_service:latest

```
Lancer le container:
```
docker run --rm -p 3000:3000 energy_consumption_service:<TAG>

```
## 9) Technologies utilisées
- Python 3.10
- Pandas / NumPy
- Scikit-Learn (Lasso Regression + Pipeline + ColumnTransformer)
- Pydantic v2
- BentoML 1.4
- Docker
- Jupyter Notebooks