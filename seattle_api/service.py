# service.py

import pandas as pd
import bentoml

from schemas import Input

MODEL_TAG = "energy_consumption_model:latest"


@bentoml.service(name="energy_consumption_service")
class EnergyConsumptionService:
    def __init__(self) -> None:
        # Chargement du pipeline sklearn (preprocess + modèle)
        self.model = bentoml.sklearn.load_model(MODEL_TAG)
        model_info = bentoml.models.get(MODEL_TAG)
        self.feature_names = model_info.custom_objects.get("feature_names", None)

    @bentoml.api
    def predict(self, input_data: Input) -> dict:
        """
        /predict
        JSON attendu :
        {
          "payload": {
            "PropertyGFATotal": ...,
            ...
          }
        }
        """
        # 1) récupérer l’objet BuildingData à l’intérieur du wrapper
        building = input_data.payload

        # 2) Pydantic -> dict avec alias (ex: "PropertyGFABuilding(s)")
        data_dict = building.model_dump(by_alias=True)
        df = pd.DataFrame([data_dict])

        # 3) Réordonner les colonnes comme à l'entraînement
        if self.feature_names is not None:
            df = df[self.feature_names]

        # 4) Prédiction
        y_pred = self.model.predict(df)
        raw = float(y_pred[0])
        y = max(raw, 0.0)  # clamp >= 0

        print(f"[DEBUG] raw_pred={raw} | clamped_to={y}")

        return {
            "prediction_SiteEnergyUse_kBtu": y,
            "input_used": data_dict,
        }
