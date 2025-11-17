# save_model.py

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

import bentoml

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

TARGET = "SiteEnergyUse(kBtu)"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruit les principales features dérivées si elles ne sont pas déjà présentes.
    """
    df = df.copy()

    # BuildingAge
    if "BuildingAge" not in df.columns and "YearBuilt" in df.columns:
        df["BuildingAge"] = 2016 - df["YearBuilt"]

    # HasParking
    if "HasParking" not in df.columns and "PropertyGFAParking" in df.columns:
        df["HasParking"] = (df["PropertyGFAParking"] > 0).astype(int)

    # ParkingRatio
    if "ParkingRatio" not in df.columns and {"PropertyGFAParking", "PropertyGFATotal"}.issubset(df.columns):
        total = df["PropertyGFATotal"].replace(0, np.nan)
        df["ParkingRatio"] = df["PropertyGFAParking"] / total

    # AreaPerFloor
    if "AreaPerFloor" not in df.columns and {"PropertyGFABuilding(s)", "NumberofFloors"}.issubset(df.columns):
        floors = df["NumberofFloors"].replace(0, np.nan)
        df["AreaPerFloor"] = df["PropertyGFABuilding(s)"] / floors

    # BuildingDensity
    if "BuildingDensity" not in df.columns and {"NumberofBuildings", "PropertyGFATotal"}.issubset(df.columns):
        total = df["PropertyGFATotal"].replace(0, np.nan)
        df["BuildingDensity"] = df["NumberofBuildings"] / total

    # GeoCluster : on suppose qu'il est déjà dans df_model.csv (issu du notebook)
    return df


def main():
    # 1) Charger les données
    df = pd.read_csv("df_model_bento.csv")
    df = build_features(df)

    # 2) Features utilisées par le modèle (doivent exister après build_features)
    features_finales = [
        "PropertyGFATotal",
        "PropertyGFABuilding(s)",
        "PropertyGFAParking",
        "NumberofBuildings",
        "NumberofFloors",
        "BuildingAge",
        "PrimaryPropertyType",
        "LargestPropertyUseType",
        "CouncilDistrictCode",
        "Neighborhood",
        "Latitude",
        "Longitude",
        "HasParking",
        "ParkingRatio",
        "AreaPerFloor",
        "BuildingDensity",
        "GeoCluster",
    ]

    # On enlève les lignes incomplètes
    df = df.dropna(subset=[TARGET] + features_finales)

    X = df[features_finales].copy()
    y = df[TARGET].copy()

    # 3) Split train / test (principalement pour contrôle)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    # 4) Colonnes numériques / catégorielles
    num_cols = [
        "PropertyGFATotal",
        "PropertyGFABuilding(s)",
        "PropertyGFAParking",
        "NumberofBuildings",
        "NumberofFloors",
        "BuildingAge",
        "Latitude",
        "Longitude",
        "HasParking",
        "ParkingRatio",
        "AreaPerFloor",
        "BuildingDensity",
    ]

    cat_cols = [
        "PrimaryPropertyType",
        "LargestPropertyUseType",
        "Neighborhood",
        "GeoCluster",
        "CouncilDistrictCode",
    ]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ])

    # 5) Gradient Boosting optimisé
    best_gb_model = GradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=2,
        n_estimators=300,
        subsample=1.0,
        random_state=RANDOM_STATE,
    )

    pipe_gb = Pipeline([
        ("prepro", preprocessor),
        ("model", best_gb_model),
    ])

    pipe_gb.fit(X_train, y_train)

    feature_names = X_train.columns.tolist()

    bento_model = bentoml.sklearn.save_model(
        "energy_consumption_model",
        pipe_gb,
        custom_objects={"feature_names": feature_names},
    )

    print("✅ Modèle Gradient Boosting sauvegardé. Tag :", bento_model.tag)


if __name__ == "__main__":
    main()
