# client_example.py

import requests

URL = "http://localhost:3000/predict"

example_payload = {
    "input_data": {          # <-- wrapper attendu par le service
        "payload": {         # <-- wrapper défini dans schemas.Input
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

def main():
    resp = requests.post(URL, json=example_payload)
    print("Status code:", resp.status_code)
    try:
        print("Response JSON:")
        print(resp.json())
    except Exception:
        print("Raw response:")
        print(resp.text)


if __name__ == "__main__":
    main()
