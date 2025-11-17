# schemas.py

from pydantic import BaseModel, Field, field_validator

class BuildingData(BaseModel):
    PropertyGFATotal: float = Field(..., gt=0)
    PropertyGFABuilding_s: float = Field(
        ...,
        gt=0,
        alias="PropertyGFABuilding(s)",
    )
    PropertyGFAParking: float = Field(..., ge=0)
    NumberofBuildings: int = Field(..., ge=1)
    NumberofFloors: int = Field(..., ge=1)
    BuildingAge: int = Field(..., ge=0)
    PrimaryPropertyType: str
    LargestPropertyUseType: str
    CouncilDistrictCode: int = Field(..., ge=0)
    Neighborhood: str
    Latitude: float
    Longitude: float
    HasParking: int = Field(..., ge=0, le=1)
    ParkingRatio: float = Field(..., ge=0)
    AreaPerFloor: float = Field(..., ge=0)
    BuildingDensity: float = Field(..., ge=0)
    GeoCluster: int = Field(..., ge=0)

    @field_validator(
        "NumberofBuildings",
        "NumberofFloors",
        "BuildingAge",
        "CouncilDistrictCode",
        "GeoCluster",
        "HasParking",
    )
    @classmethod
    def non_negative_ints(cls, v: int) -> int:
        if v < 0:
            raise ValueError("La valeur doit être >= 0")
        return v


class Input(BaseModel):
    payload: BuildingData
