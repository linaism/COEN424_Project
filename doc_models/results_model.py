from pydantic import BaseModel, Field 
from datetime import datetime

class Results(BaseModel):
    TIMESTAMP: datetime
    CODE_GENDER: str
    FLAG_OWN_CAR: str
    FLAG_OWN_REALTY: str
    CNT_CHILDREN: float
    AMT_INCOME_TOTAL: float
    NAME_INCOME_TYPE: str
    NAME_EDUCATION_TYPE: str
    NAME_FAMILY_STATUS: str
    NAME_HOUSING_TYPE: str
    FLAG_MOBIL: str
    FLAG_WORK_PHONE: str
    FLAG_PHONE: str
    FLAG_EMAIL: str
    OCCUPATION_TYPE: str
    CNT_FAM_MEMBERS: float
    MONTHS_BALANCE: float
    PREDICTION: int

    def to_json(self): 
        return self.__dict__
    