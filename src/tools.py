from enum import Enum

class FrequencyType(Enum):
    DAILY = 252 # 252 jours de trading dans une année
    WEEKLY = 52 # 52 semaines dans une année
    MONTHLY = 12 # 12 mois dans une année
    QUARTERLY = 4 # 4 trimestres dans une année
    ANNUALLY = 1 # 1 an
