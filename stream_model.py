import streamlit as st
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import warnings
st.set_page_config(page_title="Crime Prediction Using ML", page_icon=":guardsman:")
def run():

    Header_container = st.container()
    with Header_container:
    
        st.title("Crime Prediction Using Machine Learning")
        st.write("""Crime prediction is an emerging field of study that uses data analysis and machine learning to
predict criminal behavior and prevent crime.

Predictive models use advanced algorithms to identify patterns in criminal behavior and take into
account external factors to make more accurate predictions.

Predictive models can help law enforcement agencies allocate resources, prevent crime, and
improve public safety, as well as help investigators identify potential suspects and gather evidence
more efficiently.

Researchers and practitioners must be aware of ethical and legal concerns surrounding crime
prediction, such as potential biases and privacy and civil liberties, and address them through
transparent and ethical practices.""")
    Dataset_container = st.container()
    with Dataset_container:
        st.title("Training Dataset Information")
        st.write("For more information click on the following link :")
        link = '[Dataset Info](https://archive.ics.uci.edu/ml/datasets/communities+and+crime)'
        st.markdown(link, unsafe_allow_html=True)

        st.write("""Attribute Information: \n
Attribute Information: (122 predictive, 5 non-predictive, 1 goal) \n
-- state: US state (by number) - not counted as predictive above, but if considered, should be consided nominal (nominal)\n
-- county: numeric code for county - not predictive, and many missing values (numeric)\n
-- community: numeric code for community - not predictive and many missing values (numeric)\n
-- communityname: community name - not predictive - for information only (string)\n
-- fold: fold number for non-random 10 fold cross validation, potentially useful for debugging, paired tests - not predictive (numeric)\n
-- population: population for community: (numeric - decimal)\n
-- householdsize: mean people per household (numeric - decimal)\n
-- racepctblack: percentage of population that is african american (numeric - decimal)\n
-- racePctWhite: percentage of population that is caucasian (numeric - decimal)\n
-- racePctAsian: percentage of population that is of asian heritage (numeric - decimal)\n
-- racePctHisp: percentage of population that is of hispanic heritage (numeric - decimal)\n
-- agePct12t21: percentage of population that is 12-21 in age (numeric - decimal)\n
-- agePct12t29: percentage of population that is 12-29 in age (numeric - decimal)\n
-- agePct16t24: percentage of population that is 16-24 in age (numeric - decimal)\n
-- agePct65up: percentage of population that is 65 and over in age (numeric - decimal)\n
-- numbUrban: number of people living in areas classified as urban (numeric - decimal)\n
-- pctUrban: percentage of people living in areas classified as urban (numeric - decimal)\n
-- medIncome: median household income (numeric - decimal)\n
-- pctWWage: percentage of households with wage or salary income in 1989 (numeric - decimal)\n
-- pctWFarmSelf: percentage of households with farm or self employment income in 1989 (numeric - decimal)\n
-- pctWInvInc: percentage of households with investment / rent income in 1989 (numeric - decimal)\n
-- pctWSocSec: percentage of households with social security income in 1989 (numeric - decimal)\n
-- pctWPubAsst: percentage of households with public assistance income in 1989 (numeric - decimal)\n
-- pctWRetire: percentage of households with retirement income in 1989 (numeric - decimal)\n
-- medFamInc: median family income (differs from household income for non-family households) (numeric - decimal)\n
-- perCapInc: per capita income (numeric - decimal)\n
-- whitePerCap: per capita income for caucasians (numeric - decimal)\n
-- blackPerCap: per capita income for african americans (numeric - decimal)\n
-- indianPerCap: per capita income for native americans (numeric - decimal)\n
-- AsianPerCap: per capita income for people with asian heritage (numeric - decimal)\n
-- OtherPerCap: per capita income for people with 'other' heritage (numeric - decimal)\n
-- HispPerCap: per capita income for people with hispanic heritage (numeric - decimal)\n
-- NumUnderPov: number of people under the poverty level (numeric - decimal)\n
-- PctPopUnderPov: percentage of people under the poverty level (numeric - decimal)\n
-- PctLess9thGrade: percentage of people 25 and over with less than a 9th grade education (numeric - decimal)\n
-- PctNotHSGrad: percentage of people 25 and over that are not high school graduates (numeric - decimal)\n
-- PctBSorMore: percentage of people 25 and over with a bachelors degree or higher education (numeric - decimal)\n
-- PctUnemployed: percentage of people 16 and over, in the labor force, and unemployed (numeric - decimal)\n
-- PctEmploy: percentage of people 16 and over who are employed (numeric - decimal)\n
-- PctEmplManu: percentage of people 16 and over who are employed in manufacturing (numeric - decimal)\n
-- PctEmplProfServ: percentage of people 16 and over who are employed in professional services (numeric - decimal)\n
-- PctOccupManu: percentage of people 16 and over who are employed in manufacturing (numeric - decimal) ########\n
-- PctOccupMgmtProf: percentage of people 16 and over who are employed in management or professional occupations (numeric - decimal)\n
-- MalePctDivorce: percentage of males who are divorced (numeric - decimal)\n
-- MalePctNevMarr: percentage of males who have never married (numeric - decimal)\n
-- FemalePctDiv: percentage of females who are divorced (numeric - decimal)\n
-- TotalPctDiv: percentage of population who are divorced (numeric - decimal)\n
-- PersPerFam: mean number of people per family (numeric - decimal)\n
-- PctFam2Par: percentage of families (with kids) that are headed by two parents (numeric - decimal)\n
-- PctKids2Par: percentage of kids in family housing with two parents (numeric - decimal)\n
-- PctYoungKids2Par: percent of kids 4 and under in two parent households (numeric - decimal)\n
-- PctTeen2Par: percent of kids age 12-17 in two parent households (numeric - decimal)\n
-- PctWorkMomYoungKids: percentage of moms of kids 6 and under in labor force (numeric - decimal)\n
-- PctWorkMom: percentage of moms of kids under 18 in labor force (numeric - decimal)\n
-- NumIlleg: number of kids born to never married (numeric - decimal)\n
-- PctIlleg: percentage of kids born to never married (numeric - decimal)\n
-- NumImmig: total number of people known to be foreign born (numeric - decimal)\n
-- PctImmigRecent: percentage of _immigrants_ who immigated within last 3 years (numeric - decimal)\n
-- PctImmigRec5: percentage of _immigrants_ who immigated within last 5 years (numeric - decimal)\n
-- PctImmigRec8: percentage of _immigrants_ who immigated within last 8 years (numeric - decimal)\n
-- PctImmigRec10: percentage of _immigrants_ who immigated within last 10 years (numeric - decimal)\n
-- PctRecentImmig: percent of _population_ who have immigrated within the last 3 years (numeric - decimal)\n
-- PctRecImmig5: percent of _population_ who have immigrated within the last 5 years (numeric - decimal)\n
-- PctRecImmig8: percent of _population_ who have immigrated within the last 8 years (numeric - decimal)\n
-- PctRecImmig10: percent of _population_ who have immigrated within the last 10 years (numeric - decimal)\n
-- PctSpeakEnglOnly: percent of people who speak only English (numeric - decimal)\n
-- PctNotSpeakEnglWell: percent of people who do not speak English well (numeric - decimal)\n
-- PctLargHouseFam: percent of family households that are large (6 or more) (numeric - decimal)\n
-- PctLargHouseOccup: percent of all occupied households that are large (6 or more people) (numeric - decimal)\n
-- PersPerOccupHous: mean persons per household (numeric - decimal)\n
-- PersPerOwnOccHous: mean persons per owner occupied household (numeric - decimal)\n
-- PersPerRentOccHous: mean persons per rental household (numeric - decimal)\n
-- PctPersOwnOccup: percent of people in owner occupied households (numeric - decimal)\n
-- PctPersDenseHous: percent of persons in dense housing (more than 1 person per room) (numeric - decimal)\n
-- PctHousLess3BR: percent of housing units with less than 3 bedrooms (numeric - decimal)\n
-- MedNumBR: median number of bedrooms (numeric - decimal)\n
-- HousVacant: number of vacant households (numeric - decimal)\n
-- PctHousOccup: percent of housing occupied (numeric - decimal)\n
-- PctHousOwnOcc: percent of households owner occupied (numeric - decimal)\n
-- PctVacantBoarded: percent of vacant housing that is boarded up (numeric - decimal)\n
-- PctVacMore6Mos: percent of vacant housing that has been vacant more than 6 months (numeric - decimal)\n
-- MedYrHousBuilt: median year housing units built (numeric - decimal)\n
-- PctHousNoPhone: percent of occupied housing units without phone (in 1990, this was rare!) (numeric - decimal)\n
-- PctWOFullPlumb: percent of housing without complete plumbing facilities (numeric - decimal)\n
-- OwnOccLowQuart: owner occupied housing - lower quartile value (numeric - decimal)\n
-- OwnOccMedVal: owner occupied housing - median value (numeric - decimal)\n
-- OwnOccHiQuart: owner occupied housing - upper quartile value (numeric - decimal)\n
-- RentLowQ: rental housing - lower quartile rent (numeric - decimal)\n
-- RentMedian: rental housing - median rent (Census variable H32B from file STF1A) (numeric - decimal)\n
-- RentHighQ: rental housing - upper quartile rent (numeric - decimal)\n
-- MedRent: median gross rent (Census variable H43A from file STF3A - includes utilities) (numeric - decimal)\n
-- MedRentPctHousInc: median gross rent as a percentage of household income (numeric - decimal)\n
-- MedOwnCostPctInc: median owners cost as a percentage of household income - for owners with a mortgage (numeric - decimal)\n
-- MedOwnCostPctIncNoMtg: median owners cost as a percentage of household income - for owners without a mortgage (numeric - decimal)\n
-- NumInShelters: number of people in homeless shelters (numeric - decimal)\n
-- NumStreet: number of homeless people counted in the street (numeric - decimal)\n
-- PctForeignBorn: percent of people foreign born (numeric - decimal)\n
-- PctBornSameState: percent of people born in the same state as currently living (numeric - decimal)\n
-- PctSameHouse85: percent of people living in the same house as in 1985 (5 years before) (numeric - decimal)\n
-- PctSameCity85: percent of people living in the same city as in 1985 (5 years before) (numeric - decimal)\n
-- PctSameState85: percent of people living in the same state as in 1985 (5 years before) (numeric - decimal)\n
-- LemasSwornFT: number of sworn full time police officers (numeric - decimal)\n
-- LemasSwFTPerPop: sworn full time police officers per 100K population (numeric - decimal)\n
-- LemasSwFTFieldOps: number of sworn full time police officers in field operations (on the street as opposed to administrative etc) (numeric - decimal)\n
-- LemasSwFTFieldPerPop: sworn full time police officers in field operations (on the street as opposed to administrative etc) per 100K population (numeric - decimal)\n
-- LemasTotalReq: total requests for police (numeric - decimal)\n
-- LemasTotReqPerPop: total requests for police per 100K popuation (numeric - decimal)\n
-- PolicReqPerOffic: total requests for police per police officer (numeric - decimal)\n
-- PolicPerPop: police officers per 100K population (numeric - decimal)\n
-- RacialMatchCommPol: a measure of the racial match between the community and the police force. High values indicate proportions in community and police force are similar (numeric - decimal)\n
-- PctPolicWhite: percent of police that are caucasian (numeric - decimal)\n
-- PctPolicBlack: percent of police that are african american (numeric - decimal)\n
-- PctPolicHisp: percent of police that are hispanic (numeric - decimal)\n
-- PctPolicAsian: percent of police that are asian (numeric - decimal)\n
-- PctPolicMinor: percent of police that are minority of any kind (numeric - decimal)\n
-- OfficAssgnDrugUnits: number of officers assigned to special drug units (numeric - decimal)\n
-- NumKindsDrugsSeiz: number of different kinds of drugs seized (numeric - decimal)\n
-- PolicAveOTWorked: police average overtime worked (numeric - decimal)\n
-- LandArea: land area in square miles (numeric - decimal)\n
-- PopDens: population density in persons per square mile (numeric - decimal)\n
-- PctUsePubTrans: percent of people using public transit for commuting (numeric - decimal)\n
-- PolicCars: number of police cars (numeric - decimal)\n
-- PolicOperBudg: police operating budget (numeric - decimal)\n
-- LemasPctPolicOnPatr: percent of sworn full time police officers on patrol (numeric - decimal)\n
-- LemasGangUnitDeploy: gang unit deployed (numeric - decimal - but really ordinal - 0 means NO, 1 means YES, 0.5 means Part Time)\n
-- LemasPctOfficDrugUn: percent of officers assigned to drug units (numeric - decimal)\n
-- PolicBudgPerPop: police operating budget per population (numeric - decimal)\n
-- ViolentCrimesPerPop: total number of violent crimes per 100K popuation (numeric - decimal) GOAL attribute (to be predicted)\n

Summary Statistics:\n
Min Max Mean SD Correl Median Mode Missing\n
population 0 1 0.06 0.13 0.37 0.02 0.01 0\n
householdsize 0 1 0.46 0.16 -0.03 0.44 0.41 0\n
racepctblack 0 1 0.18 0.25 0.63 0.06 0.01 0\n
racePctWhite 0 1 0.75 0.24 -0.68 0.85 0.98 0\n
racePctAsian 0 1 0.15 0.21 0.04 0.07 0.02 0\n
racePctHisp 0 1 0.14 0.23 0.29 0.04 0.01 0\n
agePct12t21 0 1 0.42 0.16 0.06 0.4 0.38 0\n
agePct12t29 0 1 0.49 0.14 0.15 0.48 0.49 0\n
agePct16t24 0 1 0.34 0.17 0.10 0.29 0.29 0\n
agePct65up 0 1 0.42 0.18 0.07 0.42 0.47 0\n
numbUrban 0 1 0.06 0.13 0.36 0.03 0 0\n
pctUrban 0 1 0.70 0.44 0.08 1 1 0\n
medIncome 0 1 0.36 0.21 -0.42 0.32 0.23 0\n
pctWWage 0 1 0.56 0.18 -0.31 0.56 0.58 0\n
pctWFarmSelf 0 1 0.29 0.20 -0.15 0.23 0.16 0\n
pctWInvInc 0 1 0.50 0.18 -0.58 0.48 0.41 0\n
pctWSocSec 0 1 0.47 0.17 0.12 0.475 0.56 0\n
pctWPubAsst 0 1 0.32 0.22 0.57 0.26 0.1 0\n
pctWRetire 0 1 0.48 0.17 -0.10 0.47 0.44 0\n
medFamInc 0 1 0.38 0.20 -0.44 0.33 0.25 0\n
perCapInc 0 1 0.35 0.19 -0.35 0.3 0.23 0\n
whitePerCap 0 1 0.37 0.19 -0.21 0.32 0.3 0\n
blackPerCap 0 1 0.29 0.17 -0.28 0.25 0.18 0\n
indianPerCap 0 1 0.20 0.16 -0.09 0.17 0 0\n
AsianPerCap 0 1 0.32 0.20 -0.16 0.28 0.18 0\n
OtherPerCap 0 1 0.28 0.19 -0.13 0.25 0 1\n
HispPerCap 0 1 0.39 0.18 -0.24 0.345 0.3 0\n
NumUnderPov 0 1 0.06 0.13 0.45 0.02 0.01 0\n
PctPopUnderPov 0 1 0.30 0.23 0.52 0.25 0.08 0\n
PctLess9thGrade 0 1 0.32 0.21 0.41 0.27 0.19 0\n
PctNotHSGrad 0 1 0.38 0.20 0.48 0.36 0.39 0\n
PctBSorMore 0 1 0.36 0.21 -0.31 0.31 0.18 0\n
PctUnemployed 0 1 0.36 0.20 0.50 0.32 0.24 0\n
PctEmploy 0 1 0.50 0.17 -0.33 0.51 0.56 0\n
PctEmplManu 0 1 0.40 0.20 -0.04 0.37 0.26 0\n
PctEmplProfServ 0 1 0.44 0.18 -0.07 0.41 0.36 0\n
PctOccupManu 0 1 0.39 0.20 0.30 0.37 0.32 0\n
PctOccupMgmtProf 0 1 0.44 0.19 -0.34 0.4 0.36 0\n
MalePctDivorce 0 1 0.46 0.18 0.53 0.47 0.56 0\n
MalePctNevMarr 0 1 0.43 0.18 0.30 0.4 0.38 0\n
FemalePctDiv 0 1 0.49 0.18 0.56 0.5 0.54 0\n
TotalPctDiv 0 1 0.49 0.18 0.55 0.5 0.57 0\n
PersPerFam 0 1 0.49 0.15 0.14 0.47 0.44 0\n
PctFam2Par 0 1 0.61 0.20 -0.71 0.63 0.7 0\n
PctKids2Par 0 1 0.62 0.21 -0.74 0.64 0.72 0\n
PctYoungKids2Par 0 1 0.66 0.22 -0.67 0.7 0.91 0\n
PctTeen2Par 0 1 0.58 0.19 -0.66 0.61 0.6 0\n
PctWorkMomYoungKids 0 1 0.50 0.17 -0.02 0.51 0.51 0\n
PctWorkMom 0 1 0.53 0.18 -0.15 0.54 0.57 0\n
NumIlleg 0 1 0.04 0.11 0.47 0.01 0 0\n
PctIlleg 0 1 0.25 0.23 0.74 0.17 0.09 0\n
NumImmig 0 1 0.03 0.09 0.29 0.01 0 0\n
PctImmigRecent 0 1 0.32 0.22 0.17 0.29 0 0\n
PctImmigRec5 0 1 0.36 0.21 0.22 0.34 0 0\n
PctImmigRec8 0 1 0.40 0.20 0.25 0.39 0.26 0\n
PctImmigRec10 0 1 0.43 0.19 0.29 0.43 0.43 0\n
PctRecentImmig 0 1 0.18 0.24 0.23 0.09 0.01 0\n
PctRecImmig5 0 1 0.18 0.24 0.25 0.08 0.02 0\n
PctRecImmig8 0 1 0.18 0.24 0.25 0.09 0.02 0\n
PctRecImmig10 0 1 0.18 0.23 0.26 0.09 0.02 0\n
PctSpeakEnglOnly 0 1 0.79 0.23 -0.24 0.87 0.96 0\n
PctNotSpeakEnglWell 0 1 0.15 0.22 0.30 0.06 0.03 0\n
PctLargHouseFam 0 1 0.27 0.20 0.38 0.2 0.17 0\n
PctLargHouseOccup 0 1 0.25 0.19 0.29 0.19 0.19 0\n
PersPerOccupHous 0 1 0.46 0.17 -0.04 0.44 0.37 0\n
PersPerOwnOccHous 0 1 0.49 0.16 -0.12 0.48 0.45 0\n
PersPerRentOccHous 0 1 0.40 0.19 0.25 0.36 0.32 0\n
PctPersOwnOccup 0 1 0.56 0.20 -0.53 0.56 0.54 0\n
PctPersDenseHous 0 1 0.19 0.21 0.45 0.11 0.06 0\n
PctHousLess3BR 0 1 0.50 0.17 0.47 0.51 0.53 0\n
MedNumBR 0 1 0.31 0.26 -0.36 0.5 0.5 0\n
HousVacant 0 1 0.08 0.15 0.42 0.03 0.01 0\n
PctHousOccup 0 1 0.72 0.19 -0.32 0.77 0.88 0\n
PctHousOwnOcc 0 1 0.55 0.19 -0.47 0.54 0.52 0\n
PctVacantBoarded 0 1 0.20 0.22 0.48 0.13 0 0\n
PctVacMore6Mos 0 1 0.43 0.19 0.02 0.42 0.44 0\n
MedYrHousBuilt 0 1 0.49 0.23 -0.11 0.52 0 0\n
PctHousNoPhone 0 1 0.26 0.24 0.49 0.185 0.01 0\n
PctWOFullPlumb 0 1 0.24 0.21 0.36 0.19 0 0\n
OwnOccLowQuart 0 1 0.26 0.22 -0.21 0.18 0.09 0\n
OwnOccMedVal 0 1 0.26 0.23 -0.19 0.17 0.08 0\n
OwnOccHiQuart 0 1 0.27 0.24 -0.17 0.18 0.08 0\n
RentLowQ 0 1 0.35 0.22 -0.25 0.31 0.13 0\n
RentMedian 0 1 0.37 0.21 -0.24 0.33 0.19 0\n
RentHighQ 0 1 0.42 0.25 -0.23 0.37 1 0\n
MedRent 0 1 0.38 0.21 -0.24 0.34 0.17 0\n
MedRentPctHousInc 0 1 0.49 0.17 0.33 0.48 0.4 0\n
MedOwnCostPctInc 0 1 0.45 0.19 0.06 0.45 0.41 0\n
MedOwnCostPctIncNoMtg 0 1 0.40 0.19 0.05 0.37 0.24 0\n
NumInShelters 0 1 0.03 0.10 0.38 0 0 0\n
NumStreet 0 1 0.02 0.10 0.34 0 0 0\n
PctForeignBorn 0 1 0.22 0.23 0.19 0.13 0.03 0\n
PctBornSameState 0 1 0.61 0.20 -0.08 0.63 0.78 0\n
PctSameHouse85 0 1 0.54 0.18 -0.16 0.54 0.59 0\n
PctSameCity85 0 1 0.63 0.20 0.08 0.67 0.74 0\n
PctSameState85 0 1 0.65 0.20 -0.02 0.7 0.79 0\n
LemasSwornFT 0 1 0.07 0.14 0.34 0.02 0.02 1675\n
LemasSwFTPerPop 0 1 0.22 0.16 0.15 0.18 0.2 1675\n
LemasSwFTFieldOps 0 1 0.92 0.13 -0.33 0.97 0.98 1675\n
LemasSwFTFieldPerPop 0 1 0.25 0.16 0.16 0.21 0.19 1675\n
LemasTotalReq 0 1 0.10 0.16 0.35 0.04 0.02 1675\n
LemasTotReqPerPop 0 1 0.22 0.16 0.27 0.17 0.14 1675\n
PolicReqPerOffic 0 1 0.34 0.20 0.17 0.29 0.23 1675\n
PolicPerPop 0 1 0.22 0.16 0.15 0.18 0.2 1675\n
RacialMatchCommPol 0 1 0.69 0.23 -0.46 0.74 0.78 1675\n
PctPolicWhite 0 1 0.73 0.22 -0.44 0.78 0.72 1675\n
PctPolicBlack 0 1 0.22 0.24 0.54 0.12 0 1675\n
PctPolicHisp 0 1 0.13 0.20 0.12 0.06 0 1675\n
PctPolicAsian 0 1 0.11 0.23 0.10 0 0 1675\n
PctPolicMinor 0 1 0.26 0.23 0.49 0.2 0.07 1675\n
OfficAssgnDrugUnits 0 1 0.08 0.12 0.34 0.04 0.03 1675\n
NumKindsDrugsSeiz 0 1 0.56 0.20 0.13 0.57 0.57 1675\n
PolicAveOTWorked 0 1 0.31 0.23 0.03 0.26 0.19 1675\n
LandArea 0 1 0.07 0.11 0.20 0.04 0.01 0\n
PopDens 0 1 0.23 0.20 0.28 0.17 0.09 0\n
PctUsePubTrans 0 1 0.16 0.23 0.15 0.07 0.01 0\n
PolicCars 0 1 0.16 0.21 0.38 0.08 0.02 1675\n
PolicOperBudg 0 1 0.08 0.14 0.34 0.03 0.02 1675\n
LemasPctPolicOnPatr 0 1 0.70 0.21 -0.08 0.75 0.74 1675\n
LemasGangUnitDeploy 0 1 0.44 0.41 0.12 0.5 0 1675\n
LemasPctOfficDrugUn 0 1 0.09 0.24 0.35 0 0 0\n
PolicBudgPerPop 0 1 0.20 0.16 0.10 0.15 0.12 1675\n
ViolentCrimesPerPop 0 1 0.24 0.23 1.00 0.15 0.03 0""")
        
    Model_container = st.container()
    with Model_container:
        st.title("ML model used : Random Forest Algorithm")
        st.write("""Random Forest is a popular machine learning algorithm that is used for both classification
        and regression tasks. It is an ensemble method that combines multiple decision trees to
        make predictions.The idea behind the algorithm is to generate a large number of decision trees,
        where each tree is trained on a randomly sampled subset of the training data and a 
        random subset of the input features.


During prediction, each decision tree in the forest independently makes a prediction, and 
the final prediction is then determined by taking the majority vote of all the trees. This helps
 to reduce overfitting and improve the accuracy of the predictions.

Random Forest is a versatile algorithm that can be used for a wide range of applications,
including image classification, fraud detection, and customer churn prediction. It is also
relatively easy to use, as it requires minimal data preprocessing and hyperparameter tuning.
 Overall, Random Forest is a powerful and popular algorithm in the field of machine learning.""")
    Input_container = st.container()
    with Input_container:
        st.title("Enter the 100 features according to given info provided in the dataset section in the form (feature  values seprated by ,)")
        input=st.text_input('Enter Input Array')
         
        
    model = pickle.load(open('model.pkl', 'rb'))
    features=list([[input]])
    try:
        features = list(map(float,input.split(',')))
    except ValueError:
        features = None
    Submit= st.button('Submit')

    if Submit:
        prediction = model.predict([features])
        if prediction== 0:
            st.success("Low Chance of Crime")
        else :
            st.success("High Chance of Crime") 

    Score_container = st.container() 
    with Score_container :
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.title("Confusin Matrix and accuracy score ")
        with open('matrix.pkl', 'rb') as f1:
            matrix = pickle.load(f1)
        plt.imshow(matrix, cmap='Blues')
        plt.colorbar()
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        st.pyplot()
        with open('accuracy.pkl', 'rb') as f2:
            accuracy= pickle.load(f2)
        st.subheader(f"Accuracy of model is : {round(accuracy,3)} or {round(accuracy*100,3)}%")
        st.write("""This confusion matrix represents the performance of a binary classification model on a test 
        dataset. The rows of the matrix correspond to the actual class labels, while the columns 
        correspond to the predicted class labels.

In this particular case, the model predicted 196 instances to be positive (belonging to
 the second class) and 303 instances to be negative (belonging to the first class). Out of the 196 
 positive instances, the model correctly predicted 180, but misclassified 16 as negative. Out 
 of the 303 negative instances, the model correctly predicted 291, but misclassified 12 as
   positive.

Therefore, the confusion matrix shows that the model has a high true positive rate (TPR) of
 0.938 and a high true negative rate (TNR) of 0.960, but a relatively low precision (0.948) and 
 F1-score (0.943) due to the false positive and false negative classifications.""")
run()
