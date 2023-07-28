import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

with open('../pre_processamento_dados/census.pkl', 'rb') as f:
    X_census_treinamento, Y_census_treinamento, X_census_teste, Y_census_teste = pickle.load(f)

naive_census = GaussianNB()
naive_census.fit(X_census_treinamento, Y_census_treinamento)
previsoes = naive_census.predict(X_census_teste)

accuracy_score(Y_census_teste, previsoes)
