import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
from yellowbrick.classifier import ConfusionMatrix


with open('../pre_processamento_dados/credit.pkl', 'rb') as f:
    X_credit_treinamento, Y_credit_treinamento, X_credit_teste, Y_credit_teste = pickle.load(f)

naive_credit_data = GaussianNB()
naive_credit_data.fit(X_credit_treinamento, Y_credit_treinamento)

previsoes = naive_credit_data.predict(X_credit_teste)

#Comparando as previsoes com o Y_credit_teste

confusion_matrix(Y_credit_teste, previsoes)

cm = ConfusionMatrix(naive_credit_data)
cm.fit(X_credit_treinamento, Y_credit_treinamento)
cm.score(X_credit_teste, Y_credit_teste)

print(classification_report(Y_credit_teste, previsoes))

accuracy_score(Y_credit_teste, previsoes)