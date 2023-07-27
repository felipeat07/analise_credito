import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
import pickle

base_risco_credito = pd.read_csv('risco_credito.csv')

X_risco_credito = base_risco_credito.iloc[:,0:4].values
Y_risco_credito = base_risco_credito.iloc[:,4].values

label_encoder_historia = LabelEncoder()
label_encoder_divida = LabelEncoder()
label_encoder_garantias = LabelEncoder()
label_encoder_renda = LabelEncoder()

X_risco_credito[:,0] = label_encoder_historia.fit_transform(X_risco_credito[:,0])
X_risco_credito[:,1] = label_encoder_divida.fit_transform(X_risco_credito[:,1])
X_risco_credito[:,2] = label_encoder_garantias.fit_transform(X_risco_credito[:,2])
X_risco_credito[:,3] = label_encoder_renda.fit_transform(X_risco_credito[:,3])

with open('risco_credito.pkl', 'wb') as f:
    pickle.dump([X_risco_credito, Y_risco_credito], f)

naive_risco_credito = GaussianNB()
naive_risco_credito.fit(X_risco_credito, Y_risco_credito)


#previsao
#historia BOA(0), divida ALTA(0), garantias NENHUMA(1), renda >35 (2)
#historia RUIM(2), divida ALTA(0), garantias ADEQUADA(0), renda <15 (0)

previsao = naive_risco_credito.predict([[0,0,1,2], [2,0,0,0]])


#Base de dados credit data
with open('../pre_processamento_dados/credit.pkl', 'rb') as f:
    X_credit_treinamento, Y_credit_treinamento, X_credit_teste, Y_credit_teste = pickle.load(f)

naive_credit_data = GaussianNB()
naive_credit_data.fit(X_credit_treinamento, Y_credit_treinamento)

previsoes = naive_credit_data.predict(X_credit_teste)

#Comparando as previsoes com o Y_credit_teste

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

accuracy_score(Y_credit_teste, previsoes)

confusion_matrix(Y_credit_teste, previsoes)

from yellowbrick.classifier import ConfusionMatrix

cm = ConfusionMatrix(naive_credit_data)
cm.fit(X_credit_treinamento, Y_credit_treinamento)
cm.score(X_credit_teste, Y_credit_teste)

print(classification_report(Y_credit_teste, previsoes))

#Base de dados census

