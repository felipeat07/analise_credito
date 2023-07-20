import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

base_credit = pd.read_csv('credit_data.csv')

np.unique(base_credit['default'], return_counts=True)

# sns.countplot(x = base_credit['default'])

#removendo o atributo idade da base de dados
base_credit2 = base_credit.drop('age', axis= 1)

#removendo apenas os registros com valores inconsistentes
base_credit3 = base_credit.drop(base_credit[base_credit['age'] < 0].index)

#preenchendo os valores inconsistentes com a media dos valores
mean_age = base_credit3['age'].mean()

base_credit.loc[base_credit['age'] < 0, 'age'] = mean_age

#valores nulos de idade
null_age_data = base_credit.loc[pd.isnull(base_credit['age'])]

#aterando os valores nulos utilizando pandas
base_credit['age'].fillna(mean_age, inplace=True)


grafico = px.scatter_matrix(base_credit, dimensions=['age', 'income', 'loan'], color = 'default')
#grafico.show()

#Divisao entre previsores e classe (pre processamento dos dados)
X_credit = base_credit.iloc[: , 1:4].values
Y_credit = base_credit.iloc[: , 4].values

#normalizando/padronizando os valores de x_credit
from sklearn.preprocessing import StandardScaler

scaler_credit = StandardScaler()
X_credit = scaler_credit.fit_transform(X_credit)

max_income = X_credit[:, 0].max()
min_income = X_credit[:,0].min()

max_age = X_credit[:,1].max()
min_age = X_credit[:,1].min()

max_loan = X_credit[:,2].max()
min_loan = X_credit[:,2].min()


#Preprocessando base de dados do censu

base_census = pd.read_csv('census.csv')

np.unique(base_census['income'], return_counts=True)
#sns.countplot(x = base_census['income'])

grafico2 = px.treemap(base_census, path=['occupation', 'relationship', 'age'])
#grafico2.show()

grafico3 = px.parallel_categories(base_census, dimensions=['occupation', 'relationship'])
#grafico3.show()

#divisao entre previsores e classe

X_census = base_census.iloc[:, 0:14].values
Y_census = base_census.iloc[:, 14].values


#Tratamento de atributos categoricos com LabelEncoder

from sklearn.preprocessing import LabelEncoder

label_encoder_workclass = LabelEncoder()
label_encoder_education = LabelEncoder()
label_encoder_marital = LabelEncoder()
label_encoder_occupation = LabelEncoder()
label_encoder_relationship = LabelEncoder()
label_encoder_race = LabelEncoder()
label_encoder_sex = LabelEncoder()
label_encoder_country = LabelEncoder()

X_census[:,1] = label_encoder_workclass.fit_transform(X_census[:,1])
X_census[:,3] = label_encoder_education.fit_transform(X_census[:,3])
X_census[:,5] = label_encoder_marital.fit_transform(X_census[:,5])
X_census[:,6] = label_encoder_occupation.fit_transform(X_census[:,6])
X_census[:,7] = label_encoder_relationship.fit_transform(X_census[:,7])
X_census[:,8] = label_encoder_race.fit_transform(X_census[:,8])
X_census[:,9] = label_encoder_sex.fit_transform(X_census[:,9])
X_census[:,13] = label_encoder_country.fit_transform(X_census[:,13])

#usando o onehotencoded para evitar valores muito maiores que outros dentro dos parametros do labelencoded

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

onehotencoder_census = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [1,3,5,6,7,8,9,13])], remainder='passthrough')

X_census = onehotencoder_census.fit_transform(X_census).toarray()

#Escalonamento dos valores padronizacao/normalizacao
scaler_census = StandardScaler()
X_census = scaler_census.fit_transform(X_census)

#divisao das bases de dados em treinamento e teste
from sklearn.model_selection import train_test_split

X_credit_treinamento, X_credit_teste, Y_credit_treinamento, Y_credit_teste = train_test_split(X_credit, Y_credit, test_size = 0.25, random_state = 0 )

X_census_treinamento, X_census_teste, Y_census_treinamento, Y_census_teste = train_test_split(X_census, Y_census, test_size = 0.15, random_state = 0 )


#Salvando as variaveis
import pickle

with open('credit.pkl', mode = 'wb') as f:
    pickle.dump([X_credit_treinamento, Y_credit_treinamento, X_credit_teste, Y_credit_teste], f)

with open('census.pkl', mode = 'wb') as f:
    pickle.dump([X_census_treinamento, Y_census_treinamento, X_census_teste, Y_census_teste], f)