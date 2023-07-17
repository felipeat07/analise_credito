import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


base_credit = pd.read_csv('credit_data.csv')

print(base_credit, base_credit.describe())


np.unique(base_credit['default'], return_counts=True)

sns.countplot(x = base_credit['default'])


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
grafico.show()


#Divisao entre previsores e classe (pre processamento dos dados)
X_credit = base_credit.iloc[: , 1:4].values
Y_credit = base_credit.iloc[: , 4].values

