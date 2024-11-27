# https://www.kaggle.com/code/imoore/intro-to-exploratory-data-analysis-eda-in-python
# https://www.analyticsvidhya.com/blog/2022/07/step-by-step-exploratory-data-analysis-eda-using-python/
# https://medium.com/@jairo.data/python-e-pandas-uma-combina%C3%A7%C3%A3o-poderosa-para-an%C3%A1lise-de-dados-70b798e14c1e
# https://stackoverflow.com/questions/76451866/how-to-calculate-corr-from-a-dataframe-with-non-numeric-columns - correlação non numeric
# https://www.geeksforgeeks.org/how-to-convert-categorical-string-data-into-numeric-in-python/ Transformação de dados categóricos para númericos
# https://medium.com/@brandon93.w/converting-categorical-data-into-numerical-form-a-practical-guide-for-data-science-99fdf42d0e10 Transformação de dados categóricos para númericos

# Importação das bibliotecas para EDA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error,\
mean_squared_error

# Printando o dataset
data = pd.read_excel("/content/sample_data/Lucas - Relatório saida motobombas 2018 2024 -30.04.xls")
print(data)

# Exclusão da última linha do datset, que representava o total dos valores de venda
data = data.dropna(axis=0, how='any')
print(data)


# Removendo coluna irrelevante
data = data.drop(["Tipo de movimento da operação" ], axis = 1)

'''
# Pegando os indices problemáticos
NegativeIndex = data[(data["Qt. item"] < 0)].index
data.drop(NegativeIndex, inplace=True)

data
'''

# Criando mais um dataset que será usado mais a frente
data_div=pd.DataFrame(data)


# Extraindo informações da data de emissão, como ano, mês e dia
data_div['Dt. emissão'] =  pd.to_datetime(data['Dt. emissão'], format='%d/%m/%Y')
data_div['Ano'] = data_div['Dt. emissão'].dt.year
data_div['Mes'] = data_div['Dt. emissão'].dt.month
data_div['Dia'] = data_div['Dt. emissão'].dt.day


# Verificando as primeiras linhas
print(data_div.head())

#data_div["Vl. faturamento"] = data_div["Vl. faturamento"].str.replace(',','.').astype(float)

#Verificando as colunas, a quantidade de valores e também o tipo de cada uma
print(data_div.info())


# Total de valores duplicados
print(data_div.isnull().sum())


# Descrição padrão do pandas, mas interessantes sobre as colunas númericas do DATAFRAME, em especial da coluna QT. ITEM que é o target do nosso problema
print(data_div.describe())

# Quantidade de linhas e colunas
print(data_div.shape)
#
ax = sns.boxplot(x=data_div['Qt. item'])
ax.figure.set_size_inches(12, 8)
# Mostrar o gráfico
plt.show()


#Pegando o faturamento em motobombas por ano
all_years = data_div.groupby('Ano')['Qt. item'].sum()
print("===========================================")
print("FATURAMENTO POR ANO: ")
all_years = data_div.groupby('Ano')['Qt. item'].sum().reset_index()
all_years=round(all_years)
all_years['Ano'] = all_years['Ano'].astype(str)

#Média de venda diária durante os anos
x = np.array((all_years['Ano']))
y = np.array(all_years['Qt. item'])
all_years = all_years.groupby("Ano")["Qt. item"].mean().reset_index()
print(all_years)
fig, axarr = plt.subplots(figsize=(18, 6))
plt.bar(x, y,)
plt.xlabel('Anos')
plt.title("Faturamento Ano")
axarr.set_ylim(ymin=0)


for i, v in enumerate(y):
    axarr.text(i-0.15, v, str(v), fontsize=10, va='bottom', color="black" , fontweight = 'bold')

plt.show()

# Média de transações de vendas, isso em divisões de grupos
tab0 = data_div.groupby("Ano")["Qt. item"].mean().reset_index()
print(tab0)

print(data_div.info())


#filtrando as mendas por data de emissão ao longo dos anos
sales_average_day = data_div.groupby(['Dt. emissão', 'Mes', 'Ano']).sum().reset_index()

#Média de venda diária durante os anos
sales_average_day = sales_average_day.groupby("Ano")["Qt. item"].mean().reset_index()
print(sales_average_day)

sales_average_day=round(sales_average_day)
sales_average_day['Ano'] = sales_average_day['Ano'].astype(str)
#Média de venda diária durante os anos
x = np.array((sales_average_day['Ano']))
y = np.array(sales_average_day['Qt. item'])
sales_average_day = sales_average_day.groupby("Ano")["Qt. item"].mean().reset_index()
fig, axarr = plt.subplots(figsize=(18, 6))
plt.bar(x, y)
plt.xlabel('Anos')
plt.title("Média de vendas dia ao longo dos anos")
axarr.set_ylim(ymin=0)


for i, v in enumerate(y):
    axarr.text(i-0.15, v, str(v), fontsize=10, va='bottom', color="black" , fontweight = 'bold')

plt.show()


#Filtrando as vendas por mês e consequentemente, por ano
sales_average_month = data_div.groupby(['Ano', 'Mes'])['Qt. item'].sum().reset_index()

#Média de venda mensal durante os anos
sales_average_month = sales_average_month.groupby("Ano")["Qt. item"].mean().reset_index()
print(sales_average_month)
sales_average_month=round(sales_average_month)
sales_average_month['Ano'] = sales_average_month['Ano'].astype(str)


#Média de venda diária durante os anos
x = np.array((sales_average_month['Ano']))
y = np.array(sales_average_month['Qt. item'])
sales_average_month = sales_average_month.groupby("Ano")["Qt. item"].mean().reset_index()
fig, axarr = plt.subplots(figsize=(18, 6))
plt.bar(x, y)
plt.xlabel('Anos')
plt.title("Média de vendas por mês ao longo dos anos")
axarr.set_ylim(ymin=0)


for i, v in enumerate(y):
    axarr.text(i-0.15, v, str(v), fontsize=10, va='bottom', color="black" , fontweight = 'bold')

plt.show()


#filtrando as vendas(dinheiro) por data de emissão ao longo dos anos
invoice_average_day = data_div.groupby(['Dt. emissão', 'Mes', 'Ano']).sum().reset_index()

#Média de venda diária durante os anos
invoice_average_day = invoice_average_day.groupby("Ano")["Vl. faturamento"].mean().reset_index()
print(invoice_average_day)


invoice_average_day=round(invoice_average_day)
invoice_average_day['Ano'] = invoice_average_day['Ano'].astype(str)
#Média de venda diária durante os anos
x = np.array((invoice_average_day['Ano']))
y = np.array(invoice_average_day["Vl. faturamento"])
invoice_average_day = invoice_average_day.groupby("Ano")["Vl. faturamento"].mean().reset_index()
fig, axarr = plt.subplots(figsize=(18, 6))
plt.bar(x, y)
plt.xlabel('Anos')
plt.title("Média de faturamento por dia ao longo dos anos")
axarr.set_ylim(ymin=0)


for i, v in enumerate(y):
    axarr.text(i-0.15, v, str(v), fontsize=10, va='bottom', color="black" , fontweight = 'bold')

plt.show()

#Filtrando as vendas por mês e consequentemente, por ano
invoice_average_month = data_div.groupby(['Ano', 'Mes'])['Qt. item'].sum().reset_index()

#Média de venda mensal durante os anos
invoice_average_month = invoice_average_month.groupby("Mes")["Qt. item"].mean().reset_index()
print(invoice_average_month)



invoice_average_month=round(invoice_average_month)
invoice_average_month['Mes'] = invoice_average_month['Mes'].astype(str)
#Média de venda diária durante os anos
x = np.array((invoice_average_month['Mes']))
y = np.array(invoice_average_month['Qt. item'])
invoice_average_month = invoice_average_month.groupby("Mes")["Qt. item"].mean().reset_index()
fig, axarr = plt.subplots(figsize=(18, 6))
plt.bar(x, y)
plt.xlabel('Anos')
plt.title("Média de vendas por mês ao longo dos anos")
axarr.set_ylim(ymin=0)


for i, v in enumerate(y):
    axarr.text(i-0.25, v, str(v), fontsize=10, va='bottom', color="black" , fontweight = 'bold')

plt.show()

#Filtrando as vendas por mês e consequentemente, por ano
sales_average_month = data_div.groupby(['Ano', 'Mes'])['Vl. faturamento'].sum().reset_index()

#Média de venda mensal durante os anos
sales_average_month = sales_average_month.groupby("Mes")["Vl. faturamento"].mean().reset_index()
print(sales_average_month)


sales_average_month=round(sales_average_month)
sales_average_month['Mes'] = sales_average_month['Mes'].astype(str)
#Média de venda diária durante os anos
x = np.array((sales_average_month['Mes']))
y = np.array(sales_average_month['Vl. faturamento'])
sales_average_month = sales_average_month.groupby("Mes")['Vl. faturamento'].mean().reset_index()
fig, axarr = plt.subplots(figsize=(18, 6))
plt.bar(x, y)
plt.xlabel('Anos')
plt.title("Média de vendas por mês de todos os anos")
axarr.set_ylim(ymin=0)


for i, v in enumerate(y):
    axarr.text(i-0.32, v, str(v), fontsize=10, va='bottom', color="black" , fontweight = 'bold')

plt.show()


#Filtrando as vendas por mês e consequentemente, por ano
invoice_average_day = data_div.groupby('Dia')['Qt. item'].sum().reset_index()

#Média de venda mensal durante os anos
invoice_average_day = invoice_average_day.groupby("Dia")["Qt. item"].mean().reset_index()
print(invoice_average_day)


invoice_average_day=round(invoice_average_day)
invoice_average_day['Dia'] = invoice_average_day['Dia'].astype(str)
#Média de venda diária durante os anos
x = np.array((invoice_average_day['Dia']))
y = np.array(invoice_average_day["Qt. item"])
invoice_average_day = invoice_average_day.groupby("Dia")["Qt. item"].mean().reset_index()
fig, axarr = plt.subplots(figsize=(25, 6))
plt.bar(x, y)
plt.xlabel('Dias')
plt.title("Média de vendas por dia de todos os anos")
axarr.set_ylim(ymin=0)


for i, v in enumerate(y):
    axarr.text(i-0.34, v, str(v), fontsize=8, va='bottom', color="black" , fontweight = 'bold')

plt.show()



#Filtrando as vendas por mês e consequentemente, por ano
invoice_average_day = data_div.groupby('Dia')['Vl. faturamento'].sum().reset_index()

#Média de venda mensal durante os anos
invoice_average_day = invoice_average_day.groupby("Dia")["Vl. faturamento"].mean().reset_index()
print(invoice_average_day)


invoice_average_day=round(invoice_average_day)
invoice_average_day['Dia'] = invoice_average_day['Dia'].astype(str)
#Média de venda diária durante os anos
x = np.array((invoice_average_day['Dia']))
y = np.array(invoice_average_day['Vl. faturamento'])
invoice_average_day = invoice_average_day.groupby("Dia")['Vl. faturamento'].mean().reset_index()
fig, axarr = plt.subplots(figsize=(28, 10))
plt.bar(x, y)
plt.xlabel('Dia')
plt.title("Média de faturamento por dia de todos os anos")
axarr.set_ylim(ymin=0)


for i, v in enumerate(y):
    axarr.text(i-0.45, v, str(v), fontsize=8, va='bottom', color="black" , fontweight = 'bold')

plt.show()


#VERIFICAR VENDA DE CAIXAS POR PRODUTO
fig, axarr = plt.subplots(figsize=(20,6))
by_product = data_div.groupby('Produto')['Qt. item'].sum().nlargest(20).sort_values(ascending=True).reset_index()
by_product=round(by_product)
by_product['Produto'] = by_product['Produto'].astype(str)
print(by_product)


x = np.array(by_product['Produto'])
y = np.array(by_product['Qt. item'])
plt.barh(x,y)
plt.title("TOTAL VENDAS POR PRODUTO", fontsize=16)
#axarr.set_ylim(ymin=0)
axarr.set_xlim(xmin=0)
plt.legend()
for i, v in enumerate(y):
    axarr.text(v, i-0.40,str(round(v)), fontsize=12, va='bottom', color="red" , fontweight = 'bold')

plt.show()


#VERIFICAR VENDA DE CAIXAS POR PRODUTO
fig, axarr = plt.subplots(figsize=(20,6))
by_product = data_div.groupby('Produto')['Vl. faturamento'].sum().nlargest(20).sort_values(ascending=True).reset_index()
by_product=round(by_product)
by_product['Produto'] = by_product['Produto'].astype(str)
print(by_product)


x = np.array(by_product['Produto'])
y = np.array(by_product['Vl. faturamento'])
plt.barh(x,y)
plt.title("TOTAL FATURADO POR PRODUTO", fontsize=16)
#axarr.set_ylim(ymin=0)
axarr.set_xlim(xmin=0)
plt.legend()
for i, v in enumerate(y):
    axarr.text(v, i-0.40,str(round(v)), fontsize=12, va='bottom', color="red" , fontweight = 'bold')

plt.show()


#Definindo os grupos mais relevantes em termos de faturamento
#group = data.groupby('Grupo de produto')['Vl. faturamento'].sum().nlargest(10)
#print(group)

#Definindo os grupos mais relevantes em termos de faturamento by year
product_inv_most = data.groupby('Grupo de produto')['Vl. faturamento'].sum().nlargest(5)
print(product_inv_most)


fig = plt.figure(figsize=(12,12))
product_inv_most.plot(kind='pie',
                      autopct='%1.1f%%', fontsize=20,
                      colors = ["#20257c", "#424ad1", "#6a8ee8", "#66bbe2", "#66dee2"],
                      labeldistance = 1.1,
                      wedgeprops = {"ec": "k"},
                      pctdistance=0.8,
                      textprops = {'color': "r"},
                      )
plt.title("Faturamento por Grupo",
          fontsize = 30)



#Definindo os grupos mais relevantes em termos de faturamento by year
product_sales_most = data.groupby('Grupo de produto')['Qt. item'].sum().nlargest(5)
print(product_sales_most)

fig = plt.figure(figsize=(12,12))
product_sales_most.plot(kind='pie',
                      autopct='%1.1f%%', fontsize=20,
                      colors = ["#20257c", "#424ad1", "#6a8ee8", "#66bbe2", "#66dee2"],
                      labeldistance = 1.1,
                      wedgeprops = {"ec": "k"},
                      pctdistance=0.8,
                      textprops = {'color': "r"},
                      )
plt.title("Quantidade de vendas por Grupo",
          fontsize = 30)


#Razão e proporção dos Grupos em relação a venda e faturamento
_app0_5 = 70.9/89
_app6_10 = 19.6/5.2
_ad = 1.7/2.2
_aph = 1.8/1.9
_ap = 6/1.7
outros = 0.184632558800

ratio = [_app0_5, _app6_10, _ad, _aph, _ap,outros]
print(ratio)
groups = ["MOTOBOMBA APP 0-5", "MOTOBOMBA APP 6-10", "MOTOBOMBA AD", "MOTOBOMBA APH",  "MOTOBOMBA AP", "Outros" ]


fig = plt.figure(figsize=(12,12))
# Definições
colors = ['lightcoral', 'gold', 'yellowgreen', 'lightskyblue', 'orange', 'purple']

#  Plot das motobombas mais lucrativas
plt.pie(ratio, labels=groups, colors=colors, autopct='%1.1f%%', textprops={'fontsize': 20, 'color': "r"}, shadow=True, startangle=140, labeldistance = 1.1, )
plt.title("Razão e proporcionalidade dos grupos(Venda X Faturamento)",
          fontsize = 30)
plt.show()


# Checando os tipos de dados de cada coluna do dataset
print(data_div.dtypes)
product_data = data_div
product_data = product_data.drop(["Produto", "Grupo de produto" ], axis = 1)
product_data


#Pegando o faturamento em motobombas por mês ao longo dos anos
category = ['MOTOBOMBA APP 0-5', 'MOTOBOMBA APP 6-10','MOTOBOMBA AD', 'MOTOBOMBA APH', 'MOTOBOMBA AP']

all_months = data_div.groupby('Mes')['Qt. item'].sum().reset_index()
print("===========================================")
print("QTD. VENDIDA POR MÊS EM TODOS OS ANOS: ")
print(all_months)
print("===========================================")

# setting the dimensions of the plot
#fig, ax = plt.subplots(figsize=(30, 15))
plt.figure(figsize=(18, 18))
plt.subplot(2, 1, 1)
sns.barplot(data=data_div,
             x='Mes',
             y='Qt. item',
             hue='Grupo de produto',
             estimator=sum,
              ).set_title("QTD. VENDAS POR MÊS EM TODOS OS ANOS:", fontsize=40);
plt.xticks(rotation=45)


#Pegando o faturamento em motobombas por mês ao longo dos anos
category = ['MOTOBOMBA APP 0-5', 'MOTOBOMBA APP 6-10','MOTOBOMBA AD', 'MOTOBOMBA APH', 'MOTOBOMBA AP']

all_months = data_div.groupby('Mes')['Vl. faturamento'].sum().reset_index()
print("===========================================")
print("QTD. VENDIDA POR MÊS EM TODOS OS ANOS: ")
print(all_months)
print("===========================================")

# setting the dimensions of the plot
plt.figure(figsize=(18, 18))
plt.subplot(2, 1, 1)

sns.barplot(data=data_div,
             x='Mes',
             y='Vl. faturamento',
             hue='Grupo de produto',
             estimator=sum,
              ).set_title("FATURAMENTO POR MÊS EM TODOS OS ANOS:", fontsize=40);


data_sales = data_div

data_sales["Mes"] = data_sales["Mes"].fillna(0).astype('int64')
data_sales["Ano"] = data_sales["Ano"].fillna(0).astype('int64')



data_sales['data'] = data_sales['Mes'].map(str) +"-" +data_sales['Ano'].map(str)
#Criando a coluna conddicional
data_sales['sequencial'] = data_sales['Mes']

# Criando a parte sequencial
data_sales["sequencial"] = pd.to_numeric(data_sales["sequencial"])
data_sales.loc[data_sales['Ano'] == 2018, 'sequencial'] = data_sales['sequencial']
data_sales.loc[data_sales['Ano'] == 2019, 'sequencial'] = data_sales['sequencial'] + 12
data_sales.loc[data_sales['Ano'] == 2020, 'sequencial'] = data_sales['sequencial'] + 24
data_sales.loc[data_sales['Ano'] == 2021, 'sequencial'] = data_sales['sequencial'] + 36
data_sales.loc[data_sales['Ano'] == 2022, 'sequencial'] = data_sales['sequencial'] + 48
data_sales.loc[data_sales['Ano'] == 2023, 'sequencial'] = data_sales['sequencial'] + 60

data_sales = data_sales.drop(["Dia","Dt. emissão" ], axis = 1)

data_sales.sort_values(
["sequencial", "Cód. produto"], axis=0,
ascending=True, inplace=True
)
sequencialIndex = data[(data_sales["sequencial"]==0)].index
data_sales.drop(sequencialIndex, inplace=True)

print(data_sales)


