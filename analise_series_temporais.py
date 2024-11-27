#CONTINUAÇÃO DO CÓDIGO EDA_TCC.PY


group_data = data_sales.loc[data_sales['Grupo de produto'] == 'MOTOBOMBA APP 0-5']

#TRANSFORMANDO O CONJUNTO DE DADOS PARA MENSAL
group_data['data'] = pd.to_datetime(group_data['data'])
group_data['data'] = group_data['data'].dt.to_period('M')#Isso aqui já faz ser sequencial

#Soma todos os valores de venda no mês
monthly_sales = group_data.groupby(['data','Ano', 'Mes']).sum().reset_index()
monthly_sales['data'] = monthly_sales['data'].dt.to_timestamp()

monthly_sales

#AQUI AJUDA A TER UMA IDEIA S DA VARIAÇÃO DE VENDAS, POR MÊS DE TODOS OS ANOS
plt.figure(figsize=(12,6))
plt.plot(monthly_sales['data'], monthly_sales['Qt. item'])
plt.xlabel('Anos')
plt.title("Variação das vendas em todos os anos")
plt.show()


#print(monthly_sales)

#Análise de Estacionariedade
from statsmodels.tsa.stattools import adfuller

#Média móvel
rolmean = pd.Series(monthly_sales['Qt. item']).rolling(window=12).mean()
#Desvio Padrão móvel
rolstd = pd.Series(monthly_sales['Qt. item']).rolling(window=12).std()

#Plotando as estatísticas médias:
orig = plt.plot(monthly_sales['Qt. item'], color='blue',label='Original')
mean = plt.plot(rolmean, color='red', label='Média Móvel')
std = plt.plot(rolstd, color='black', label = 'Desvio Padrão')
plt.legend(loc='best')
plt.title('Média Móvel & Desvio Padrão')
plt.show(block=False)
print('\n\n')

#Teste Dickey-Fuller:
#Se o valor de p for menor que 0.05 e o valor estatítico
#for menor ou igual aos valores críticos,
#a série é estacionária, caso contrário é não estacionária
print('Resultados do Test Dickey-Fuller :')
dftest = adfuller(monthly_sales['Qt. item'], autolag='AIC')
dfoutput = pd.Series(dftest[0:4],
                     index=['Test Statistic','p-value',
                            '#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print(dfoutput)

print(dfoutput['p-value'])


from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# Decompose the time series
decomposition_plot_add = seasonal_decompose(monthly_sales['Qt. item'], model='additive', period=12)

# Plot the decomposition
fig = decomposition_plot_add.plot()
fig.set_size_inches((16, 9))

# Set y-axis labels in Portuguese
fig.axes[0].set_ylabel('Original')
fig.axes[1].set_ylabel('Tendência')
fig.axes[2].set_ylabel('Sazonalidade')
fig.axes[3].set_ylabel('Resíduo')

# Tight layout to realign things
fig.tight_layout()
plt.show()




