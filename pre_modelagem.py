#CONTINUAÇÃO DE ANALISE_SERIES_TEMPORAIS.PY

from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt

# A autocorrelação nos ajuda a definir quais features são importantes no momento em que estamos modelando o nosso
# algoritmo. Essa correlação mede o quão fortemente ligada são duas variáveis (dois lags) entre si. Ou seja, seria uma relação
# como por exemplo, se a autocorrelação for um, significa que se uma variável subir ou descer, a outra adotará o mesmo comportamento.

# Podemos observar claramente um ciclo de 60 meses, porém tem um valor negativo.
# O que isso significa? Uma autocorrelação negativa implica que, se um valor
# passado estiver acima da média, é mais provável que o valor mais recente
# esteja abaixo da média (ou vice-versa).

# Plot ACF
fig_acf, ax_acf = plt.subplots(figsize=(8, 6))
plot_acf(monthly_sales['Qt. item'], lags=len(monthly_sales)-1, ax=ax_acf)
ax_acf.set_title('Função de Autocorrelação')
plt.show()


from statsmodels.graphics.tsaplots import plot_pacf
import matplotlib.pyplot as plt

# Plot PACF
fig_pacf, ax_pacf = plt.subplots(figsize=(8, 6))
plot_pacf(monthly_sales['Qt. item'], lags=(len(monthly_sales)/2)-1, ax=ax_pacf)
ax_pacf.set_title('Função de Autocorrelação Parcial')
plt.show()


if dfoutput['p-value'] >= 0.05:
    df_log = np.log(monthly_sales['Qt. item'])
    ma_log = df_log.rolling(12).mean()
    fig, ax = plt.subplots()
    df_log.plot(ax=ax, legend=False)
    ma_log.plot(ax=ax, legend=False, color='r')
    plt.show(block=False)

    #subtrair média do log dos dados, iremos fazer a média em 12, pois iremos fazer anualmente
    df_sub = (df_log - ma_log).dropna().reset_index()
    ma_sub = df_sub.rolling(12).mean()
    #desvio padrão
    std_sub = df_sub.rolling(12).std()

    print('\n\n')


    #repetir o ADF
    print('PÓS ESTACIONARIZAÇÃO')
    print('==============================================================================')
    dftest_again = adfuller(df_sub['Qt. item'], autolag='AIC')
    print('Dickey-Fuller Aumentado')
    print('Teste Estatístico: {:.4f}'.format(dftest_again[0]))
    print('Valor-p: {:.10f}'.format(dftest_again[1]))
    print('Valores Críticos:')
    for key, value in dftest_again[4].items():
      print('\t{}: {:.4f}'.format(key, value))
    print('==============================================================================')
    print('\n\n')


    from statsmodels.graphics.tsaplots import plot_acf
    from statsmodels.graphics.tsaplots import plot_pacf

    #A autocorrelação nos ajuda a definir quais features são importantes no momento em que estamos modelando o nosso
    #algoritmo. Essa correlação mede o quão fortemente ligada são duas variaveis(dois lags) entre si. Ou seja, seria uma relação
    #como por exemplo, se a autocorrelação for um, significa que se uma varíavel subir ou descer, a outra adotará o mesmo comportamento.


    #Podemos observar claramente um ciclo de 60 meses, porém tem um valor negativo.
    # O que isso significa? Uma autocorrelação negativa implica que, se um valor
    # passado estiver acima da média, é mais provável que o valor mais recente
    # esteja abaixo da média (ou vice-versa).
    plot_acf(df_sub['Qt. item'], lags= 12)
    plot_acf(df_sub['Qt. item'], lags= len(df_sub)-1)
    plot_pacf(df_sub['Qt. item'], lags= (len(df_sub)/2)-1)
    plt.show()
    monthly_sales = pd.DataFrame(df_sub)



else:
    print("A série é estacionária")


#ENGENHARIA DE ATRIBUTOS
#Eliminação de certas colunas do DataFrame
monthly_sales = monthly_sales.drop(["Cód. produto", "Grupo de produto", "Produto", "sequencial","Vl. faturamento"     ], axis = 1)


from datetime import datetime

if monthly_sales["Mes"].iloc[-1]<12:
  print("batata")
  d_iserido = pd.DataFrame({'data': datetime.strptime(str(int(monthly_sales["Ano"].iloc[-1]))+'-'+str(int(monthly_sales["Mes"].iloc[-1]+1))+'-01',  '%Y-%m-%d'), "Ano": monthly_sales["Ano"].iloc[-1],'Mes': monthly_sales["Mes"].iloc[-1]+1, 'Qt. item': monthly_sales["Qt. item"].iloc[-2]}, index=[0])
  print(d_iserido)
  monthly_sales.loc[len(monthly_sales)] = d_iserido.iloc[0]
  print(monthly_sales)
else:
  print("Repolho")
  
  
#Aqui acontecem a criação dos lags 1 ao 6 para início dos testes, porém mais lags poderão ser testados posteriormente
def create_lag_features(monthly_sales, lags=0):
  y = monthly_sales.loc[:, "Qt. item"]
  for lag in range(lags):
    monthly_sales[f"lag_{lag + 1}"] = y.shift(lag + 1)
  return monthly_sales

monthly_sales = create_lag_features(monthly_sales, lags=6)
monthly_sales = monthly_sales.fillna(0.0)
print(monthly_sales)

#Aqui acontecem a criação das médias móveis em intervalos de 2 a 7 para início dos testes, porém mais lags poderão ser testados posteriormente
def create_rolling_mean(monthly_sales, roll=0):
  y = monthly_sales.loc[:, "Qt. item"]
  for roll in range(1, roll+1):
    monthly_sales[f"rolling_mean_{roll + 1}"] = y.rolling(window=roll+1).mean()
  return monthly_sales

monthly_sales = create_rolling_mean(monthly_sales, roll=6)
monthly_sales = monthly_sales.fillna(0.0)
print(monthly_sales)


#Aqui acontecem a criação das diferenças entre intervalos de cálculos 1 ao 6 para início dos testes, porém mais lags poderão ser testados posteriormente
def create_diff_features(monthly_sales, diff=0):
  y = monthly_sales.loc[:, "Qt. item"]
  for diff in range(diff):
    monthly_sales[f"diff{diff + 1}"] = y.diff((diff) + 1)*(-1)
  return monthly_sales

monthly_sales = create_diff_features(monthly_sales, diff=6)
monthly_sales = monthly_sales.fillna(0.0)
print(monthly_sales)


monthly_sales_numerical = pd.DataFrame(monthly_sales)
monthly_sales_numerical = monthly_sales_numerical.drop(["data",], axis = 1)
monthly_sales_numerical = monthly_sales_numerical

monthly_sales_numerical


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

#Crie um histograma da sua variável e veja se ele tem a forma clássica de "sino", característica de uma distribuição normal.
# Histograma
plt.hist(monthly_sales["Qt. item"], bins=30, density=True)
plt.title("Histograma")
plt.show()

import matplotlib.pyplot as plt
import scipy.stats as stats

# Compare os quantis da sua amostra com os quantis de uma distribuição normal.
# Se a distribuição for Gaussiana, os pontos do gráfico estarão próximos de uma linha reta.
# Gráfico Q-Q
fig = plt.figure()
res = stats.probplot(monthly_sales["Qt. item"], dist="norm", plot=plt)
plt.title("Gráfico Q-Q")
plt.ylabel("Valores Ordenados")
plt.xlabel("Quantis Teóricos")
plt.show()


from scipy.stats import shapiro, anderson, kstest

#Um teste formal para avaliar a normalidade dos dados. Se o p-valor do teste for maior que 0.05, não rejeitamos a hipótese nula de que os dados seguem uma distribuição normal.
# Teste de Shapiro-Wilk
stat, p_value = shapiro(monthly_sales["Qt. item"])
print(f"Shapiro-Wilk Test: Stat={stat}, p-value={p_value}")
print("")
# Outro teste de hipótese para normalidade que, além de fornecer o p-valor, também indica quão bem os dados seguem uma distribuição normal.
# Teste de Anderson-Darling
result = anderson(monthly_sales["Qt. item"])
print(f"Anderson-Darling Test: Stat={result.statistic}, Critical Values={result.critical_values}")
print("")

#Compara a amostra com uma distribuição de referência (normal) e verifica a hipótese de que os dados vêm dessa distribuição.
# Teste de Kolmogorov-Smirnov
stat, p_value = kstest(monthly_sales["Qt. item"], 'norm')
print(f"Kolmogorov-Smirnov Test: Stat={stat}, p-value={p_value}")
print("")


# 4. Teste de Anderson-Darling
# O valor crítico para diferentes níveis de significância é retornado com a estatística.
result_ad = stats.anderson(monthly_sales["Qt. item"])
print('Teste Anderson-Darling:')
print(f'Estatística: {result_ad.statistic}')
for i in range(len(result_ad.critical_values)):
    sl, cv = result_ad.significance_level[i], result_ad.critical_values[i]
    print(f'  Nível de significância {sl}%: valor crítico={cv}, estatística={result_ad.statistic}')
if result_ad.statistic < result_ad.critical_values[2]:  # 5% de significância
    print('Distribuição Gaussiana (Normal) presumida pelo teste Anderson-Darling.\n')
else:
    print('Distribuição não é Gaussiana (Normal) pelo teste Anderson-Darling.\n')

# 5. Teste de Kolmogorov-Smirnov (KS Test)
# Testa se os dados seguem uma distribuição específica (nesse caso, normal).
stat_ks, p_ks = stats.kstest(monthly_sales["Qt. item"], 'norm')
print(f'Teste Kolmogorov-Smirnov: estatística={stat_ks}, p-valor={p_ks}')
if p_ks > 0.05:
    print('Distribuição Gaussiana (Normal) presumida pelo teste KS.\n')
else:
    print('Distribuição não é Gaussiana (Normal) pelo teste KS.\n')

# 6. Teste de Jarque-Bera
# Verifica a normalidade baseada na assimetria (skewness) e curtose (kurtosis).
stat_jb, p_jb = stats.jarque_bera(monthly_sales["Qt. item"])
print(f'Teste Jarque-Bera: estatística={stat_jb}, p-valor={p_jb}')
if p_jb > 0.05:
    print('Distribuição Gaussiana (Normal) presumida pelo teste Jarque-Bera.\n')
else:
    print('Distribuição não é Gaussiana (Normal) pelo teste Jarque-Bera.\n')

# 7. Skewness (Assimetria) e Kurtosis (Curtose)
# A assimetria próxima de 0 e curtose próxima de 3 indicam uma distribuição normal.
skewness = stats.skew(monthly_sales["Qt. item"])
kurtosis = stats.kurtosis(monthly_sales["Qt. item"], fisher=False)  # fisher=False para curtose não ajustada
print(f'Assimetria (Skewness): {skewness}')
print(f'Curtose: {kurtosis}')
if abs(skewness) < 0.5 and abs(kurtosis - 3) < 0.5:
    print('Distribuição parece Gaussiana (Normal) baseada na assimetria e curtose.\n')
else:
    print('Distribuição pode não ser Gaussiana (Normal) baseada na assimetria e curtose.\n')
    
    
    
from sklearn.preprocessing import LabelEncoder

# Creating a instance of label Encoder.
le = LabelEncoder()

for col in monthly_sales_numerical:
  if monthly_sales_numerical[col].dtype =='object':
    monthly_sales_numerical[col] = le.fit_transform(monthly_sales_numerical[col])

print("Com transformação de tipos categóricos para númericos: ")


from scipy.stats.stats import kendalltau

fat_heat_map_corr = monthly_sales_numerical.corr(method='pearson')
fat_heat_map_corr


#Aqui é gerado uma mapa para terntarmos identificar uma correlação entre as colunas
ax = sns.heatmap(data=fat_heat_map_corr, annot=True)
ax.figure.set_size_inches(12, 8)
plt.tight_layout()
plt.show()


def correlation(dataset, threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr


corr_features = correlation(monthly_sales_numerical, 0.65)
len(set(corr_features))
monthly_sales = monthly_sales.drop(corr_features, axis=1)
# Import library for VIF
monthly_sales_numerical = monthly_sales_numerical.drop(columns=[])

from statsmodels.stats.outliers_influence import variance_inflation_factor

def calc_vif(X):

    # Calculando o  VIF
    vif = pd.DataFrame()
    vif["Variáveis"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)

X = monthly_sales_numerical.iloc[:,:-1]
calc_vif(X)

from sklearn.feature_selection import VarianceThreshold
var_thres=VarianceThreshold(threshold=0.5)
var_thres.fit(monthly_sales_numerical)
constant_columns = [column for column in monthly_sales_numerical.columns
                    if column not in monthly_sales_numerical.columns[var_thres.get_support()]]

print(len(constant_columns))

#Separação do conjunto entre treinamento e teste
train_size = int(len(monthly_sales) * 0.80)
train_data, test_data = monthly_sales[:train_size], monthly_sales[train_size:]


# normalize the numeric columns
from sklearn.preprocessing import MinMaxScaler
import pickle


data_to_normalize = monthly_sales[['Ano',	'Mes','lag_1',	'diff1',	'diff2']]

scaler=MinMaxScaler()
scaler.fit(data_to_normalize)

data_normalized=scaler.transform(data_to_normalize)
data_normalized=pd.DataFrame(data_normalized, index=data_to_normalize.index, columns=data_to_normalize.columns)
print(data_normalized)
# Salvar os parâmetros de normalização para serem usadas mais a frente, quando formos prever e utilizar a predição para gerar novas features que deverão estar normalizadas.
# Salvar o scaler ajustado
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

print("Scaler ajustado e salvo.")


#Separação do conjunto entre treinamento e teste
train_size = int(len(data_normalized) * 0.80)
data_normalized_train, data_normalized_test = data_normalized[:train_size], data_normalized[train_size:]

predict = data_normalized_test.iloc[-1].reset_index()
data_normalized_test = data_normalized_test.iloc[:-1]
data_normalized_test

#predict["constante"] = 1
predict = predict.pivot_table(
    values=76,
    columns="index",
)

## Reordene as colunas
predict = predict.reindex(['Ano', 'Mes', 'lag_1','diff1','diff2'  ], axis=1)
predict

#Conjunto de treinamento da coordenada X com as colunas que ajudarão na avaliação do modelo
X_train = data_normalized_train[['Ano',	'Mes','lag_1',	'diff1',	'diff2']]
#Conjunto de dados de treinamento com a coluna alvo
y_train = train_data["Qt. item"]

#Conjunto de treinamento da coordenada X com as colunas que ajudarão na avaliação do modelo
X_test = data_normalized_test[['Ano',	'Mes','lag_1',	'diff1',	'diff2']]
y_test = test_data["Qt. item"].iloc[:-1]
