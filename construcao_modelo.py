#CONTINUAÇÃO DO CÓDIGO PRÉ_MODELAGEM.PY


#=============================================XGBOOST===========================================================

import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import month_plot, quarter_plot
from statsmodels.tsa.statespace.tools import diff
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.stattools import jarque_bera
import scipy.stats as sct
from statsmodels.tools.eval_measures import rmse
from sklearn.metrics import mean_absolute_error
from sklearn import metrics


# Aqui ocorre a aplicação do modelo XGBoost

from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot

#Parâmetros abaixos definidos através de testes e leitura de material
#Melhores parâmetros até o momento: 13/05/2024 - 01:00am
cv_split = TimeSeriesSplit()
model = XGBRegressor(alpha=1, reg_lambda=1)
parameters = {
    "max_depth": [3, 5, 7], #Profundidade da árvore
    "learning_rate": [0.0045,0.005,0.0055, 0.3], # Velocidade de aprendizagem
    "n_estimators": [8], # o número de execuções que o XGBoost tentará aprender
    "colsample_bytree": [1.0], # Profundidade da árvore
    'eval_metric': ['mae'], # Profundidade da árvore
    'eta': [0.3], # Profundidade da árvore
}


grid_search = GridSearchCV(estimator=model, cv=cv_split, param_grid=parameters)
grid_search.fit(X_train, y_train)

# Obtenha a melhor combinação de hiperparâmetros
xgb_best_params = grid_search.best_params_
print("Melhores hiperparâmetros:", xgb_best_params)

from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
# Treine o modelo novamente, porém agora com os melhores  hiperparâmetros
best_xgb_model = XGBRegressor(**xgb_best_params)
best_xgb_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)])
# Faça previsões sobre os dados de teste
y_pred_xgb = best_xgb_model.predict(X_test)
x_pred_xgb = best_xgb_model.predict(X_train)


#Verificar a importância das colunas
plt.figure(figsize=(12, 8))
pyplot.barh(X_test.columns, best_xgb_model.feature_importances_)
plt.title("IMPORTÂNCIA DOS ATRIBUTOS NA MODELAGEM", fontsize=20)
# Aumentar o tamanho das fontes dos rótulos dos eixos
plt.xlabel("Importância", fontsize=18)
plt.ylabel("Atributos", fontsize=18)
# Aumentar o tamanho das fontes dos ticks dos eixos
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
pyplot.show()


#Plotando a árvore do modelo para entender melhor como este trabalha
from xgboost import plot_tree
plot_tree(best_xgb_model)
plt.gcf().set_size_inches(18.5, 10.5)
plt.show()


def evaluate_model(y_train, prediction):
  accuracy=100-(mean_absolute_percentage_error(y_train, prediction_xgb_train)*100)
  print('\nXGBOOST Acurácia Média TREINO: ', round(accuracy,2),'%.')
  print('XGBOOST Erro Médio Absoluto (MAE) TREINO: ', mean_absolute_error(y_train, prediction_xgb_train))
  print('XGBOOST Erro Quadrático Médio (MSE) TREINO: ', mean_squared_error(y_train, prediction_xgb_train))
  print('XGBOOST Raiz do Erro Quadrático Médio (RMSE) TREINO: ', mean_squared_error(y_train, prediction_xgb_train, squared=False ))
  print('XGBOOST Porcentagem do Erro Médio Absoluto (MAPE) TREINO: ',  round(mean_absolute_percentage_error(y_train, prediction_xgb_train)*100, 2),'%')
  #Proporção de variação dos dados explicados pelo modelo. Quanto mais perto de 1, melhor
  print('XGBOOST Explained Variance Score TREINO: ', metrics.explained_variance_score(y_train, prediction_xgb_train))
  #Maior erro entre previsões
  print('XGBOOST Erro Máximo TREINO: ', metrics.max_error(y_train, prediction_xgb_train))
  print('XGBOOST Erro Logarítmico Médio TREINO: ', metrics.mean_squared_log_error(y_train, prediction_xgb_train))
  #Mediana do erro entre valores previstos e valores reais
  print('XGBOOST Mediana do Erro Absoluto TREINO: ', metrics.median_absolute_error(y_train, prediction_xgb_train))

def plot_predictions(train_dates, y_train, prediction):
  df_xgb_train = pd.DataFrame({"Ano": train_data['Ano'], "Original": y_train, "Predição": prediction_xgb_train })
  figure, ax = plt.subplots(figsize=(10, 5))
  df_xgb_train.plot(ax=ax, label="Original", x="Ano", y="Original")
  df_xgb_train.plot(ax=ax, label="Predição", x="Ano", y="Predição")
  plt.legend(["Original", "Predição"])
  plt.title("XGBoost + Divisão Simplificada Treino + LAG")
  plt.show()
  
  
from sklearn.metrics import r2_score
# Validando os resultados
prediction_xgb_train = best_xgb_model.predict(X_train).round()
plot_predictions(train_data['Ano'], y_train, prediction_xgb_train)
evaluate_model(y_train, prediction_xgb_train)


def evaluate_model(y_test, prediction):
  accuracy=100-(mean_absolute_percentage_error(y_test, prediction_xgb_test)*100)
  print('\nXGBOOST Acurácia Média TESTE: ', round(accuracy,2),'%.')
  print('XGBOOST Erro Médio Absoluto (MAE) TESTE: ', metrics.mean_absolute_error(y_test, prediction_xgb_test))
  print('XGBOOST Erro Quadrático Médio (MSE) TESTE: ', metrics.mean_squared_error(y_test, prediction_xgb_test))
  print('XGBOOST Raiz do Erro Quadrático Médio (RMSE) TESTE: ', metrics.mean_squared_error(y_test, prediction_xgb_test, squared=False ))
  print('XGBOOST Porcentagem do Erro Médio Absoluto (MAPE) TESTE: ',  round(mean_absolute_percentage_error(y_test, prediction_xgb_test)*100, 2),'%')
  #Proporção de variação dos dados explicados pelo modelo. Quanto mais perto de 1, melhor
  print('XGBOOST Explained Variance Score TESTE: ', metrics.explained_variance_score(y_test, prediction_xgb_test))
  #Maior erro entre previsões
  print('XGBOOST Erro Máximo TESTE: ', metrics.max_error(y_test, prediction_xgb_test))
  print('XGBOOST Erro Logarítmico Médio TESTE: ', metrics.mean_squared_log_error(y_test, prediction_xgb_test))
  #Mediana do erro entre valores previstos e valores reais
  print('XGBOOST Mediana do Erro Absoluto TESTE: ', metrics.median_absolute_error(y_test, prediction_xgb_test))

def plot_predictions(testing_dates, y_test, prediction):
  df_test_xgb = pd.DataFrame({"Ano": testing_dates, "Original": y_test, "Predição": prediction_xgb_test })
  figure, ax = plt.subplots(figsize=(10, 5))
  df_test_xgb.plot(ax=ax, label="Original", x="Ano", y="Original")
  df_test_xgb.plot(ax=ax, label="Predição", x="Ano", y="Predição")
  plt.legend(["Original", "Predição"])
  plt.title("XGBoost - Validação em Divisão Simplificada")
  plt.show()
  
  
from sklearn.metrics import r2_score
# Validando os resultados
prediction_xgb_test = best_xgb_model.predict(X_test).round()
plot_predictions(testing_dates, y_test, prediction_xgb_test)
evaluate_model(y_test, prediction_xgb_test)


results_xgb = best_xgb_model.evals_result()
print(results_xgb)
epochs_xgb = len(results_xgb['validation_0']['mae'])
x_axis_xgb = range(0, epochs_xgb)


fig, ax = pyplot.subplots()
ax.plot(x_axis_xgb, results_xgb['validation_0']['mae'], label='Treino')
ax.plot(x_axis_xgb, results_xgb['validation_1']['mae'], label='Teste')
ax.legend()
pyplot.ylabel('MAE')
pyplot.title('VALIDAÇÃO MAE - XGBOOST')
pyplot.show()

#Prevendo o próximo mês
prediction_xgb_new = best_xgb_model.predict(predict)
print(prediction_xgb_new)

#==================================================================================================================


#=============================================LIGHTGBM===========================================================
import lightgbm as lgb
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, TimeSeriesSplit
from lightgbm import LGBMRegressor
from lightgbm import  plot_tree, plot_importance, plot_metric, plot_split_value_histogram



# LGBM
cv_split = TimeSeriesSplit()
model_lgbm = lgb.LGBMRegressor(random_state=None, n_jobs=8)
parameters_lgbm = {
    'num_leaves': [15, 31, 63, 189],           # Number of leaves in each tree
    'learning_rate': [0.01, 0.299410158161, 0.05, 0.1],   # Learning rate for gradient boosting
    'n_estimators': [10, 30, 50, 148, 200, 300, 500],         # Number of boosting iterations
    'objective': ['mean_absolute_error'],
    'metric': ['mean_absolute_error'],
    'boosting_type':['gbdt'],
    'max_depth':[10,40,50,100,200,300],
    'subsample_for_bin':[200000],
    'min_split_gain':[0.803068465953],
    'min_child_weight': [0.661596727364],
    "subsample":[1.0],
    "subsample_freq":[0],
    "colsample_bytree":[0.944268451069],
    "reg_alpha":[0.8562556466464],
    "reg_lambda":[0.554082451686],

}

grid_search_lgbm = RandomizedSearchCV(estimator=model_lgbm, cv=cv_split, param_distributions=parameters_lgbm)
grid_search_lgbm.fit(X_train, y_train)


# Treine o modelo novamente, porém agora com os melhores  hiperparâmetros
best_lgbm = LGBMRegressor(**grid_search_lgbm.best_params_)
best_lgbm.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)])


# Faça previsões sobre os dados de teste
y_pred_lgbm = best_lgbm.predict(X_test)
x_pred_lgbm = best_lgbm.predict(X_train)

print("===========================================================================================================================================")
print(grid_search_lgbm.best_estimator_)
print(grid_search_lgbm.best_params_)
print("===========================================================================================================================================")

#Verificar a importância das colunas
plt.figure(figsize=(12, 8))
pyplot.barh(X_test.columns, best_lgbm.feature_importances_)
plt.title("IMPORTÂNCIA DOS ATRIBUTOS NA MODELAGEM", fontsize=20)
# Aumentar o tamanho das fontes dos rótulos dos eixos
plt.xlabel("Importância", fontsize=18)
plt.ylabel("Atributos", fontsize=18)
# Aumentar o tamanho das fontes dos ticks dos eixos
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
pyplot.show()

# A métrica padrão para LGBM é a L1 que é basicamente.
#l1, Lasso: soma dos quadrados dos resíduos + penalidade * | inclinação |
#Como a regularização é aplicada aqui, durante o treinamento e não na validação, isso penaliza mais os erros de treino e
#faz com que as métricas fiquem diferentes, como abaixo

lgb.plot_metric(best_lgbm, metric=None)

plot_tree(best_lgbm)
plt.gcf().set_size_inches(10, 5)
plt.show()

def evaluate_model(y_train, prediction):
  accuracy=100-(mean_absolute_percentage_error(y_train, prediction_lgbm_train)*100)
  print('\n\nLGBM Acurácia Média TREINO: ', round(accuracy,2),'%.')
  print('LGBM Erro Médio Absoluto (MAE) TREINO: ', mean_absolute_error(y_train, prediction_lgbm_train))
  print('LGBM Erro Quadrático Médio (MSE) TREINO: ', mean_squared_error(y_train, prediction_lgbm_train))
  print('LGBM Raiz do Erro Quadrático Médio (RMSE) TREINO: ', mean_squared_error(y_train, prediction_lgbm_train, squared=False ))
  print('LGBM Porcentagem do Erro Médio Absoluto (MAPE) TREINO: ',  round(mean_absolute_percentage_error(y_train, prediction_lgbm_train)*100, 2),'%')
  #Proporção de variação dos dados explicados pelo modelo. Quanto mais perto de 1, melhor
  print('LGBM Explained Variance Score TREINO: ', metrics.explained_variance_score(y_train, prediction_lgbm_train))
  #Maior erro entre previsões
  print('LGBM Erro Máximo TREINO: ', metrics.max_error(y_train, prediction_lgbm_train))
  print('LGBM Erro Logarítmico Médio TREINO: ', metrics.mean_squared_log_error(y_train, prediction_lgbm_train))
  #Mediana do erro entre valores previstos e valores reais
  print('LGBM Mediana do Erro Absoluto TREINO: ', metrics.median_absolute_error(y_train, prediction_lgbm_train))

def plot_predictions(train_dates, y_train, prediction):
  df_train = pd.DataFrame({"Ano": train_data['Ano'], "Original": y_train, "Predição": prediction_lgbm_train })
  figure, ax = plt.subplots(figsize=(10, 5))
  df_train.plot(ax=ax, label="Original", x="Ano", y="Original")
  df_train.plot(ax=ax, label="Predição", x="Ano", y="Predição")
  plt.legend(["Original", "Predição"])
  plt.title("LGBM + Divisão Simplificada Treino + LAG")
  plt.show()
  
  
def evaluate_model(y_test, prediction):
    accuracy=100-(mean_absolute_percentage_error(y_test, prediction_lgbm_test)*100)
    print('\n\nLGBM Acurácia Média TESTE: ', round(accuracy,2),'%.')
    print('LGBM Erro Médio Absoluto (MAE) TESTE: ', metrics.mean_absolute_error(y_test, prediction_lgbm_test))
    print('LGBM Erro Quadrático Médio (MSE) TESTE: ', metrics.mean_squared_error(y_test, prediction_lgbm_test))
    print('LGBM Raiz do Erro Quadrático Médio (RMSE) TESTE: ', metrics.mean_squared_error(y_test, prediction_lgbm_test, squared=False ))
    print('LGBM Porcentagem do Erro Médio Absoluto (MAPE) TESTE: ',  round(mean_absolute_percentage_error(y_test, prediction_lgbm_test)*100, 2),'%')
    #Proporção de variação dos dados explicados pelo modelo. Quanto mais perto de 1, melhor
    print('LGBM Explained Variance Score TESTE: ', metrics.explained_variance_score(y_test, prediction_lgbm_test))
    #Maior erro entre previsões
    print('LGBM Erro Máximo TESTE: ', metrics.max_error(y_test, prediction_lgbm_test))
    print('LGBM Erro Logarítmico Médio TESTE: ', metrics.mean_squared_log_error(y_test, prediction_lgbm_test))
    #Mediana do erro entre valores previstos e valores reais
    print('LGBM Mediana do Erro Absoluto TESTE: ', metrics.median_absolute_error(y_test, prediction_lgbm_test))

def plot_predictions(testing_dates, y_test, prediction):
  df_test_lgbm = pd.DataFrame({"data": testing_dates, "Original": y_test, "Predição": prediction_lgbm_test })
  figure, ax = plt.subplots(figsize=(10, 5))
  df_test_lgbm.plot(ax=ax, label="Original", x="data", y="Original")
  df_test_lgbm.plot(ax=ax, label="Predição", x="data", y="Predição")
  plt.legend(["Original", "Predição"])
  plt.title("LGBM  - Validação em Divisão Simplificada")
  plt.show()

from sklearn import metrics
# Validação dos Resultados
prediction_lgbm_test = best_lgbm.predict(X_test)
plot_predictions(testing_dates, y_test, prediction_lgbm_test)
evaluate_model(y_test, prediction_lgbm_test)

#Prevendo o próximo mês
prediction_lgbm_new = best_lgbm.predict(predict)
print(prediction_lgbm_new)

#====================================================================================================================



#===============================================RANDOM FOREST=============================================================
#RandomForest
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import numpy as np


print("Starting model train..")
# Number of trees in random forest
n_estimators_rf = [10,50,70,100]
# Number of features to consider at every split
max_features_rf = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth_rf = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth_rf.append(None)
# Minimum number of samples required to split a node
min_samples_split_rf = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf_rf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap_rf = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators_rf,
               'max_features': max_features_rf,
               'max_depth': max_depth_rf,
               'min_samples_split': min_samples_split_rf,
               'min_samples_leaf': min_samples_leaf_rf,
               'bootstrap': bootstrap_rf,
               'criterion' :[ 'absolute_error']
               }


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1, scoring='neg_median_absolute_error')
# Fit the random search model
rf_random.fit(X_train, y_train)

print(rf_random.best_estimator_)
print(rf_random.best_params_)

# Treine o modelo novamente, porém agora com os melhores  hiperparâmetros
best_rf_random = RandomForestRegressor(**rf_random.best_params_)
best_rf_random.fit(X_train, y_train)

# Faça previsões sobre os dados de teste
y_pred_rf_random = best_rf_random.predict(X_test)
x_pred_slide_lgbm = best_rf_random.predict(X_train)

#Verificar a importância das colunas
plt.figure(figsize=(12, 8))
pyplot.barh(X_test.columns, best_rf_random.feature_importances_)
plt.title("IMPORTÂNCIA DOS ATRIBUTOS NA MODELAGEM", fontsize=20)
# Aumentar o tamanho das fontes dos rótulos dos eixos
plt.xlabel("Importância", fontsize=18)
plt.ylabel("Atributos", fontsize=18)
# Aumentar o tamanho das fontes dos ticks dos eixos
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
pyplot.show()

plt.figure(figsize=(12, 8)) # Define o tamanho da figura (largura, altura)
plt.scatter(X_test['Mes'].values, y_test, color = 'red')
plt.scatter(X_test['Mes'].values, y_pred_rf_random, color = 'green')
plt.title('Random Forest')
plt.xlabel('Ano')
plt.ylabel("Qt. item")
plt.legend() # Adiciona a legenda
plt.show()

def evaluate_model(y_train, prediction):
    accuracy=100-(mean_absolute_percentage_error(y_train, prediction_rf_train)*100)
    print('\n\Random Forest Acurácia Média TREINO: ', round(accuracy,2),'%.')
    print('Random Forest Erro Médio Absoluto (MAE) TREINO: ', metrics.mean_absolute_error(y_train, prediction_rf_train))
    print('Random Forest Erro Quadrático Médio (MSE) TREINO: ', metrics.mean_squared_error(y_train, prediction_rf_train))
    print('Random Forest Raiz do Erro Quadrático Médio (RMSE) TREINO: ', metrics.mean_squared_error(y_train, prediction_rf_train, squared=False ))
    print('Random Forest Porcentagem do Erro Médio Absoluto (MAPE) TREINO: ',  round(mean_absolute_percentage_error(y_train, prediction_rf_train)*100, 2),'%')
    #Proporção de variação dos dados explicados pelo modelo. Quanto mais perto de 1, melhor
    print('Random Forest Explained Variance Score TREINO: ', metrics.explained_variance_score(y_train, prediction_rf_train))
    #Maior erro entre previsões
    print('Random Forest Erro Máximo TREINO: ', metrics.max_error(y_train, prediction_rf_train))
    print('Random Forest Erro Logarítmico Médio TREINO: ', metrics.mean_squared_log_error(y_train, prediction_rf_train))
    #Mediana do erro entre valores previstos e valores reais
    print('Random Forest Mediana do Erro Absoluto TREINO: ', metrics.median_absolute_error(y_train, prediction_rf_train))

def plot_predictions(testing_dates, y_train, prediction):
  df_train_rf_random = pd.DataFrame({"data": train_data['Ano'], "Original": y_train, "Predição": prediction_rf_train })
  figure, ax = plt.subplots(figsize=(10, 5))
  df_train_rf_random.plot(ax=ax, label="Original", x="data", y="Original")
  df_train_rf_random.plot(ax=ax, label="Predição", x="data", y="Predição")
  plt.legend(["Original", "Predição"])
  plt.title("Random Forest + Treino em Divisão Simplificada  + LAG")
  plt.show()

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error,\
  mean_squared_error

def evaluate_model(y_test, prediction):
  print('Random Forest Erro Médio Absoluto (MAE) Test:', metrics.mean_absolute_error(y_test, prediction_rf))
  print('Random Forest Erro Quadrático Médio (MSE) Test:', metrics.mean_squared_error(y_test, prediction_rf))
  print('Random Forest Raiz do Erro Quadrático Médio (RMSE) Test:', metrics.mean_squared_error(y_test, prediction_rf, squared=False ))
  print('Random Forest Porcentagem do Erro Médio Absoluto (MAPE) Test:',  round(mean_absolute_percentage_error(y_test, prediction_rf)*100, 2),'%')
  #Proporção de variação dos dados explicados pelo modelo. Quanto mais perto de 1, melhor
  print('Random Forest Explained Variance Score Test:', metrics.explained_variance_score(y_test, prediction_rf))
  #Maior erro entre previsões
  print('Random Forest Erro Máximo Test:', metrics.max_error(y_test, prediction_rf))
  print('Random Forest Erro Logarítmico Médio Test:', metrics.mean_squared_log_error(y_test, prediction_rf))
  #Mediana do erro entre valores previstos e valores reais
  print('Random Forest Mediana do Erro Absoluto Test:', metrics.median_absolute_error(y_test, prediction_rf))

def plot_predictions(testing_dates, y_test, prediction):
  df_test_rf_random = pd.DataFrame({"data": testing_dates, "Original": y_test, "Predição": prediction_rf })
  figure, ax = plt.subplots(figsize=(10, 5))
  df_test_rf_random.plot(ax=ax, label="Original", x="data", y="Original")
  df_test_rf_random.plot(ax=ax, label="Predição", x="data", y="Predição")
  plt.legend(["Original", "Predição"])
  plt.title("RandomForest - Validação em Divisão Simplificada")
  plt.show()

#Verificar a importância das colunas
# Validando os resultados

prediction_rf = best_rf_random.predict(X_test)
plot_predictions(testing_dates, y_test, prediction_rf)
evaluate_model(y_test, prediction_rf)

#Prevendo o próximo mês
prediction_rf_new = best_rf_random.predict(predict)
print(prediction_rf_new)

#======================================================================================================================

#======================================================================================================================

#                                               JANELA DESLIZANTE

#======================================================================================================================

#===============================================XGBOOST=============================================================
#XGBRegressor - Sliding Window Validation
n_split = 4
tss = TimeSeriesSplit(n_splits=n_split, test_size=13, gap=0)
monthly_sales = monthly_sales.sort_index()

fold = 0
preds = []
scores_train = []
scores_test = []
for train_idx, val_idx in tss.split(monthly_sales):
    fold = fold + 1
    print(f'\nFold {fold}\n')
    train_slide = monthly_sales.iloc[train_idx]
    test_slide = monthly_sales.iloc[val_idx]
    FEATURES_slide = ['Ano',	'Mes','lag_1',	'diff1',	'diff2']
    scaler_slide=MinMaxScaler()
    scaler_slide.fit(train_slide[FEATURES_slide])

    data_normalized_train_slide=scaler.transform(train_slide[FEATURES_slide])
    data_normalized_train_slide=pd.DataFrame(data_normalized_train_slide, index=train_slide[FEATURES_slide].index, columns=train_slide[FEATURES_slide].columns)
    data_normalized_test_slide=scaler.transform(test_slide[FEATURES_slide])
    data_normalized_test_slide=pd.DataFrame(data_normalized_test_slide, index=test_slide[FEATURES_slide].index, columns=test_slide[FEATURES_slide].columns)

    TARGET_slide = 'Qt. item'

    X_train_slide_xgb= data_normalized_train_slide
    y_train_slide_xgb = train_slide[TARGET_slide]
    X_test_slide_xgb = data_normalized_test_slide
    y_test_slide_xgb = test_slide[TARGET_slide]
    print(len(X_train_slide_xgb))
    print(len(y_train_slide_xgb))
    print(len(X_test_slide_xgb))
    print(len(y_test_slide_xgb))


    '''
    {'colsample_bytree': 1.0, 'learning_rate': 0.3, 'max_depth': 3, 'n_estimators': 300}
    '''

    reg_slide_xgb = XGBRegressor(base_score=0.7, booster='gbtree',
                           n_estimators=8,
                           early_stopping_rounds=100,
                           max_depth=3,
                           learning_rate=0.3,
                           colsample_bytree= 1.0,
                           eval_metric= ['mae'])

    reg_slide_xgb.fit(X_train_slide_xgb, y_train_slide_xgb, eval_set=[(X_train_slide_xgb, y_train_slide_xgb), (X_test_slide_xgb, y_test_slide_xgb)], )

    y_pred_slide_xgb = reg_slide_xgb.predict(X_test_slide_xgb)
    preds.append(y_pred_slide_xgb)
    score_train_slide_xgb = mean_absolute_error(y_train_slide_xgb, reg_slide_xgb.predict(X_train_slide_xgb))
    scores_train.append(score_train_slide_xgb)
    score_test_slide_xgb = mean_absolute_error(y_test_slide_xgb, y_pred_slide_xgb)
    scores_test.append(score_test_slide_xgb)

# Calculate the average MAE and standard deviation for train and test data
average_mae_train = np.mean(scores_train)
std_mae_train = np.std(scores_train)
average_mae_test = np.mean(scores_test)
std_mae_test = np.std(scores_test)
print(f'Average MAE Train: {average_mae_train}')
print(f'Standard Deviation of MAE Train: {std_mae_train}')
print(f'Average MAE Test: {average_mae_test}')
print(f'Standard Deviation of MAE Test: {std_mae_test}')

#Plotando a árvore do modelo para entender melhor como este trabalha
from xgboost import plot_tree
plot_tree(reg_slide_xgb)
plt.gcf().set_size_inches(18.5, 10.5)
plt.show()

def evaluate_model(y_test, prediction):
  print('XGBOOST Erro Médio Absoluto (MAE) TREINO: ', mean_absolute_error(y_train_slide_xgb, prediction_slide_xgb_train))
  print('XGBOOST Erro Quadrático Médio (MSE) TREINO: ', mean_squared_error(y_train_slide_xgb, prediction_slide_xgb_train))
  print('XGBOOST Raiz do Erro Quadrático Médio (RMSE) TREINO: ', mean_squared_error(y_train_slide_xgb, prediction_slide_xgb_train, squared=False ))
  print('XGBOOST Porcentagem do Erro Médio Absoluto (MAPE) TREINO: ',  round(mean_absolute_percentage_error(y_train_slide_xgb, prediction_slide_xgb_train)*100, 2),'%')
  #Proporção de variação dos dados explicados pelo modelo. Quanto mais perto de 1, melhor
  print('XGBOOST Explained Variance Score TREINO: ', metrics.explained_variance_score(y_train_slide_xgb, prediction_slide_xgb_train))
  #Maior erro entre previsões
  print('XGBOOST Erro Máximo TREINO: ', metrics.max_error(y_train_slide_xgb, prediction_slide_xgb_train))
  print('XGBOOST Erro Logarítmico Médio TREINO: ', metrics.mean_squared_log_error(y_train_slide_xgb, prediction_slide_xgb_train))
  #Mediana do erro entre valores previstos e valores reais
  print('XGBOOST Mediana do Erro Absoluto TREINO: ', metrics.median_absolute_error(y_train_slide_xgb, prediction_slide_xgb_train))

def plot_predictions(testing_dates, y_test, prediction):

  df_xgb_train = pd.DataFrame({"Ano": train_data['Ano'], "Original": y_train_slide_xgb, "Predição": prediction_slide_xgb_train })
  figure, ax = plt.subplots(figsize=(10, 5))
  df_xgb_train.plot(ax=ax, label="Original", x="Ano", y="Original")
  df_xgb_train.plot(ax=ax, label="Predição", x="Ano", y="Predição")
  plt.legend(["Original", "Predição"])
  plt.title("XGBoost + Divisão Simplificada Treino + LAG")
  plt.show()
  
from sklearn.metrics import r2_score
# Validando os resultados
prediction_slide_xgb_train = reg_slide_xgb.predict(X_train_slide_xgb).round()
plot_predictions(train_data['Ano'], y_train_slide_xgb, prediction_slide_xgb_train)
evaluate_model(y_train_slide_xgb, prediction_slide_xgb_train)


def evaluate_model(y_test, prediction):
  accuracy=100-(mean_absolute_percentage_error(y_test_slide_xgb, prediction_slide_xgb_test)*100)
  print('\nXGBOOST Acurácia Média TESTE: ', round(accuracy,2),'%.')
  print('XGBOOST Erro Médio Absoluto (MAE) TESTE: ', metrics.mean_absolute_error(y_test_slide_xgb, prediction_slide_xgb_test))
  print('XGBOOST Erro Quadrático Médio (MSE) TESTE: ', metrics.mean_squared_error(y_test_slide_xgb, prediction_slide_xgb_test))
  print('XGBOOST Raiz do Erro Quadrático Médio (RMSE) TESTE: ', metrics.mean_squared_error(y_test_slide_xgb, prediction_slide_xgb_test, squared=False ))
  print('XGBOOST Porcentagem do Erro Médio Absoluto (MAPE) TESTE: ',  round(mean_absolute_percentage_error(y_test_slide_xgb, prediction_slide_xgb_test)*100, 2),'%')
  #Proporção de variação dos dados explicados pelo modelo. Quanto mais perto de 1, melhor
  print('XGBOOST Explained Variance Score TESTE: ', metrics.explained_variance_score(y_test_slide_xgb, prediction_slide_xgb_test))
  #Maior erro entre previsões
  print('XGBOOST Erro Máximo TESTE: ', metrics.max_error(y_test_slide_xgb, prediction_slide_xgb_test))
  print('XGBOOST Erro Logarítmico Médio TESTE: ', metrics.mean_squared_log_error(y_test_slide_xgb, prediction_slide_xgb_test))
  #Mediana do erro entre valores previstos e valores reais
  print('XGBOOST Mediana do Erro Absoluto TESTE: ', metrics.median_absolute_error(y_test_slide_xgb, prediction_slide_xgb_test))

def plot_predictions(testing_dates, y_test, prediction):
  df_test_slide_xgb = pd.DataFrame({"data": testing_dates, "actual": y_test_slide_xgb, "prediction": prediction_slide_xgb_test })
  figure, ax = plt.subplots(figsize=(10, 5))
  df_test_slide_xgb.plot(ax=ax, label="Actual", x="data", y="actual")
  df_test_slide_xgb.plot(ax=ax, label="Prediction", x="data", y="prediction")
  plt.legend(["Actual", "Prediction"])
  plt.title("#XGBRegressor - Janela Deslizante")
  plt.show()
  
from sklearn.metrics import r2_score
# Validando os resultados
prediction_slide_xgb_test = reg_slide_xgb.predict(X_test_slide_xgb).round()
plot_predictions(testing_dates, y_test_slide_xgb, prediction_slide_xgb_test)
evaluate_model(y_test_slide_xgb, prediction_slide_xgb_test)

#Verificar a importância das colunas
plt.figure(figsize=(12, 8))
pyplot.barh(X_test_slide_xgb.columns, reg_slide_xgb.feature_importances_)
plt.title("IMPORTÂNCIA DOS ATRIBUTOS NA MODELAGEM", fontsize=20)
# Aumentar o tamanho das fontes dos rótulos dos eixos
plt.xlabel("Importância", fontsize=18)
plt.ylabel("Atributos", fontsize=18)
# Aumentar o tamanho das fontes dos ticks dos eixos
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
pyplot.show()

#VERIFICAÇÃO DE OVERFITTING

lm_xgboost = average_mae_train+std_mae_train
mae_teste_xgboost = metrics.mean_absolute_error(y_test_slide_xgb, prediction_slide_xgb_test)

if mae_teste_xgboost<=lm_xgboost:
  print("O modelo não está sofrendo de overfitting")
else:
  print("O modelo está sofrendo de overfitting")

results_slide_xgb = reg_slide_xgb.evals_result()
print(results_slide_xgb)
epochs_slide_xgb = len(results_slide_xgb['validation_0']['mae'])
x_axis_slide_xgb = range(0, epochs_slide_xgb)

fig, ax = pyplot.subplots()
ax.plot(x_axis_slide_xgb, results_slide_xgb['validation_0']['mae'], label='Treino')
ax.plot(x_axis_slide_xgb, results_slide_xgb['validation_1']['mae'], label='Validação')
ax.legend()
pyplot.ylabel('MAE')
pyplot.title('VALIDAÇÃO MAE - XGBOOST')
pyplot.show()

#Prevendo o próximo mês
prediction__slide_xgb_new = reg_slide_xgb.predict(predict)
print(prediction__slide_xgb_new)
#====================================================================================================================


#===============================================LIGHTGBM=============================================================
#LGBM
n_split = 4
tss = TimeSeriesSplit(n_splits=n_split, test_size=13, gap=0)
#organiza os valores por index
monthly_sales = monthly_sales.sort_index()


fold = 0
preds_slide_lgbm = []
scores = []
for train_idx, val_idx in tss.split(monthly_sales):
    train_slide_lgbm = monthly_sales.iloc[train_idx]
    test_slide_lgbm = monthly_sales.iloc[val_idx]

    FEATURES_slide_lgbm = ['Ano',	'Mes','lag_1',	'diff1',	'diff2']

    scaler_slide_lgbm=MinMaxScaler()
    scaler_slide_lgbm.fit(train_slide_lgbm[FEATURES_slide_lgbm])

    data_normalized_train_slide_lgbm=scaler.transform(train_slide_lgbm[FEATURES_slide_lgbm])
    data_normalized_train_slide_lgbm=pd.DataFrame(data_normalized_train_slide_lgbm, index=train_slide_lgbm[FEATURES_slide_lgbm].index, columns=train_slide_lgbm[FEATURES_slide_lgbm].columns)
    print(data_normalized_train_slide_lgbm)


    data_normalized_test_slide_lgbm=scaler.transform(test_slide_lgbm[FEATURES_slide_lgbm])
    data_normalized_test_slide_lgbm=pd.DataFrame(data_normalized_test_slide_lgbm, index=test_slide_lgbm[FEATURES_slide_lgbm].index, columns=test_slide_lgbm[FEATURES_slide_lgbm].columns)
    print(data_normalized_test_slide_lgbm)

    TARGET_slide_lgbm = 'Qt. item'

    X_train_slide_lgbm = data_normalized_train_slide_lgbm
    y_train_slide_lgbm = train_slide_lgbm[TARGET_slide_lgbm]

    X_test_slide_lgbm = data_normalized_test_slide_lgbm
    y_test_slide_lgbm = test_slide_lgbm[TARGET_slide_lgbm]

    '''
    {'colsample_bytree': 1.0, 'learning_rate': 0.3, 'max_depth': 3, 'n_estimators': 300}
    '''
    # LGBM
    model_slide_lgbm = lgb.LGBMRegressor()
    parameters_slide_lgbm = {
       "max_depth": [3, 5, 7],
      "num_leaves": [10],
      'learning_rate': [0.01, 0.05, 0.1],   # Learning rate for gradient boosting
      'n_estimators': [50, 60, 100],         # Number of boosting iterations
      "colsample_bytree": [0.3],
      'objective': ['mae'],
      'metric': ['mae'],
    }


    grid_search_slide_lgbm = GridSearchCV(estimator=model_slide_lgbm, param_grid=parameters_slide_lgbm)
    grid_search_slide_lgbm.fit(X_train_slide_lgbm, y_train_slide_lgbm, eval_set=[(X_train_slide_lgbm, y_train_slide_lgbm), (X_test_slide_lgbm, y_test_slide_lgbm)])
    print(grid_search_slide_lgbm.best_estimator_)
    print(grid_search_slide_lgbm.best_params_)

    # Treine o modelo novamente, porém agora com os melhores  hiperparâmetros
    best_slide_lgbm = LGBMRegressor(**grid_search_slide_lgbm.best_params_)
    best_slide_lgbm.fit(X_train_slide_lgbm, y_train_slide_lgbm, eval_set=[(X_train_slide_lgbm, y_train_slide_lgbm), (X_test_slide_lgbm, y_test_slide_lgbm)])


    # Faça previsões sobre os dados de teste
    y_pred_slide_lgbm = best_slide_lgbm.predict(X_test_slide_lgbm)
    x_pred_slide_lgbm = best_slide_lgbm.predict(X_train_slide_lgbm)


    preds_slide_lgbm.append(y_pred_slide_lgbm)
    score_slide_lgbm = mean_absolute_error(y_test_slide_lgbm, y_pred_slide_lgbm)
    scores.append(score_slide_lgbm)

#Verificar a importância das colunas
plt.figure(figsize=(12, 8))
pyplot.barh(X_test_slide_lgbm.columns, best_slide_lgbm.feature_importances_)
plt.title("IMPORTÂNCIA DOS ATRIBUTOS NA MODELAGEM", fontsize=20)
# Aumentar o tamanho das fontes dos rótulos dos eixos
plt.xlabel("Importância", fontsize=18)
plt.ylabel("Atributos", fontsize=18)
# Aumentar o tamanho das fontes dos ticks dos eixos
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
pyplot.show()


# A métrica padrão para LGBM é a L1 que é basicamente.
#l1, Lasso: soma dos quadrados dos resíduos + penalidade * | inclinação |
#Como a regularização é aplicada aqui, durante o treinamento e não na validação, isso penaliza mais os erros de treino e
#faz com que as métricas fiquem diferentes, como abaixo

lgb.plot_metric(best_slide_lgbm)

def evaluate_model(y_train, prediction):
  accuracy=100-(mean_absolute_percentage_error(y_train_slide_lgbm, prediction_slide_lgbm_train)*100)
  print('\n\nLGBM Acurácia Média TREINO: ', round(accuracy,2),'%.')
  print('LGBM Erro Médio Absoluto (MAE) TREINO: ', mean_absolute_error(y_train_slide_lgbm, prediction_slide_lgbm_train))
  print('LGBM Erro Quadrático Médio (MSE) TREINO: ', mean_squared_error(y_train_slide_lgbm, prediction_slide_lgbm_train))
  print('LGBM Raiz do Erro Quadrático Médio (RMSE) TREINO: ', mean_squared_error(y_train_slide_lgbm, prediction_slide_lgbm_train, squared=False ))
  print('LGBM Porcentagem do Erro Médio Absoluto (MAPE) TREINO: ',  round(mean_absolute_percentage_error(y_train_slide_lgbm, prediction_slide_lgbm_train)*100, 2),'%')
  #Proporção de variação dos dados explicados pelo modelo. Quanto mais perto de 1, melhor
  print('LGBM Explained Variance Score TREINO: ', metrics.explained_variance_score(y_train_slide_lgbm, prediction_slide_lgbm_train))
  #Maior erro entre previsões
  print('LGBM Erro Máximo TREINO: ', metrics.max_error(y_train_slide_lgbm, prediction_slide_lgbm_train))
  print('LGBM Erro Logarítmico Médio TREINO: ', metrics.mean_squared_log_error(y_train_slide_lgbm, prediction_slide_lgbm_train))
  #Mediana do erro entre valores previstos e valores reais
  print('LGBM Mediana do Erro Absoluto TREINO: ', metrics.median_absolute_error(y_train_slide_lgbm, prediction_slide_lgbm_train))

def plot_predictions(train_dates, y_train, prediction):
  df_slide_train = pd.DataFrame({"Ano": train_data['Ano'], "Original": y_train_slide_lgbm, "Predição": prediction_slide_lgbm_train })
  figure, ax = plt.subplots(figsize=(10, 5))
  df_slide_train.plot(ax=ax, label="Original", x="Ano", y="Original")
  df_slide_train.plot(ax=ax, label="Predição", x="Ano", y="Predição")
  plt.legend(["Original", "Predição"])
  plt.title("LGBM - Divisão Simplificada Treino")
  plt.show()
  
from sklearn.metrics import r2_score
# Validando os resultados
prediction_slide_lgbm_train = best_slide_lgbm.predict(X_train_slide_lgbm).round()
plot_predictions(train_data['Ano'], y_train_slide_lgbm, prediction_slide_lgbm_train)
evaluate_model(y_train_slide_lgbm, prediction_slide_lgbm_train)


def evaluate_model(y_test, prediction):
    accuracy=100-(mean_absolute_percentage_error(y_test_slide_lgbm, prediction_lgbm_slide_test)*100)
    print('\n\nLGBM Acurácia Média TESTE: ', round(accuracy,2),'%.')
    print('LGBM Erro Médio Absoluto (MAE) TESTE: ', metrics.mean_absolute_error(y_test_slide_lgbm, prediction_lgbm_slide_test))
    print('LGBM Erro Quadrático Médio (MSE) TESTE: ', metrics.mean_squared_error(y_test_slide_lgbm, prediction_lgbm_slide_test))
    print('LGBM Raiz do Erro Quadrático Médio (RMSE) TESTE: ', metrics.mean_squared_error(y_test_slide_lgbm, prediction_lgbm_slide_test, squared=False ))
    print('LGBM Porcentagem do Erro Médio Absoluto (MAPE) TESTE: ',  round(mean_absolute_percentage_error(y_test_slide_lgbm, prediction_lgbm_slide_test)*100, 2),'%')
    #Proporção de variação dos dados explicados pelo modelo. Quanto mais perto de 1, melhor
    print('LGBM Explained Variance Score TESTE: ', metrics.explained_variance_score(y_test_slide_lgbm, prediction_lgbm_slide_test))
    #Maior erro entre previsões
    print('LGBM Erro Máximo TESTE: ', metrics.max_error(y_test_slide_lgbm, prediction_lgbm_slide_test))
    print('LGBM Erro Logarítmico Médio TESTE: ', metrics.mean_squared_log_error(y_test_slide_lgbm, prediction_lgbm_slide_test))
    #Mediana do erro entre valores previstos e valores reais
    print('LGBM Mediana do Erro Absoluto TESTE: ', metrics.median_absolute_error(y_test_slide_lgbm, prediction_lgbm_slide_test))

def plot_predictions(testing_dates, y_test, prediction):
  df_test_slide_lgbm = pd.DataFrame({"data": testing_dates, "actual": y_test_slide_lgbm, "prediction": y_pred_slide_lgbm })
  figure, ax = plt.subplots(figsize=(10, 5))
  df_test_slide_lgbm.plot(ax=ax, label="Actual", x="data", y="actual")
  df_test_slide_lgbm.plot(ax=ax, label="Prediction", x="data", y="prediction")
  plt.legend(["Actual", "Prediction"])
  plt.title("#LGBM - Janela Deslizante")
  plt.show()
  
from sklearn import metrics
# Validação dos Resultados
prediction_lgbm_slide_test = best_slide_lgbm.predict(X_test_slide_lgbm)
plot_predictions(testing_dates, y_test_slide_lgbm, prediction_lgbm_slide_test)
evaluate_model(y_test_slide_lgbm, prediction_lgbm_slide_test)


#Prevendo o próximo mês
prediction__slide_lgbm_new = best_slide_lgbm.predict(predict)
print(prediction__slide_lgbm_new)

#====================================================================================================================

#===============================================RANDOM FOREST=============================================================
#Random Forest - Expanding Window Validation

#RF
n_split = 4
tss = TimeSeriesSplit(n_splits=n_split, test_size=13, gap=0)
monthly_sales = monthly_sales.sort_index()


fold = 0
preds = []
scores = []
for train_idx, val_idx in tss.split(monthly_sales):
    train = monthly_sales.iloc[train_idx]
    test = monthly_sales.iloc[val_idx]

    FEATURES = ['Ano',	'Mes','lag_1',	'diff1',	'diff2']

    scaler=MinMaxScaler()
    scaler.fit(train[FEATURES])

    data_normalized_train=scaler.transform(train[FEATURES])
    data_normalized_train=pd.DataFrame(data_normalized_train, index=train[FEATURES].index, columns=train[FEATURES].columns)
    print(data_normalized_train)


    data_normalized_test=scaler.transform(test[FEATURES])
    data_normalized_test=pd.DataFrame(data_normalized_test, index=test[FEATURES].index, columns=test[FEATURES].columns)
    print(data_normalized_test)

    TARGET = 'Qt. item'

    X_train_slide_rf = data_normalized_train
    y_train_slide_rf = train[TARGET]

    X_test_slide_rf = data_normalized_test
    y_test_slide_rf = test[TARGET]

print("Starting model train..")
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
              'max_features': max_features,
              'max_depth': max_depth,
              'min_samples_split': min_samples_split,
              'min_samples_leaf': min_samples_leaf,
              'bootstrap': bootstrap}


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random_slide = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random_slide.fit(X_train_slide_rf, y_train_slide_rf)

print(rf_random_slide.best_estimator_)
print(rf_random_slide.best_params_)

# Treine o modelo novamente, porém agora com os melhores  hiperparâmetros
best_rf_random_slide = RandomForestRegressor(**rf_random_slide.best_params_)
best_rf_random_slide.fit(X_train_slide_rf, y_train_slide_rf)

y_pred_slide_rf= rf_random_slide.predict(X_test_slide_rf)
preds.append(y_pred_slide_rf)
score = mean_absolute_error(y_test_slide_rf, y_pred_slide_rf)
scores.append(score)

#Verificar a importância das colunas
plt.figure(figsize=(12, 8))
pyplot.barh(X_test_slide_rf.columns, best_rf_random_slide.feature_importances_)
plt.title("IMPORTÂNCIA DOS ATRIBUTOS NA MODELAGEM", fontsize=20)
# Aumentar o tamanho das fontes dos rótulos dos eixos
plt.xlabel("Importância", fontsize=18)
plt.ylabel("Atributos", fontsize=18)
# Aumentar o tamanho das fontes dos ticks dos eixos
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
pyplot.show()

def evaluate_model(y_train, prediction):
    accuracy=100-(mean_absolute_percentage_error(y_train_slide_rf, prediction_slide_rf_train)*100)
    print('\n\Random Forest Acurácia Média TREINO: ', round(accuracy,2),'%.')
    print('Random Forest Erro Médio Absoluto (MAE) TREINO: ', metrics.mean_absolute_error(y_train_slide_rf, prediction_slide_rf_train))
    print('Random Forest Erro Quadrático Médio (MSE) TREINO: ', metrics.mean_squared_error(y_train_slide_rf, prediction_slide_rf_train))
    print('Random Forest Raiz do Erro Quadrático Médio (RMSE) TREINO: ', metrics.mean_squared_error(y_train_slide_rf, prediction_slide_rf_train, squared=False ))
    print('Random Forest Porcentagem do Erro Médio Absoluto (MAPE) TREINO: ',  round(mean_absolute_percentage_error(y_train_slide_rf, prediction_slide_rf_train)*100, 2),'%')
    #Proporção de variação dos dados explicados pelo modelo. Quanto mais perto de 1, melhor
    print('Random Forest Explained Variance Score TREINO: ', metrics.explained_variance_score(y_train_slide_rf, prediction_slide_rf_train))
    #Maior erro entre previsões
    print('Random Forest Erro Máximo TREINO: ', metrics.max_error(y_train_slide_rf, prediction_slide_rf_train))
    print('Random Forest Erro Logarítmico Médio TREINO: ', metrics.mean_squared_log_error(y_train_slide_rf, prediction_slide_rf_train))
    #Mediana do erro entre valores previstos e valores reais
    print('Random Forest Mediana do Erro Absoluto TREINO: ', metrics.median_absolute_error(y_train_slide_rf, prediction_slide_rf_train))

def plot_predictions(testing_dates, y_train, prediction):
  df_train_rf_random = pd.DataFrame({"data": train_data['Ano'], "Original": y_train_slide_rf, "Predição": prediction_slide_rf_train })
  figure, ax = plt.subplots(figsize=(10, 5))
  df_train_rf_random.plot(ax=ax, label="Original", x="data", y="Original")
  df_train_rf_random.plot(ax=ax, label="Predição", x="data", y="Predição")
  plt.legend(["Original", "Predição"])
  plt.title("Random Forest + Treino em Divisão Simplificada  + LAG")
  plt.show()
  



from sklearn import metrics
# Validação dos Resultados
prediction_slide_rf_train = best_rf_random.predict(X_train_slide_rf)
plot_predictions(train_data['Ano'], y_train_slide_rf, prediction_rf_train)
evaluate_model(y_train_slide_rf, prediction_rf_train)



from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error,\
  mean_squared_error

def evaluate_model(y_test, prediction):
  accuracy=100-(mean_absolute_percentage_error(y_test_slide_rf, prediction_slide_rf_test)*100)
  print('\n\Random Forest Acurácia Média TESTE: ', round(accuracy,2),'%.')
  print('Random Forest Erro Médio Absoluto (MAE) TESTE: ', metrics.mean_absolute_error(y_test_slide_rf, prediction_slide_rf_test))
  print('Random Forest Erro Quadrático Médio (MSE) TESTE: ', metrics.mean_squared_error(y_test_slide_rf, prediction_slide_rf_test))
  print('Random Forest Raiz do Erro Quadrático Médio (RMSE) TESTE: ', metrics.mean_squared_error(y_test_slide_rf, prediction_slide_rf_test, squared=False ))
  print('Random Forest Porcentagem do Erro Médio Absoluto (MAPE) TESTE: ',  round(mean_absolute_percentage_error(y_test_slide_rf, prediction_slide_rf_test)*100, 2),'%')
  #Proporção de variação dos dados explicados pelo modelo. Quanto mais perto de 1, melhor
  print('Random Forest Explained Variance Score TESTE: ', metrics.explained_variance_score(y_test_slide_rf, prediction_slide_rf_test))
  #Maior erro entre previsões
  print('Random Forest Erro Máximo TESTE: ', metrics.max_error(y_test_slide_rf, prediction_slide_rf_test))
  print('Random Forest Erro Logarítmico Médio TESTE: ', metrics.mean_squared_log_error(y_test_slide_rf, prediction_slide_rf_test))
  #Mediana do erro entre valores previstos e valores reais
  print('Random Forest Mediana do Erro Absoluto TESTE: ', metrics.median_absolute_error(y_test_slide_rf, prediction_slide_rf_test))


def plot_predictions(testing_dates, y_test, prediction):
  df_test = pd.DataFrame({"data": testing_dates, "actual": y_test_slide_rf, "prediction": prediction_slide_rf_test })
  figure, ax = plt.subplots(figsize=(10, 5))
  df_test.plot(ax=ax, label="Actual", x="data", y="actual")
  df_test.plot(ax=ax, label="Prediction", x="data", y="prediction")
  plt.legend(["Actual", "Prediction"])
  plt.title("RandomForest - Janela Deslizante")
  plt.show()

# Validando os resultados

prediction_slide_rf_test = rf_random_slide.predict(X_test_slide_rf)
plot_predictions(testing_dates, y_test_slide_rf, prediction_slide_rf_test)
evaluate_model(y_test_slide_rf, prediction_slide_rf_test)

#Prevendo o próximo mês
prediction__slide_rf_new = rf_random_slide.predict(predict)
print(prediction__slide_rf_new)
