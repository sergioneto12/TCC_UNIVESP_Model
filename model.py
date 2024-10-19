import pathlib
import pandas           as pd
import numpy            as np
import matplotlib.pylab as plt
import plotly.express   as px
import seaborn          as sns
from collections import Counter

# Pré-processamento e divisão para treino e testes
from sklearn                 import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss

# Algoritmos para seleção do melhor modelo de ML
from sklearn.ensemble        import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Métricas para validação de cada modelo
from sklearn.metrics         import confusion_matrix, accuracy_score, mean_absolute_error, roc_curve, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from pycaret.classification import *

# ######################### Treinamento do modelo #########################

path = pathlib.Path().cwd() / 'files/bs140513_032310.csv'
fraud_df = pd.read_csv(path)

# Definindo novos intervalos (bins) para rebalancear a distribuição
bins = [0, 20, 40, 60, 80, 100, 200, 400, 600, 800, 1000, 2000, 4000, 6000, 8000, 8400]
labels = ['0-20', '20-40', '40-60', '60-80', '80-100', '100-200', '200-400', '400-600', '600-800', '800-1000', '1000-2000', '2000-4000', '4000-6000', '6000-8000', '8000-8400']

# Criando uma nova coluna categorizada
fraud_df['amount_category'] = pd.cut(fraud_df['amount'], bins=bins, labels=labels, include_lowest=True)

# Aqui estamos convertendo textos para registros numéricos, permitindo que o algoritmo de machine learning leia as colunas
label_encoder = preprocessing.LabelEncoder()
one_hot_encoder_gender = preprocessing.OneHotEncoder(sparse_output=False)
one_hot_encoder_category = preprocessing.OneHotEncoder(sparse_output=False)

columns_1 = ['customer', 'merchant', 'amount_category', 'zipcodeOri', 'zipMerchant']
columns_2 = ['gender', 'category']

# Pré-processamento final dos dados
fraud_ml = fraud_df.copy()
fraud_ml.replace("'", "", regex=True, inplace=True)
fraud_ml['age'].replace("U", "7", inplace=True)
fraud_ml['age'] = fraud_ml['age'].astype(int)

for i in columns_1:
    fraud_ml[i] = label_encoder.fit_transform(fraud_ml[i])

# Ajustando e transformando o OneHotEncoder no mesmo conjunto de dados
fraud_ml_gender = one_hot_encoder_gender.fit_transform(fraud_ml[['gender']])
fraud_ml_category = one_hot_encoder_category.fit_transform(fraud_ml[['category']])

# Obtendo os nomes das colunas corretamente
df_gender_encoded = pd.DataFrame(fraud_ml_gender, columns=one_hot_encoder_gender.get_feature_names_out(['gender']))
df_category_encoded = pd.DataFrame(fraud_ml_category, columns=one_hot_encoder_category.get_feature_names_out(['category']))

# Concatenando os resultados
fraud_ml = pd.concat([fraud_ml, df_gender_encoded, df_category_encoded], axis=1)

fraud_ml.drop(['step', 'gender', 'category'], axis=1, inplace=True)

# fraud_ml_1 = fraud_ml.loc[fraud_ml['fraud'] == 0][:75001]
# fraud_ml_2 = fraud_ml.loc[fraud_ml['fraud'] == 1]

# fraud_ml = pd.concat([fraud_ml_1, fraud_ml_2], axis=0)

x = fraud_ml.drop('fraud', axis=1)
y = fraud_ml['fraud']

# Mostra a distribuição original das classes
print(f'Distribuição original das classes: {Counter(y)}')

# Cria o objeto RandomUnderSampler
rus = NearMiss()
X_resampled, y_resampled = rus.fit_resample(x, y)

# Este trecho comentado pode ser utilizado para avaliar nosso modelo com oversampling. Por hora, o melhor nos testes é o undersampling, já habilitado aqui
# smote = SMOTE(random_state=42)
# X_resampled, y_resampled = smote.fit_resample(x, y)

# Mostra a nova distribuição das classes
print(f'Nova distribuição das classes: {Counter(y_resampled)}')

# Converte de volta para DataFrame se necessário
df_resampled = pd.DataFrame(X_resampled, columns=x.columns)
df_resampled['fraud'] = y_resampled

# df_resampled.head()
fraud_ml = df_resampled

x = fraud_ml.drop('fraud', axis=1)
y = fraud_ml['fraud']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.3, random_state = 42, stratify=y, shuffle=True)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Random Forest Classifier
rf = RandomForestClassifier(bootstrap=True, max_depth=None, min_samples_leaf=1, min_samples_split=10, n_estimators=200, class_weight='balanced', random_state=42) 
model = rf.fit(x_train, y_train)

y_pred = model.predict(x_test)
# accuracy_score(y_test, y_pred)

# Avaliar o modelo
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print(f"Acurácia: {accuracy}")
print(f"Precisão: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")
print(f"AUC-ROC: {roc_auc}")

# Testando agora com logistic regression
lr = LogisticRegression(C=10, penalty='l1', solver='liblinear')
model_2 = lr.fit(x_train, y_train)

y_pred = model_2.predict(x_test)
# accuracy_score(y_test, y_pred)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print(f"Acurácia: {accuracy}")
print(f"Precisão: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")
print(f"AUC-ROC: {roc_auc}")

# # ######################### Teste com novo dataset #########################

# # Função para gerar valores de amount com maior concentração entre 0 e 100
# def generate_amount():
#     if np.random.rand() < 0.8:  # 80% de chance de estar entre 0 e 100
#         return np.round(np.random.uniform(0.1, 100), 2)
#     else:  # 20% de chance de estar entre 100 e 8400
#         return np.round(np.random.uniform(100, 8400), 2)
    
# label_encoder = preprocessing.LabelEncoder()
# one_hot_encoder_gender = preprocessing.OneHotEncoder(sparse_output=False)
# one_hot_encoder_category = preprocessing.OneHotEncoder(sparse_output=False)

# novo_exemplo = pd.DataFrame({
#     'step': np.random.randint(1, 200, 500),
#     'customer': [f'C{np.random.randint(100000000, 999999999)}' for _ in range(500)],
#     'age': np.random.choice([0, 1, 2, 3, 4, 5, 6, 7], 500),
#     'gender': np.random.choice(['M', 'F', 'E', 'U'], 500),
#     'merchant': [f'M{np.random.randint(100000000, 999999999)}' for _ in range(500)],
#     'category': np.random.choice([
#         "es_transportation", "es_health", "es_otherservices", "es_food",
#         "es_hotelservices", "es_barsandrestaurants", "es_tech",
#         "es_sportsandtoys", "es_wellnessandbeauty", "es_hyper", "es_fashion",
#         "es_home", "es_contents", "es_travel", "es_leisure"
#     ], 500),
#     'amount': [generate_amount() for _ in range(500)],
#     'fraud': np.random.choice([0, 1], 500, p=[0.95, 0.05]),
#     'zipcodeOri': '28007',
#     'zipMerchant': '28007',
# })

# label_columns = ['customer', 'merchant', 'amount_category', 'zipcodeOri', 'zipMerchant']

# columns_order = ['step', 'customer', 'age', 'gender', 'zipcodeOri', 'merchant',
#        'zipMerchant', 'category', 'amount', 'fraud']

# novo_exemplo = novo_exemplo[columns_order]
# # Definindo os intervalos (bins) e os rótulos (labels)
# bins = [0, 20, 40, 60, 80, 100, 200, 400, 600, 800, 1000, 2000, 4000, 6000, 8000, 8400]
# labels = ['0-20', '20-40', '40-60', '60-80', '80-100', '100-200', '200-400', '400-600', '600-800', '800-1000', '1000-2000', '2000-4000', '4000-6000', '6000-8000', '8000-8400']

# # Criando a coluna categorizada
# novo_exemplo['amount_category'] = pd.cut(novo_exemplo['amount'], bins=bins, labels=labels, include_lowest=True)

# # Função para lidar com novos valores no LabelEncoder
# def handle_new_labels(encoder, value):
#     if value in encoder.classes_:
#         return encoder.transform([value])[0]
#     else:
#         # Adiciona a nova classe temporariamente e transforma o valor
#         new_classes = np.append(encoder.classes_, value)
#         encoder.classes_ = new_classes
#         return encoder.transform([value])[0]

# # Aplique o mesmo pré-processamento no novo exemplo
# novo_exemplo['age'].replace("U", "7", inplace=True)
# novo_exemplo['age'] = novo_exemplo['age'].astype(int)

# # Ajuste inicial para garantir que o LabelEncoder tenha classes
# for col in label_columns:
#     label_encoder.fit(novo_exemplo[col])

# # Aplicar LabelEncoder nas colunas categóricas com verificação de novos valores
# for i in label_columns:
#     novo_exemplo[i] = novo_exemplo[i].apply(lambda x: handle_new_labels(label_encoder, x))

# # Aplicar OneHotEncoder nas colunas 'gender' e 'category'
# novo_exemplo_gender = one_hot_encoder_gender.fit_transform(novo_exemplo[['gender']])
# novo_exemplo_category = one_hot_encoder_category.fit_transform(novo_exemplo[['category']])

# # Convertendo os resultados para DataFrame
# df_novo_exemplo_gender = pd.DataFrame(novo_exemplo_gender, columns=one_hot_encoder_gender.get_feature_names_out(['gender']))
# df_novo_exemplo_category = pd.DataFrame(novo_exemplo_category, columns=one_hot_encoder_category.get_feature_names_out(['category']))

# # Concatenando as novas colunas com o novo exemplo
# novo_exemplo = pd.concat([novo_exemplo, df_novo_exemplo_gender, df_novo_exemplo_category], axis=1)

# y = novo_exemplo['fraud']

# # Removendo colunas desnecessárias
# novo_exemplo.drop(['gender', 'category', 'fraud', 'step'], axis=1, inplace=True)

# # Aplicando o mesmo escalonamento de valores
# scaler = MinMaxScaler()
# novo_exemplo_scaled = scaler.fit_transform(novo_exemplo)

# # Realizando a previsão
# y_pred_novo_exemplo = model.predict(novo_exemplo_scaled) # Random Forest
# # y_pred_novo_exemplo = model_auto_ml.predict(novo_exemplo_scaled) # XGBoost

# # print(f'Previsão para o novo exemplo: {y_pred_novo_exemplo}')
# print(accuracy_score(y, y_pred_novo_exemplo))
