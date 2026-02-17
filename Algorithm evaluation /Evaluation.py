import pandas as pd
from scipy.stats import zscore
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from joblib import dump
import sys



######################################################################1##################################################################

# Load the dataset
df = pd.read_excel('Dry_Bean_Dataset.xlsx')  

# Dimensioni del dataset
print(df.shape)

# Display basic information about the dataset
# Da questa tabella posso vedere che unica feature categorica e' il target, quindi non devo gestire conversione dei valori delle features in numerici
print("Dataset Overview:")
print(df.info())

# Display the first few rows of the dataset
print("\nFirst 5 Rows of the Dataset:")
print(df.head())

# Summary statistics for numerical columns
print("\nSummary Statistics:")
print(df.describe())

# Unique values in each column
print("\nUnique Values in Each Column:")
for column in df.columns:
    print(f"{column}: {df[column].nunique()} unique values")


# Identify the target variable
target_column = 'Class'
print(f"\nTarget Variable: {target_column}")

# Display distribution of the target variable
print(f"\nDistribution of {target_column}:")
print(df[target_column].value_counts())

# Identify descriptive attributes (features)
descriptive_attributes = [col for col in df.columns if col != target_column]
print("\nFeatures:")
print(descriptive_attributes)


######################################################################2##################################################################

# Rimuove tutte le righe con valori mancanti
df = df.dropna()

# Viene calcolato lo z-score per ogni colonna e si vede quanto un punto si discosta dalla media.
z_scores = zscore(df.select_dtypes(include=['float64']))
abs_z_scores = np.abs(z_scores)
df_no_outliers = df[(abs_z_scores < 3).all(axis=1)]

# Conversione in numeri delle colonne categoriche (solo target)
categorical_columns = df.select_dtypes(include=['object']).columns
print("\n\nCategorical Columns:", categorical_columns)

# Converte categorie della colonna target in valori numerici

label_encoder = LabelEncoder()
df[target_column] = label_encoder.fit_transform(df[target_column])

# Stampo il numero di esempi per classe nel dataset completo
print("\nNumero di esempi per classe nel set iniziale (prima della divisione e prima del bilanciamento):")
print(df[target_column].value_counts())

######################################################################3##################################################################

# Splitto il dataset in training and test sets (validation non necessario in quanto si sceglie successivamente la cross-validation che crea dinamicamente questo set)
X_train, X_test, y_train, y_test = train_test_split(df[descriptive_attributes], df[target_column], test_size=0.30, random_state=42)

print("\nTrain Set Info:")
print("X_train Shape:", X_train.shape)
print("y_train Shape:", y_train.shape)

print("\nTest Set Info:")
print("X_test Shape:", X_test.shape)
print("y_test Shape:", y_test.shape)

print("\nNumero di esempi per classe nel set di train (dopo la divisione e prima del bilanciamento):")
print(y_train.value_counts())

# Bilancio il training set, portando gli esempi per ogni classe al valore minimo tra le varie classi
min_samples = y_train.value_counts().min()
undersampler = RandomUnderSampler(sampling_strategy={label: min_samples for label in y_train.unique()}, random_state=42)

X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)

print("\nNumero di esempi per classe nel set di train (dopo la divisione e dopo il bilanciamento):")
print(y_train_resampled.value_counts())

######################################################################4##################################################################

# Decision Tree 
modello_albero = DecisionTreeClassifier(random_state=42)

parametri_albero = {'max_depth': [None, 5, 10, 15],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]}

# Early stopping for Decision Tree
num_fold_dt = 100
skf_dt = StratifiedKFold(n_splits=num_fold_dt, shuffle=True, random_state=42)

max_no_improvement_dt = 2
best_accuracy_dt = 0
no_improvement_count_dt = 0

grid_search_albero = GridSearchCV(modello_albero, parametri_albero, cv=skf_dt, scoring='accuracy')
grid_search_albero.fit(X_train_resampled, y_train_resampled)

modello_albero_migliore = grid_search_albero.best_estimator_


y_train_pred_dt_all = []
y_train_true_dt_all = []
y_val_pred_dt_all = []
y_val_true_dt_all = []

for fold, (train_index_dt, val_index_dt) in enumerate(skf_dt.split(X_train_resampled, y_train_resampled), 1):
    X_train_fold_dt, X_val_fold_dt = X_train_resampled.iloc[train_index_dt], X_train_resampled.iloc[val_index_dt]
    y_train_fold_dt, y_val_fold_dt = y_train_resampled.iloc[train_index_dt], y_train_resampled.iloc[val_index_dt]

    modello_albero_migliore.fit(X_train_fold_dt, y_train_fold_dt)

    # Valutazione sul set di addestramento
    y_train_pred_dt = modello_albero_migliore.predict(X_train_fold_dt)
    y_train_true_dt_all.extend(y_train_fold_dt)
    y_train_pred_dt_all.extend(y_train_pred_dt)

    # Valutazione sul set di validazione
    y_val_pred_dt = modello_albero_migliore.predict(X_val_fold_dt)
    y_val_true_dt_all.extend(y_val_fold_dt)
    y_val_pred_dt_all.extend(y_val_pred_dt)

    accuratezza_fold_dt = accuracy_score(y_val_fold_dt, y_val_pred_dt)

    if accuratezza_fold_dt > best_accuracy_dt:
        best_accuracy_dt = accuratezza_fold_dt
        no_improvement_count_dt = 0
    else:
        no_improvement_count_dt += 1

    print(f"Fold {fold}: Accuratezza Decision Tree = {accuratezza_fold_dt}")

    if no_improvement_count_dt >= max_no_improvement_dt:
        print(f"\nEarly stopping Decision Tree! Miglior accuratezza raggiunta: {best_accuracy_dt}\n")
        break

# Calcolo e stampa delle metriche per il set di addestramento
conf_matrix_train_dt = confusion_matrix(y_train_true_dt_all, y_train_pred_dt_all)
precision_train_dt = precision_score(y_train_true_dt_all, y_train_pred_dt_all, average='weighted')
recall_train_dt = recall_score(y_train_true_dt_all, y_train_pred_dt_all, average='weighted')
f1_score_train_dt = f1_score(y_train_true_dt_all, y_train_pred_dt_all, average='weighted')

print("\nMatrice di Confusione Decision Tree (Set di Addestramento):")
print(conf_matrix_train_dt)
print("\nPrecision Decision Tree (Set di Addestramento):", precision_train_dt)
print("Recall Decision Tree (Set di Addestramento):", recall_train_dt)
print("F1 Score Decision Tree (Set di Addestramento):", f1_score_train_dt)

# Calcolo e stampa delle metriche per il set di validazione
conf_matrix_val_dt = confusion_matrix(y_val_true_dt_all, y_val_pred_dt_all)
precision_val_dt = precision_score(y_val_true_dt_all, y_val_pred_dt_all, average='weighted')
recall_val_dt = recall_score(y_val_true_dt_all, y_val_pred_dt_all, average='weighted')
f1_score_val_dt = f1_score(y_val_true_dt_all, y_val_pred_dt_all, average='weighted')

print("\nMatrice di Confusione Decision Tree (Set di Validazione):")
print(conf_matrix_val_dt)
print("\nPrecision Decision Tree (Set di Validazione):", precision_val_dt)
print("Recall Decision Tree (Set di Validazione):", recall_val_dt)
print("F1 Score Decision Tree (Set di Validazione):", f1_score_val_dt)

# KNN model with feature normalization, Grid Search, Cross-Validation, Early Stopping
scaler = StandardScaler()
X_train_resampled_scaled = scaler.fit_transform(X_train_resampled)
#X_val_scaled_knn = scaler.transform(X_train)

modello_knn = KNeighborsClassifier()

parametri_knn = {'n_neighbors': [3, 5, 7],
                 'weights': ['uniform', 'distance'],
                 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}

# Early stopping for KNN
num_fold_knn = 100
skf_knn = StratifiedKFold(n_splits=num_fold_knn, shuffle=True, random_state=42)

max_no_improvement_knn = 2
best_accuracy_knn = 0
no_improvement_count_knn = 0

grid_search_knn = GridSearchCV(modello_knn, parametri_knn, cv=skf_knn, scoring='accuracy')
grid_search_knn.fit(X_train_resampled_scaled, y_train_resampled)

modello_knn_migliore = grid_search_knn.best_estimator_

# Valori per la valutazione
y_train_pred_knn_all = []
y_train_true_knn_all = []
y_val_pred_knn_all = []
y_val_true_knn_all = []

for fold, (train_index_knn, val_index_knn) in enumerate(skf_knn.split(X_train_resampled_scaled, y_train_resampled), 1):
    X_train_fold_knn, X_val_fold_knn = X_train_resampled_scaled[train_index_knn], X_train_resampled_scaled[val_index_knn]
    y_train_fold_knn, y_val_fold_knn = y_train_resampled.iloc[train_index_knn], y_train_resampled.iloc[val_index_knn]

    modello_knn_migliore.fit(X_train_fold_knn, y_train_fold_knn)

    # Valutazione sul set di addestramento
    y_train_pred_knn = modello_knn_migliore.predict(X_train_fold_knn)
    y_train_true_knn_all.extend(y_train_fold_knn)
    y_train_pred_knn_all.extend(y_train_pred_knn)

    # Valutazione sul set di validazione
    y_val_pred_knn = modello_knn_migliore.predict(X_val_fold_knn)
    y_val_true_knn_all.extend(y_val_fold_knn)
    y_val_pred_knn_all.extend(y_val_pred_knn)

    accuratezza_fold_knn = accuracy_score(y_val_fold_knn, y_val_pred_knn)

    if accuratezza_fold_knn > best_accuracy_knn:
        best_accuracy_knn = accuratezza_fold_knn
        no_improvement_count_knn = 0
    else:
        no_improvement_count_knn += 1

    print(f"Fold {fold}: Accuratezza KNN = {accuratezza_fold_knn}")

    if no_improvement_count_knn >= max_no_improvement_knn:
        print(f"\nEarly stopping KNN! Miglior accuratezza raggiunta: {best_accuracy_knn}\n")
        break

# Calcolo e stampa delle metriche per il set di addestramento
conf_matrix_train_knn = confusion_matrix(y_train_true_knn_all, y_train_pred_knn_all)
precision_train_knn = precision_score(y_train_true_knn_all, y_train_pred_knn_all, average='weighted')
recall_train_knn = recall_score(y_train_true_knn_all, y_train_pred_knn_all, average='weighted')
f1_score_train_knn = f1_score(y_train_true_knn_all, y_train_pred_knn_all, average='weighted')

print("\nMatrice di Confusione KNN (Set di Addestramento):")
print(conf_matrix_train_knn)
print("\nPrecision KNN (Set di Addestramento):", precision_train_knn)
print("Recall KNN (Set di Addestramento):", recall_train_knn)
print("F1 Score KNN (Set di Addestramento):", f1_score_train_knn)

# Calcolo e stampa delle metriche per il set di validazione
conf_matrix_val_knn = confusion_matrix(y_val_true_knn_all, y_val_pred_knn_all)
precision_val_knn = precision_score(y_val_true_knn_all, y_val_pred_knn_all, average='weighted')
recall_val_knn = recall_score(y_val_true_knn_all, y_val_pred_knn_all, average='weighted')
f1_score_val_knn = f1_score(y_val_true_knn_all, y_val_pred_knn_all, average='weighted')

print("\nMatrice di Confusione KNN (Set di Validazione):")
print(conf_matrix_val_knn)
print("\nPrecision KNN (Set di Validazione):", precision_val_knn)
print("Recall KNN (Set di Validazione):", recall_val_knn)
print("F1 Score KNN (Set di Validazione):", f1_score_val_knn)

# SVM model with feature normalization, Grid Search, Cross-Validation, Early Stopping
modello_svm = SVC(kernel='linear', random_state=42)

parametri_svm = {'C': [0.1, 1, 10],
                 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                 'degree': [2, 3, 4]}

# Early stopping for SVM
num_fold_svm = 100
skf_svm = StratifiedKFold(n_splits=num_fold_svm, shuffle=True, random_state=42)

max_no_improvement_svm = 2
best_accuracy_svm = 0
no_improvement_count_svm = 0

grid_search_svm = GridSearchCV(modello_svm, parametri_svm, cv=skf_svm, scoring='accuracy')
grid_search_svm.fit(X_train_resampled_scaled, y_train_resampled)

modello_svm_migliore = grid_search_svm.best_estimator_

# Valori per la valutazione
y_train_pred_svm_all = []
y_train_true_svm_all = []
y_val_pred_svm_all = []
y_val_true_svm_all = []

for fold, (train_index_svm, val_index_svm) in enumerate(skf_svm.split(X_train_resampled_scaled, y_train_resampled), 1):
    X_train_fold_svm, X_val_fold_svm = X_train_resampled_scaled[train_index_svm], X_train_resampled_scaled[val_index_svm]
    y_train_fold_svm, y_val_fold_svm = y_train_resampled.iloc[train_index_svm], y_train_resampled.iloc[val_index_svm]

    modello_svm_migliore.fit(X_train_fold_svm, y_train_fold_svm)

    # Valutazione sul set di addestramento
    y_train_pred_svm = modello_svm_migliore.predict(X_train_fold_svm)
    y_train_true_svm_all.extend(y_train_fold_svm)
    y_train_pred_svm_all.extend(y_train_pred_svm)

    # Valutazione sul set di validazione
    y_val_pred_svm = modello_svm_migliore.predict(X_val_fold_svm)
    y_val_true_svm_all.extend(y_val_fold_svm)
    y_val_pred_svm_all.extend(y_val_pred_svm)

    accuratezza_fold_svm = accuracy_score(y_val_fold_svm, y_val_pred_svm)

    if accuratezza_fold_svm > best_accuracy_svm:
        best_accuracy_svm = accuratezza_fold_svm
        no_improvement_count_svm = 0
    else:
        no_improvement_count_svm += 1

    print(f"Fold {fold}: Accuratezza SVM = {accuratezza_fold_svm}")

    if no_improvement_count_svm >= max_no_improvement_svm:
        print(f"\nEarly stopping SVM! Miglior accuratezza raggiunta: {best_accuracy_svm}\n")
        break

# Calcolo e stampa delle metriche per il set di addestramento
conf_matrix_train_svm = confusion_matrix(y_train_true_svm_all, y_train_pred_svm_all)
precision_train_svm = precision_score(y_train_true_svm_all, y_train_pred_svm_all, average='weighted')
recall_train_svm = recall_score(y_train_true_svm_all, y_train_pred_svm_all, average='weighted')
f1_score_train_svm = f1_score(y_train_true_svm_all, y_train_pred_svm_all, average='weighted')

print("\nMatrice di Confusione SVM (Set di Addestramento):")
print(conf_matrix_train_svm)
print("\nPrecision SVM (Set di Addestramento):", precision_train_svm)
print("Recall SVM (Set di Addestramento):", recall_train_svm)
print("F1 Score SVM (Set di Addestramento):", f1_score_train_svm)

# Calcolo e stampa delle metriche per il set di validazione
conf_matrix_val_svm = confusion_matrix(y_val_true_svm_all, y_val_pred_svm_all)
precision_val_svm = precision_score(y_val_true_svm_all, y_val_pred_svm_all, average='weighted')
recall_val_svm = recall_score(y_val_true_svm_all, y_val_pred_svm_all, average='weighted')
f1_score_val_svm = f1_score(y_val_true_svm_all, y_val_pred_svm_all, average='weighted')

print("\nMatrice di Confusione SVM (Set di Validazione):")
print(conf_matrix_val_svm)
print("\nPrecision SVM (Set di Validazione):", precision_val_svm)
print("Recall SVM (Set di Validazione):", recall_val_svm)
print("F1 Score SVM (Set di Validazione):", f1_score_val_svm)


# Random Forest with early stopping, Grid Search, Cross-Validation
modello_random_forest = RandomForestClassifier(n_estimators=100, random_state=42)

parametri_random_forest = {'n_estimators': [50, 100, 200],
                           'max_depth': [None, 10, 20],
                           'min_samples_split': [2, 5, 10],
                           'min_samples_leaf': [1, 2, 4]}

# Early stopping for Random Forest
num_fold_rf = 5
skf_rf = StratifiedKFold(n_splits=num_fold_rf, shuffle=True, random_state=42)

max_no_improvement_rf = 2
best_accuracy_rf = 0
no_improvement_count_rf = 0

grid_search_random_forest = GridSearchCV(modello_random_forest, parametri_random_forest, cv=skf_rf, scoring='accuracy')
grid_search_random_forest.fit(X_train_resampled, y_train_resampled)

modello_random_forest_migliore = grid_search_random_forest.best_estimator_

# Valori per la valutazione
y_train_pred_rf_all = []
y_train_true_rf_all = []
y_val_pred_rf_all = []
y_val_true_rf_all = []

for fold_rf, (train_index_rf, val_index_rf) in enumerate(skf_rf.split(X_train_resampled, y_train_resampled), 1):
    X_train_fold_rf, X_val_fold_rf = X_train_resampled.iloc[train_index_rf], X_train_resampled.iloc[val_index_rf]
    y_train_fold_rf, y_val_fold_rf = y_train_resampled.iloc[train_index_rf], y_train_resampled.iloc[val_index_rf]

    modello_random_forest_migliore.fit(X_train_fold_rf, y_train_fold_rf)

    # Valutazione sul set di addestramento
    y_train_pred_rf = modello_random_forest_migliore.predict(X_train_fold_rf)
    y_train_true_rf_all.extend(y_train_fold_rf)
    y_train_pred_rf_all.extend(y_train_pred_rf)

    # Valutazione sul set di validazione
    y_val_pred_rf = modello_random_forest_migliore.predict(X_val_fold_rf)
    y_val_true_rf_all.extend(y_val_fold_rf)
    y_val_pred_rf_all.extend(y_val_pred_rf)

    accuratezza_fold_rf = accuracy_score(y_val_fold_rf, y_val_pred_rf)

    if accuratezza_fold_rf > best_accuracy_rf:
        best_accuracy_rf = accuratezza_fold_rf
        no_improvement_count_rf = 0
    else:
        no_improvement_count_rf += 1

    print(f"Fold {fold_rf}: Accuratezza Random Forest = {accuratezza_fold_rf}")

    if no_improvement_count_rf >= max_no_improvement_rf:
        print(f"\nEarly stopping Random Forest! Miglior accuratezza raggiunta: {best_accuracy_rf}\n")
        break

# Calcolo e stampa delle metriche per il set di addestramento
conf_matrix_train_rf = confusion_matrix(y_train_true_rf_all, y_train_pred_rf_all)
precision_train_rf = precision_score(y_train_true_rf_all, y_train_pred_rf_all, average='weighted')
recall_train_rf = recall_score(y_train_true_rf_all, y_train_pred_rf_all, average='weighted')
f1_score_train_rf = f1_score(y_train_true_rf_all, y_train_pred_rf_all, average='weighted')

print("\nMatrice di Confusione Random Forest (Set di Addestramento):")
print(conf_matrix_train_rf)
print("\nPrecision Random Forest (Set di Addestramento):", precision_train_rf)
print("Recall Random Forest (Set di Addestramento):", recall_train_rf)
print("F1 Score Random Forest (Set di Addestramento):", f1_score_train_rf)

# Calcolo e stampa delle metriche per il set di validazione
conf_matrix_val_rf = confusion_matrix(y_val_true_rf_all, y_val_pred_rf_all)
precision_val_rf = precision_score(y_val_true_rf_all, y_val_pred_rf_all, average='weighted')
recall_val_rf = recall_score(y_val_true_rf_all, y_val_pred_rf_all, average='weighted')
f1_score_val_rf = f1_score(y_val_true_rf_all, y_val_pred_rf_all, average='weighted')

print("\nMatrice di Confusione Random Forest (Set di Validazione):")
print(conf_matrix_val_rf)
print("\nPrecision Random Forest (Set di Validazione):", precision_val_rf)
print("Recall Random Forest (Set di Validazione):", recall_val_rf)
print("F1 Score Random Forest (Set di Validazione):", f1_score_val_rf)

###############################################################5##########################################################################
# Valutazione sul set di test
y_test_pred_dt = modello_albero_migliore.predict(X_test)


# Calcolo e stampa delle metriche per il set di test
conf_matrix_test_dt = confusion_matrix(y_test, y_test_pred_dt)
precision_test_dt = precision_score(y_test, y_test_pred_dt, average='weighted')
recall_test_dt = recall_score(y_test, y_test_pred_dt, average='weighted')
f1_score_test_dt = f1_score(y_test, y_test_pred_dt, average='weighted')
accuracy_test_dt = accuracy_score(y_test, y_test_pred_dt)

print("\nMatrice di Confusione Decision Tree (Set di Test):")
print(conf_matrix_test_dt)
print("\nPrecision Decision Tree (Set di Test):", precision_test_dt)
print("Recall Decision Tree (Set di Test):", recall_test_dt)
print("F1 Score Decision Tree (Set di Test):", f1_score_test_dt)
print("Accuracy Decision Tree (Set di Test):", accuracy_test_dt)

# Valutazione sul set di test
X_test_scaled_knn = scaler.transform(X_test)
y_test_pred_knn = modello_knn_migliore.predict(X_test_scaled_knn)

# Calcolo e stampa delle metriche per il set di test
conf_matrix_test_knn = confusion_matrix(y_test, y_test_pred_knn)
precision_test_knn = precision_score(y_test, y_test_pred_knn, average='weighted')
recall_test_knn = recall_score(y_test, y_test_pred_knn, average='weighted')
f1_score_test_knn = f1_score(y_test, y_test_pred_knn, average='weighted')
accuracy_test_knn = accuracy_score(y_test, y_test_pred_knn)


print("\nMatrice di Confusione KNN (Set di Test):")
print(conf_matrix_test_knn)
print("\nPrecision KNN (Set di Test):", precision_test_knn)
print("Recall KNN (Set di Test):", recall_test_knn)
print("F1 Score KNN (Set di Test):", f1_score_test_knn)
print("Accuracy Decision Tree (Set di Test):", accuracy_test_knn)

# Valutazione sul set di test
y_test_pred_svm = modello_svm_migliore.predict(X_test_scaled_knn)

# Calcolo e stampa delle metriche per il set di test
conf_matrix_test_svm = confusion_matrix(y_test, y_test_pred_svm)
precision_test_svm = precision_score(y_test, y_test_pred_svm, average='weighted')
recall_test_svm = recall_score(y_test, y_test_pred_svm, average='weighted')
f1_score_test_svm = f1_score(y_test, y_test_pred_svm, average='weighted')
accuracy_test_svm = accuracy_score(y_test, y_test_pred_svm)

print("\nMatrice di Confusione SVM (Set di Test):")
print(conf_matrix_test_svm)
print("\nPrecision SVM (Set di Test):", precision_test_svm)
print("Recall SVM (Set di Test):", recall_test_svm)
print("F1 Score SVM (Set di Test):", f1_score_test_svm)
print("Accuracy Decision Tree (Set di Test):", accuracy_test_svm)


# Valutazione sul set di test
y_test_pred_rf = modello_random_forest_migliore.predict(X_test)

# Calcolo e stampa delle metriche per il set di test
conf_matrix_test_rf = confusion_matrix(y_test, y_test_pred_rf)
precision_test_rf = precision_score(y_test, y_test_pred_rf, average='weighted')
recall_test_rf = recall_score(y_test, y_test_pred_rf, average='weighted')
f1_score_test_rf = f1_score(y_test, y_test_pred_rf, average='weighted')
accuracy_test_rf = accuracy_score(y_test, y_test_pred_rf)

print("\nMatrice di Confusione Random Forest (Set di Test):")
print(conf_matrix_test_rf)
print("\nPrecision Random Forest (Set di Test):", precision_test_rf)
print("Recall Random Forest (Set di Test):", recall_test_rf)
print("F1 Score Random Forest (Set di Test):", f1_score_test_rf)
print("Accuracy Decision Tree (Set di Test):", accuracy_test_rf)


