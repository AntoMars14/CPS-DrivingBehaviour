import pandas as pd
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score, accuracy_score, classification_report

# --- 1. CONFIGURAZIONE ---
current_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.normpath(os.path.join(current_dir, '..', 'dataset', 'train_motion_data.csv'))
test_path = os.path.normpath(os.path.join(current_dir, '..', 'dataset', 'test_motion_data.csv'))

df_train_raw = pd.read_csv(train_path)
df_test_raw = pd.read_csv(test_path)


def create_features(df_raw, window_size):
    if len(df_raw) < window_size:
        return None, None

    sensor_cols = ['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ']

    # Feature Engineering
    rolling_std = df_raw[sensor_cols].rolling(window=window_size).std(ddof=0).add_suffix('_std')
    rolling_mean = df_raw[sensor_cols].rolling(window=window_size).mean().add_suffix('_mean')

    df_windowed = pd.concat([df_raw['Class'], rolling_std, rolling_mean], axis=1).dropna()

    X = df_windowed.drop(columns=['Class'])
    y = df_windowed['Class']
    return X, y


# --- 2. FINE TUNING (BILANCIATO) ---

print(f"Ottimizzazione Bilanciata: Priorità Min Recall (50%) + Accuratezza (50%)")
print(f"Max Depth: 3 (Fissa)")
print("-" * 90)
print(f"{'Win':<4} | {'Min Recall':<10} | {'Accuracy':<10} | {'Class Recalls (0, 1, 2)':<25} | {'Score':<6}")
print("-" * 90)

best_score = -1
best_window = 0
best_model = None
best_params = {}
best_X_cols = []

# Window da 2 a 60
window_range = range(2, 61, 2)

# Parametri griglia
param_grid = {
    'max_depth': [3],
    'criterion': ['gini', 'entropy'],
    'min_samples_leaf': [1, 5]
}

for w_size in window_range:
    X_tr, y_tr = create_features(df_train_raw, w_size)
    X_te, y_te = create_features(df_test_raw, w_size)

    if X_tr is None or X_te is None: continue

    # Grid Search usando 'balanced_accuracy' per favorire l'equilibrio in addestramento
    clf = DecisionTreeClassifier(random_state=42)
    grid = GridSearchCV(clf, param_grid, cv=3, scoring='balanced_accuracy', n_jobs=-1)
    grid.fit(X_tr, y_tr)

    # Valutazione
    model = grid.best_estimator_
    y_pred = model.predict(X_te)

    # Metriche
    recalls = recall_score(y_te, y_pred, average=None)  # Array per ogni classe
    min_rec = min(recalls)
    acc = accuracy_score(y_te, y_pred)

    # --- FORMULA DI SELEZIONE ---
    combined_score = (min_rec * 0.5) + (acc * 0.5)

    # Logica per marcare il migliore
    is_best = combined_score > best_score
    log_marker = "(*)" if is_best else ""

    # Formattazione stringa recalls
    r_str = " ".join([f"{r:.2f}" for r in recalls])

    print(f"{w_size:<4} | {min_rec:.4f}     | {acc:.4f}     | {r_str:<25} | {combined_score:.3f} {log_marker}")

    if is_best:
        best_score = combined_score
        best_window = w_size
        best_model = model
        best_params = grid.best_params_
        best_X_cols = X_tr.columns

print("-" * 90)
print(f"VINCITORE -> Window: {best_window}")
print(f"Parametri: {best_params}")
print(f"Punteggio Combinato: {best_score:.3f}")


# --- 3. GENERAZIONE CODICE PTOLEMY ---
def generate_ptolemy_code(tree, feature_names, class_names, win_size):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != -2 else "undefined!"
        for i in tree_.feature
    ]

    print("\n" + "#" * 70)
    print(f" CODICE PTOLEMY OTTIMIZZATO (Window Size: {win_size}, Depth: 3)")
    print("#" * 70)
    print("def classify_window(input_features):")
    print(f"    # IMPORTANTE: Calcolare STD e MEAN su finestra di {win_size} campioni.")
    print("    # Input atteso (12 valori):")
    print(f"    # {', '.join(feature_names)}")
    print("")

    for idx, name in enumerate(feature_names):
        print(f"    {name} = input_features[{idx}]")
    print("")

    def recurse(node, depth):
        indent = "    " * (depth + 1)
        if tree_.feature[node] != -2:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            # Usiamo .5f per precisione sulle soglie
            print(f"{indent}if {name} <= {threshold:.5f}:")
            recurse(tree_.children_left[node], depth + 1)
            print(f"{indent}else:  # {name} > {threshold:.5f}")
            recurse(tree_.children_right[node], depth + 1)
        else:
            class_label = class_names[np.argmax(tree_.value[node])]
            print(f"{indent}return \"{class_label}\"")

    recurse(0, 0)


# Stampa report finale del vincitore
X_test_final, y_test_final = create_features(df_test_raw, best_window)
y_pred_final = best_model.predict(X_test_final)

print("\nReport Dettagliato Vincitore:")
print(classification_report(y_test_final, y_pred_final))

generate_ptolemy_code(best_model, list(best_X_cols), best_model.classes_, best_window)