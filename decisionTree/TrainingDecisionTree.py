import pandas as pd
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# --- 1. CONFIGURAZIONE ---
current_dir = os.path.dirname(os.path.abspath(__file__))
# Percorso relativo per trovare il dataset
train_path = os.path.normpath(os.path.join(current_dir, '..', 'dataset', 'train_motion_data.csv'))
test_path = os.path.normpath(os.path.join(current_dir, '..', 'dataset', 'test_motion_data.csv'))


def load_and_window_data(path, window_size=60):
    if not os.path.exists(path):
        print(f"ERRORE: File non trovato in {path}")
        return None, None

    df = pd.read_csv(path)

    # --- FEATURE ENGINEERING: SLIDING WINDOW ---
    # Calcoliamo statistiche su una finestra mobile di 'window_size' campioni
    sensor_cols = ['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ']

    # Calcolo STD (Variazione) e MEDIA (Orientamento)
    # .add_suffix aggiunge _std o _mean ai nomi delle colonne
    rolling_std = df[sensor_cols].rolling(window=window_size).std().add_suffix('_std')
    rolling_mean = df[sensor_cols].rolling(window=window_size).mean().add_suffix('_mean')

    # Uniamo le nuove feature
    # Nota: Le prime 19 righe saranno NaN e verranno scartate
    df_windowed = pd.concat([df['Class'], rolling_std, rolling_mean], axis=1).dropna()

    X = df_windowed.drop(columns=['Class'])
    y = df_windowed['Class']

    return X, y


print(f"Caricamento dataset da: {train_path}...")
X_train, y_train = load_and_window_data(train_path)
X_test, y_test = load_and_window_data(test_path)

if X_train is None:
    exit()

# --- 2. ADDESTRAMENTO ---
# Usiamo max_depth=3 per avere regole abbastanza potenti ma scrivibili in if-else
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# --- 3. VALUTAZIONE ---
if X_test is not None:
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuratezza Modello (Window Size=60): {acc:.4f} ({acc * 100:.2f}%)")
    print("\nReport Classificazione:")
    print(classification_report(y_test, y_pred))


# --- 4. GENERAZIONE CODICE PER PTOLEMY ---
def generate_ptolemy_code(tree, feature_names, class_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != -2 else "undefined!"
        for i in tree_.feature
    ]

    print("\n" + "=" * 50)
    print(" CODICE DA COPIARE IN PTOLEMY (PythonScript Actor) oppure usare le condizioni direttamente.")
    print("=" * 50)
    print("def classify_window(input_features):")
    print("    # Input atteso: array con 12 valori nell'ordine seguente:")
    print(f"    # {', '.join(feature_names)}")
    print("    # (Calcolare std e mean sugli ultimi 60 campioni prima di chiamare questa funzione)")
    print("")
    # Genera mapping variabili per leggibilità
    for idx, name in enumerate(feature_names):
        print(f"    {name} = input_features[{idx}]")
    print("")

    def recurse(node, depth):
        indent = "    " * (depth + 1)
        if tree_.feature[node] != -2:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print(f"{indent}if {name} <= {threshold:.4f}:")
            recurse(tree_.children_left[node], depth + 1)
            print(f"{indent}else:  # {name} > {threshold:.4f}")
            recurse(tree_.children_right[node], depth + 1)
        else:
            class_label = class_names[np.argmax(tree_.value[node])]
            print(f"{indent}return \"{class_label}\"")

    recurse(0, 0)


generate_ptolemy_code(clf, list(X_train.columns), clf.classes_)