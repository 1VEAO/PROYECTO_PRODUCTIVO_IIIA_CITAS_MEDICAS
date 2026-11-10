import os
import sys
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix
)

# Garantizar que Python reconozca los módulos internos de src/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import supabase

FEATURES = [
    'lead_time', 'age', 'scholarship', 'hipertension',
    'diabetes', 'alcoholism', 'handcap', 'sms_received', 'historical_noshows'
]
META_RECALL = 0.85  # Objetivo SMART del CPP1 (punto 3 y 10)

def registrar_log(proceso, capa, estado, registros=0, error=None):
    """Registra trazabilidad de auditoría en meta.log_procesos"""
    try:
        log_data = {
            "proceso_nombre": proceso,
            "capa_destino": capa,
            "registros_procesados": registros,
            "estado": estado,
            "mensaje_error": str(error) if error else None,
            "usuario_ejecutor": "PYTHON_SCRIPT"
        }
        supabase.schema("meta").table("log_procesos").insert(log_data).execute()
    except Exception as e:
        print(f" No se pudo registrar log en meta: {e}")

def extraer_silver_paginado(tamano_pagina=1000):
    """Extrae registros de silver.citas_cleaned con paginación determinista"""
    print(" Leyendo datos limpios desde silver.citas_cleaned...")
    registros = []
    inicio = 0
    while True:
        respuesta = supabase.schema("silver").table("citas_cleaned") \
            .select("*") \
            .order("appointment_id", desc=False) \
            .range(inicio, inicio + tamano_pagina - 1) \
            .execute()
        datos = respuesta.data
        if not datos:
            break
        registros.extend(datos)
        inicio += tamano_pagina
        if len(datos) < tamano_pagina:
            break
    print(f" Total leídos de Silver: {len(registros)} registros.")
    return pd.DataFrame(registros)

def buscar_umbral(y_real, probabilidades, meta_recall=META_RECALL):
    """
    Threshold Tuning: recorre umbrales de decisión y selecciona, entre los que
    cumplen Recall >= meta_recall, el de mejor F1-Score (mejor equilibrio).
    """
    print(f"\n⏱️  THRESHOLD TUNING (meta: Recall >= {meta_recall:.2f})...")
    candidatos = []
    for umbral in np.arange(0.05, 0.96, 0.01):
        y_pred = (probabilidades >= umbral).astype(int)
        rec = recall_score(y_real, y_pred, pos_label=1)
        if rec >= meta_recall:
            candidatos.append({
                "umbral": float(umbral),
                "recall": rec,
                "precision": precision_score(y_real, y_pred, pos_label=1, zero_division=0),
                "f1": f1_score(y_real, y_pred, pos_label=1, zero_division=0)
            })

    if not candidatos:
        raise RuntimeError(f" Ningún umbral alcanzó Recall >= {meta_recall:.2f}.")

    # Entre los que cumplen la meta, elegir el de mayor F1 (menos falsos positivos)
    mejor = max(candidatos, key=lambda c: c['f1'])
    print(f"   ➜ {len(candidatos)} umbrales cumplen la meta | Umbral elegido: {mejor['umbral']:.2f}")
    print(f"   ➜ Recall: {mejor['recall']:.4f} | Precision: {mejor['precision']:.4f} | F1: {mejor['f1']:.4f}")
    return mejor

def tunear_modelo_campeon():
    """
    Punto 12 del CPP1 (Fase 5 - Evaluación):
      1. Calibración de hiperparámetros del Random Forest con RandomizedSearchCV
         (n_estimators, max_depth, min_samples_split, min_samples_leaf).
      2. Threshold Tuning para asegurar Recall >= 0.85.
      3. Guardar el artefacto del campeón con el umbral de decisión incluido.
    """
    proceso_nombre = "TUNING_RF_THRESHOLD_GOLD"
    registrar_log(proceso_nombre, "GOLD", "INICIADO")

    try:
        df = extraer_silver_paginado()
        if df.empty:
            print(" No hay registros en Silver para tunear.")
            return

        X = df[FEATURES]
        y = df['no_show_target']

        # Mismo split estratificado del entrenamiento original (03) → partición idéntica
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # ------------------------------------------------------------------
        # 1. Línea base: RF tal como quedó en 03_compare_train_models.py
        # ------------------------------------------------------------------
        print("\n🌲 Entrenando RF base (configuración del benchmark original)...")
        rf_base = RandomForestClassifier(
            n_estimators=100, max_depth=10, class_weight='balanced', random_state=42
        ).fit(X_train, y_train)
        auc_base = roc_auc_score(y_test, rf_base.predict_proba(X_test)[:, 1])
        print(f"   ➜ RF base  | ROC-AUC: {auc_base:.4f}")

        # ------------------------------------------------------------------
        # 2. Calibración de hiperparámetros (RandomizedSearchCV, scoring ROC-AUC)
        # ------------------------------------------------------------------
        print("\n🔎 RandomizedSearchCV sobre Random Forest (12 candidatos, CV=3)...")
        param_distributions = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [None, 10, 15, 20, 25],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        busqueda = RandomizedSearchCV(
            RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=1),
            param_distributions=param_distributions,
            n_iter=12, scoring='roc_auc', cv=3,
            random_state=42, n_jobs=-1, verbose=1
        )
        busqueda.fit(X_train, y_train)

        rf_tuned = busqueda.best_estimator_
        auc_tuned = roc_auc_score(y_test, rf_tuned.predict_proba(X_test)[:, 1])
        print(f"   ➜ RF tuned | ROC-AUC: {auc_tuned:.4f}")
        print(f"   ➜ Mejores hiperparámetros: {busqueda.best_params_}")

        # Se queda el mejor de los dos por ROC-AUC en test
        if auc_tuned >= auc_base:
            modelo_campeon, nombre, auc_final, params = (
                rf_tuned, "RandomForest_Tuned", auc_tuned, busqueda.best_params_
            )
        else:
            modelo_campeon, nombre, auc_final, params = (
                rf_base, "RandomForest_Baseline", auc_base,
                {"n_estimators": 100, "max_depth": 10, "class_weight": "balanced"}
            )

        # ------------------------------------------------------------------
        # 3. Threshold Tuning sobre el campeón (meta SMART: Recall >= 0.85)
        # ------------------------------------------------------------------
        probabilidades_test = modelo_campeon.predict_proba(X_test)[:, 1]
        auc_final = roc_auc_score(y_test, probabilidades_test)
        mejor_umbral = buscar_umbral(y_test, probabilidades_test)

        y_pred_umbral = (probabilidades_test >= mejor_umbral['umbral']).astype(int)
        cm = confusion_matrix(y_test, y_pred_umbral)
        print("\n📊 Matriz de confusión con el umbral optimizado (test):")
        print(f"   ➜ Asistió correcto: {cm[0][0]:>6} | Falso positivo: {cm[0][1]:>6}")
        print(f"   ➜ Falso negativo:    {cm[1][0]:>6} | Faltó correcto:  {cm[1][1]:>6}")

        # ------------------------------------------------------------------
        # 4. Guardar artefacto del campeón con metadatos de tuning
        # ------------------------------------------------------------------
        os.makedirs("models", exist_ok=True)
        ruta = "models/champion_model.pkl"
        with open(ruta, "wb") as f:
            pickle.dump({
                "model": modelo_campeon,
                "features": FEATURES,
                "model_name": nombre,
                "roc_auc": auc_final,
                "best_threshold": mejor_umbral['umbral'],
                "recall_at_threshold": mejor_umbral['recall'],
                "precision_at_threshold": mejor_umbral['precision'],
                "f1_at_threshold": mejor_umbral['f1'],
                "tuned_params": params
            }, f)

        print(f"\n ¡MODELO CAMPEÓN TUNEADO GUARDADO EN '{ruta}'!")
        print(f"   ➜ Modelo: {nombre} | ROC-AUC: {auc_final:.4f} | Umbral: {mejor_umbral['umbral']:.2f}")
        registrar_log(proceso_nombre, "GOLD", "EXITOSO", registros=len(df))

    except Exception as e:
        print(f" Error durante el tuning del modelo: {e}")
        registrar_log(proceso_nombre, "GOLD", "ERROR", error=str(e))

if __name__ == "__main__":
    tunear_modelo_campeon()
