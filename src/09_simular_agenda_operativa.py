"""
09_simular_agenda_operativa.py
Construye la AGENDA OPERATIVA SIMULADA (modo replay) para demostrar la app
como si el sistema estuviera operando en una clínica real:

- Toma citas REALES del histórico (gold.fact_citas, con sus features reales).
- Las asigna a los próximos 5 días (hoy +0..+4) con horas de atención
  simuladas, y recalcula `scheduled_day` de forma coherente
  (fecha de cita − lead_time real → parece reservada hace N días).
- Puebla gold.score_input (citas pendientes de evaluar) y ejecuta el
  pipeline REAL de scoring (src/04_predict_scoring.py) para producir
  gold.score_output: la agenda priorizada que consume la app.

Determinista (semilla 42). Idempotente: --force para regenerar.
"""
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import supabase

SEMILLA = 42
CITAS_POR_DIA = [60, 50, 40, 30, 20]          # agenda decreciente como semana real
HORA_INICIO, HORA_FIN = 7.5, 17.0             # atención de 07:30 a 17:00
SLOT_MINUTOS = 15

def registrar_log(proceso, capa, estado, registros=0, error=None):
    try:
        supabase.schema("meta").table("log_procesos").insert({
            "proceso_nombre": proceso, "capa_destino": capa,
            "registros_procesados": registros, "estado": estado,
            "mensaje_error": str(error) if error else None,
            "usuario_ejecutor": "PYTHON_SCRIPT"
        }).execute()
    except Exception as e:
        print(f"⚠️  No se pudo registrar log en meta: {e}")

def contar(tabla):
    r = supabase.schema("gold").table(tabla).select("id", count="exact").limit(0).execute()
    return r.count

def extraer_candidatas(n_total):
    """Citas reales con lead_time suficiente para que la reserva quede en el pasado."""
    print(" Consultando citas reales candidatas en gold.fact_citas...")
    r = (supabase.schema("gold").table("fact_citas")
         .select("appointment_id, id_paciente, edad, lead_time, scholarship, "
                 "hipertension, diabetes, alcoholism, handcap, sms_received, "
                 "historical_noshows")
         .gte("lead_time", 5)                     # garantiza scheduled_day en el pasado
         .order("appointment_id").limit(5000).execute())
    df = pd.DataFrame(r.data)
    # Nombres del modelo estrella → nombres del pipeline de scoring
    df = df.rename(columns={"id_paciente": "patient_id", "edad": "age"})
    if len(df) < n_total:
        raise ValueError(f" Solo {len(df)} candidatas; se necesitan {n_total}.")
    rng = np.random.default_rng(SEMILLA)
    return df.iloc[np.sort(rng.choice(len(df), size=n_total, replace=False))].reset_index(drop=True)

def construir_agenda(df):
    """Asigna fecha futura + hora de atención y recalcula la fecha de reserva."""
    hoy = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    rng = np.random.default_rng(SEMILLA + 1)
    filas = []
    idx = 0
    print("\n Construyendo agenda de los próximos 5 días (modo replay):")
    for offset, n_dia in enumerate(CITAS_POR_DIA):
        fecha_cita = hoy + timedelta(days=offset)
        etiqueta = "HOY" if offset == 0 else f"+{offset} días"
        # Horas simuladas ordenadas dentro del día (slots de 15 min)
        fracciones = np.sort(rng.uniform(HORA_INICIO, HORA_FIN, size=n_dia))
        for frac in fracciones:
            cita = df.iloc[idx]
            hora = int(frac)
            minuto = int((frac - hora) * 60)
            fecha_hora = fecha_cita.replace(hour=hora, minute=minuto)
            fecha_reserva = fecha_hora - timedelta(days=int(cita["lead_time"]))
            filas.append({
                "appointment_id": int(cita["appointment_id"]),
                "patient_id": int(cita["patient_id"]),
                "scheduled_day": fecha_reserva.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "appointment_day": fecha_hora.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "lead_time": int(cita["lead_time"]),
                "age": int(cita["age"]),
                "scholarship": int(cita["scholarship"]),
                "hipertension": int(cita["hipertension"]),
                "diabetes": int(cita["diabetes"]),
                "alcoholism": int(cita["alcoholism"]),
                "handcap": int(cita["handcap"]),
                "sms_received": int(cita["sms_received"]),
                "historical_noshows": int(cita["historical_noshows"]),
            })
            idx += 1
        print(f"   ▶ {etiqueta}: {n_dia} citas pendientes ({fecha_cita:%Y-%m-%d})")
    return pd.DataFrame(filas)

def main():
    print("=" * 70)
    print("  AGENDA OPERATIVA SIMULADA → score_input → scoring → score_output")
    print("=" * 70)

    n_previo = contar("score_input")
    if n_previo and "--force" not in sys.argv:
        print(f"⚠️  score_input ya tiene {n_previo} citas. Usa '--force' para regenerar la agenda.")
        return
    if n_previo:
        supabase.schema("gold").table("score_input").delete().neq("id", 0).execute()
        print(f"🗑️  Agenda previa eliminada ({n_previo} citas de score_input).")

    n_total = sum(CITAS_POR_DIA)
    df_candidatas = extraer_candidatas(n_total)
    df_agenda = construir_agenda(df_candidatas)

    print(f"\n Insertando {len(df_agenda)} citas pendientes en gold.score_input...")
    registros = df_agenda.where(pd.notnull(df_agenda), None).to_dict(orient="records")
    for i in range(0, len(registros), 100):
        supabase.schema("gold").table("score_input").insert(registros[i:i + 100]).execute()

    registrar_log("09_simular_agenda_operativa", "gold.score_input", "COMPLETADO", registros=len(registros))

    # Ejecutar el JOB REAL de scoring (src/04) sobre la agenda nueva
    print("\n Ejecutando el job diario de scoring (src/04_predict_scoring.py)...")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "scoring", os.path.join(os.path.dirname(os.path.abspath(__file__)), "04_predict_scoring.py"))
    scoring = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scoring)
    scoring.ejecutar_scoring_predicciones()

    print(f"\n✅ Agenda operativa lista: {len(registros)} citas pendientes puntuadas en gold.score_output.")

if __name__ == "__main__":
    main()
