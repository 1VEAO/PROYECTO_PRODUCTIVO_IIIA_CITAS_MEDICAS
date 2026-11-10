import os
import sys
import pandas as pd
import numpy as np

# Garantizar que Python reconozca los módulos internos de src/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import supabase

DIAS_SEMANA = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']

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

def cargar_en_lotes(tabla, registros, clave_conflicto, tamano_lote=1000):
    """Inserta registros con upsert (idempotente) en lotes de N filas"""
    total = len(registros)
    for i in range(0, total, tamano_lote):
        lote = registros[i:i + tamano_lote]
        tabla.upsert(lote, on_conflict=clave_conflicto).execute()
        print(f"   ➜ Lote {i // tamano_lote + 1} procesado ({min(i + tamano_lote, total)}/{total})")

def construir_modelo_estrella():
    """Puebla gold.dim_barrio, gold.dim_paciente, gold.dim_tiempo y gold.fact_citas"""
    proceso_nombre = "GOLD_BUILD_STAR_SCHEMA"
    registrar_log(proceso_nombre, "GOLD", "INICIADO")

    try:
        df = extraer_silver_paginado()
        if df.empty:
            print(" No hay registros en Silver para construir el modelo estrella.")
            return

        df.columns = [c.lower() for c in df.columns]
        df['appointment_day_dt'] = pd.to_datetime(df['appointment_day'], errors='coerce')

        # ------------------------------------------------------------------
        # 1. DIM_BARARIO: catálogo único de barrios
        # ------------------------------------------------------------------
        print("📍 Construyendo gold.dim_barrio...")
        barrios = sorted(set(b for b in df['neighbourhood'].dropna().unique()))
        respuesta = supabase.schema("gold").table("dim_barrio").select("id_barrio,nombre_barrio").execute()
        existentes = {r['nombre_barrio']: r['id_barrio'] for r in (respuesta.data or [])}
        nuevos = [{'nombre_barrio': b} for b in barrios if b not in existentes]
        if nuevos:
            supabase.schema("gold").table("dim_barrio").insert(nuevos).execute()
            respuesta = supabase.schema("gold").table("dim_barrio").select("id_barrio,nombre_barrio").execute()
        mapa_barrio = {r['nombre_barrio']: r['id_barrio'] for r in (respuesta.data or [])}
        print(f"   ➜ {len(mapa_barrio)} barrios en dimensión.")

        # ------------------------------------------------------------------
        # 2. DIM_TIEMPO: calendario derivado de las fechas de cita
        # ------------------------------------------------------------------
        print("📅 Construyendo gold.dim_tiempo...")
        fechas = pd.Series(df['appointment_day_dt'].dropna().dt.date).unique()
        respuesta = supabase.schema("gold").table("dim_tiempo").select("id_tiempo").execute()
        tiempos_existentes = {r['id_tiempo'] for r in (respuesta.data or [])}
        dim_tiempo = []
        for f in fechas:
            id_tiempo = int(f.strftime('%Y%m%d'))
            if id_tiempo in tiempos_existentes:
                continue
            dim_tiempo.append({
                "id_tiempo": id_tiempo,
                "fecha": f.strftime('%Y-%m-%d'),
                "anio": f.year,
                "mes": f.month,
                "trimestre": (f.month - 1) // 3 + 1,
                "dia_semana": DIAS_SEMANA[f.weekday()],
                "es_fin_semana": f.weekday() >= 5
            })
        if dim_tiempo:
            cargar_en_lotes(supabase.schema("gold").table("dim_tiempo"), dim_tiempo, "id_tiempo")
        print(f"   ➜ {len(pd.Series(fechas))} fechas en dimensión.")

        # ------------------------------------------------------------------
        # 3. DIM_PACIENTE: aggregates históricos por paciente
        # ------------------------------------------------------------------
        print("👥 Construyendo gold.dim_paciente...")
        g = df.groupby('patient_id').agg(
            primera_cita=('appointment_day_dt', 'min'),
            ultima_cita=('appointment_day_dt', 'max'),
            total_citas=('appointment_id', 'count'),
            total_noshows=('no_show_target', 'sum')
        ).reset_index()
        dim_paciente = [{
            "id_paciente": int(row.patient_id),
            "primera_cita": row.primera_cita.strftime('%Y-%m-%d') if pd.notnull(row.primera_cita) else None,
            "ultima_cita": row.ultima_cita.strftime('%Y-%m-%d') if pd.notnull(row.ultima_cita) else None,
            "total_citas": int(row.total_citas),
            "total_noshows": int(row.total_noshows)
        } for row in g.itertuples()]
        print(f"   ➜ {len(dim_paciente)} pacientes únicos a cargar...")
        cargar_en_lotes(supabase.schema("gold").table("dim_paciente"), dim_paciente, "id_paciente")

        # ------------------------------------------------------------------
        # 4. FACT_CITAS: una fila por cita con claves foráneas a las dimensiones
        # ------------------------------------------------------------------
        print("⭐ Construyendo gold.fact_citas...")
        df['id_barrio'] = df['neighbourhood'].map(mapa_barrio)
        df['id_tiempo'] = df['appointment_day_dt'].dt.strftime('%Y%m%d').apply(
            lambda x: int(x) if pd.notnull(x) else None
        )
        fact = [{
            "appointment_id": int(row.appointment_id),
            "id_paciente": int(row.patient_id),
            "id_barrio": int(row.id_barrio) if pd.notnull(row.id_barrio) else None,
            "id_tiempo": int(row.id_tiempo) if pd.notnull(row.id_tiempo) else None,
            "genero": row.gender,
            "edad": int(row.age) if pd.notnull(row.age) else None,
            "lead_time": int(row.lead_time) if pd.notnull(row.lead_time) else None,
            "scholarship": int(row.scholarship) if pd.notnull(row.scholarship) else None,
            "hipertension": int(row.hipertension) if pd.notnull(row.hipertension) else None,
            "diabetes": int(row.diabetes) if pd.notnull(row.diabetes) else None,
            "alcoholism": int(row.alcoholism) if pd.notnull(row.alcoholism) else None,
            "handcap": int(row.handcap) if pd.notnull(row.handcap) else None,
            "sms_received": int(row.sms_received) if pd.notnull(row.sms_received) else None,
            "historical_noshows": int(row.historical_noshows) if pd.notnull(row.historical_noshows) else None,
            "no_show_target": int(row.no_show_target) if pd.notnull(row.no_show_target) else None
        } for row in df.itertuples()]
        cargar_en_lotes(supabase.schema("gold").table("fact_citas"), fact, "appointment_id")

        print(" ¡Modelo Estrella (Gold analítico) construido con éxito!")
        registrar_log(proceso_nombre, "GOLD", "EXITOSO", registros=len(fact))

    except Exception as e:
        print(f" Error durante la construcción del modelo estrella: {e}")
        registrar_log(proceso_nombre, "GOLD", "ERROR", error=str(e))

if __name__ == "__main__":
    construir_modelo_estrella()
