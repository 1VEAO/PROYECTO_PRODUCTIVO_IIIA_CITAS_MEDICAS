"""
08_simular_encuestas.py
Simula encuestas de satisfacción (piloto del ciclo de fidelización) y las
registra en gold.feedback_loop, vinculadas a citas reales del dataset:
- Citas ATENDIDAS (no_show_target = 0) → "Atención Completada Normal" con
  satisfacción alta (medias de 4-5 estrellas) y duraciones típicas de consulta.
- Citas INASISTIDAS (no_show_target = 1) → "Reprogramada Otra Fecha"
  (paciente recuperado por el sistema) con satisfacción moderada (3-4).

Determinista: semilla 42. Idempotente: se niega a duplicar si la tabla ya
tiene registros (usar --force para vaciar y regenerar).
"""
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import supabase

SEMILLA = 42
N_COMPLETADAS = 120
N_REPROGRAMADAS = 30

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

def contar_filas():
    r = supabase.schema("gold").table("feedback_loop").select("id", count="exact").limit(0).execute()
    return r.count

def extraer_muestra(no_show_target, n, rng):
    """Extrae `n` appointment_ids reales de gold.fact_citas según el desenlace."""
    r = (supabase.schema("gold").table("fact_citas")
         .select("appointment_id")
         .eq("no_show_target", no_show_target)
         .limit(5000).execute())
    ids = [fila["appointment_id"] for fila in r.data]
    elegidos = rng.choice(len(ids), size=n, replace=False)
    return [ids[i] for i in elegidos]

def main():
    print("=" * 70)
    print("  SIMULACIÓN DE ENCUESTAS DE SATISFACCIÓN → gold.feedback_loop")
    print("=" * 70)

    existentes = contar_filas()
    if existentes and "--force" not in sys.argv:
        print(f"⚠️  La tabla ya tiene {existentes} registros. "
              f"Usa '--force' para vaciarla y regenerar el piloto.")
        return
    if existentes and "--force" in sys.argv:
        supabase.schema("gold").table("feedback_loop").delete().neq("id", 0).execute()
        print(f"🗑️  {existentes} registros previos eliminados (--force).")

    rng = np.random.default_rng(SEMILLA)
    encuestas = []

    # Citas atendidas → encuestas de atención normal
    ids_atendidas = extraer_muestra(0, N_COMPLETADAS, rng)
    for aid in ids_atendidas:
        satisfaccion = int(rng.choice([3, 4, 5], p=[0.12, 0.43, 0.45]))
        duracion = int(np.clip(rng.normal(23, 6), 10, 50))
        encuestas.append({
            "appointment_id": int(aid),
            "actual_duration_min": duracion,
            "satisfaction_score": satisfaccion,
            "rescheduled_status": "Atención Completada Normal",
        })

    # Citas inasistidas → paciente recuperado vía reprogramación
    ids_ausentes = extraer_muestra(1, N_REPROGRAMADAS, rng)
    for aid in ids_ausentes:
        satisfaccion = int(rng.choice([2, 3, 4, 5], p=[0.10, 0.40, 0.35, 0.15]))
        duracion = int(np.clip(rng.normal(27, 8), 10, 55))
        encuestas.append({
            "appointment_id": int(aid),
            "actual_duration_min": duracion,
            "satisfaction_score": satisfaccion,
            "rescheduled_status": "Reprogramada Otra Fecha",
        })

    # Inserción por lotes
    lote = 100
    for i in range(0, len(encuestas), lote):
        supabase.schema("gold").table("feedback_loop").insert(encuestas[i:i + lote]).execute()

    total = contar_filas()
    registrar_log("08_simular_encuestas", "gold.feedback_loop", "COMPLETADO", registros=total)

    print(f"✅ Encuestas insertadas: {len(encuestas)} | Total en tabla: {total}")
    print(f"   • Atención Completada Normal : {N_COMPLETADAS}")
    print(f"   • Reprogramada Otra Fecha    : {N_REPROGRAMADAS} (pacientes recuperados)")
    sat = [e["satisfaction_score"] for e in encuestas]
    dur = [e["actual_duration_min"] for e in encuestas]
    print(f"   • Satisfacción media: {np.mean(sat):.2f} / 5.0")
    print(f"   • Duración media    : {np.mean(dur):.1f} min")

if __name__ == "__main__":
    main()
