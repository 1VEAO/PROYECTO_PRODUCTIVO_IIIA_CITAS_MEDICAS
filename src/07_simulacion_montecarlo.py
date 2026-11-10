import os
import csv
import numpy as np

# ============================================================================
# SIMULACIÓN DE MONTECARLO - Viabilidad económica del sistema
# Respalda los puntos 9 y 10 del CPP1 (Marco Lógico y Resultados Esperados):
#   - Recuperación mensual de cupos (evaluada sobre 10,000 citas/mes)
#   - Reducción del costo operativo por inasistencia por cada 1,000 citas
#
# Supuestos (documentados y modificables):
#   - Tasa global de no-show: posterior Beta(22319, 88208) del dataset real
#     (22,319 inasistencias sobre 110,527 citas → media 20.19%).
#   - Distribución de la agenda por nivel de riesgo: Triangular con moda en
#     el 15.3% de Riesgo Alto reportado en el punto 10 del CPP1.
#   - Efectividad de las intervenciones (proporción de no-shows recuperados),
#     basada en la literatura de recordatorios en salud (Topuz et al., 2018;
#     Hospital Kuala Lumpur, 2023): llamada > WhatsApp/SMS > correo pasivo.
#   - Valor de la consulta recuperada: S/ 50 (costo estándar de una consulta
#     no ejecutada en el sector público) y costos de contacto por canal.
# ============================================================================

N_SIMULACIONES = 10_000
CITAS_MES = 10_000
VALOR_CONSULTA = 50.0          # S/ por consulta recuperada
COSTO_LLAMADA = 0.80           # S/ por contacto telefónico
COSTO_WHATSAPP = 0.10          # S/ por mensaje interactivo
COSTO_CORREO = 0.02            # S/ por correo estándar
SEMILLA = 42

# (mínimo, moda, máximo) de inasistencia por nivel de riesgo
TASA_NOSHOW = {
    "alto":  (0.40, 0.50, 0.60),
    "medio": (0.18, 0.24, 0.30),
    "bajo":  (0.06, 0.10, 0.14)
}

# Rangos de efectividad (proporción de inasistencias que la acción recupera)
ESCENARIOS = {
    "Pesimista": {
        "alto":  (0.15, 0.22), "medio": (0.08, 0.12), "bajo": (0.01, 0.03)
    },
    "Base": {
        "alto":  (0.22, 0.30), "medio": (0.12, 0.18), "bajo": (0.03, 0.06)
    },
    "Optimista": {
        "alto":  (0.30, 0.40), "medio": (0.18, 0.25), "bajo": (0.06, 0.10)
    }
}

# Semillas deterministas por escenario (hash() de str NO es reproducible entre
# ejecuciones porque Python lo aleatoriza por proceso)
SEMILLA_ESCENARIO = {"Pesimista": 1, "Base": 2, "Optimista": 3}

def simular_escenario(rng, efectividad):
    """Ejecuta N simulaciones de un escenario y devuelve métricas por mes."""
    # Posterior Beta de la tasa global de no-show (del dataset real)
    tasa_global = rng.beta(22319, 88208, N_SIMULACIONES)

    # Distribución de la agenda por nivel de riesgo (moda 15.3% alto, CPP1 p.10)
    prop_alto = rng.triangular(0.10, 0.153, 0.20, N_SIMULACIONES)
    prop_medio = rng.triangular(0.20, 0.25, 0.30, N_SIMULACIONES)
    prop_bajo = 1.0 - prop_alto - prop_medio

    # Tasa de inasistencia por nivel
    ns_alto = rng.triangular(*TASA_NOSHOW['alto'], N_SIMULACIONES)
    ns_medio = rng.triangular(*TASA_NOSHOW['medio'], N_SIMULACIONES)
    ns_bajo = rng.triangular(*TASA_NOSHOW['bajo'], N_SIMULACIONES)

    # Efectividad de cada intervención (uniforme en el rango del escenario)
    ef_alto = rng.uniform(*efectividad['alto'], N_SIMULACIONES)
    ef_medio = rng.uniform(*efectividad['medio'], N_SIMULACIONES)
    ef_bajo = rng.uniform(*efectividad['bajo'], N_SIMULACIONES)

    # Citas recuperadas por mes = Σ citas_nivel × inasistencias_nivel × efectividad
    recuperadas = (
        CITAS_MES * prop_alto * ns_alto * ef_alto
        + CITAS_MES * prop_medio * ns_medio * ef_medio
        + CITAS_MES * prop_bajo * ns_bajo * ef_bajo
    )

    # Costos mensuales (S/)
    costo_actual = CITAS_MES * tasa_global * VALOR_CONSULTA
    costo_contactos = (
        CITAS_MES * prop_alto * COSTO_LLAMADA
        + CITAS_MES * prop_medio * COSTO_WHATSAPP
        + CITAS_MES * prop_bajo * COSTO_CORREO
    )
    inasistencias_totales = CITAS_MES * (
        prop_alto * ns_alto + prop_medio * ns_medio + prop_bajo * ns_bajo
    )
    costo_intervencion = (
        (inasistencias_totales - recuperadas) * VALOR_CONSULTA + costo_contactos
    )
    return recuperadas, costo_actual, costo_intervencion, costo_contactos

def main():
    rng = np.random.default_rng(SEMILLA)
    filas = []
    columnas = [
        "escenario", "citas_recuperadas_p5", "citas_recuperadas_p50",
        "citas_recuperadas_p95", "costo_actual_p50_soles",
        "costo_intervencion_p50_soles", "ahorro_p50_soles",
        "costo_por_1000_citas_actual", "costo_por_1000_citas_intervencion"
    ]

    print("=" * 74)
    print("  SIMULACIÓN DE MONTECARLO - Recuperación de citas y ahorro operativo")
    print(f"  Escenarios: {len(ESCENARIOS)} | Iteraciones: {N_SIMULACIONES:,} | Agenda: {CITAS_MES:,} citas/mes")
    print("=" * 74)

    for nombre, efectividad in ESCENARIOS.items():
        # Cada escenario usa un generador derivado para reproducibilidad total
        rng_escenario = np.random.default_rng(SEMILLA + SEMILLA_ESCENARIO[nombre])
        recuperadas, c_actual, c_interv, c_contactos = simular_escenario(rng_escenario, efectividad)

        p5, p50, p95 = np.percentile(recuperadas, [5, 50, 95])
        ahorro = c_actual - c_interv
        filas.append({
            "escenario": nombre,
            "citas_recuperadas_p5": round(p5),
            "citas_recuperadas_p50": round(p50),
            "citas_recuperadas_p95": round(p95),
            "costo_actual_p50_soles": round(np.median(c_actual), 2),
            "costo_intervencion_p50_soles": round(np.median(c_interv), 2),
            "ahorro_p50_soles": round(np.median(ahorro), 2),
            "costo_por_1000_citas_actual": round(np.median(c_actual) / (CITAS_MES / 1000), 2),
            "costo_por_1000_citas_intervencion": round(np.median(c_interv) / (CITAS_MES / 1000), 2)
        })

        print(f"\n ▶ Escenario {nombre}:")
        print(f"   ➜ Citas recuperadas/mes: P5={p5:,.0f} | P50={p50:,.0f} | P95={p95:,.0f}")
        print(f"   ➜ Costo mensual sin sistema:  S/ {np.median(c_actual):>12,.2f}")
        print(f"   ➜ Costo mensual con sistema:  S/ {np.median(c_interv):>12,.2f}  (contactos: S/ {np.median(c_contactos):,.2f})")
        print(f"   ➜ Ahorro mensual esperado:    S/ {np.median(ahorro):>12,.2f}")
        print(f"   ➜ Costo por 1,000 citas: S/ {np.median(c_actual)/(CITAS_MES/1000):,.2f} → S/ {np.median(c_interv)/(CITAS_MES/1000):,.2f}")

    # Guardar resumen reproducible en reports/
    ruta = os.path.join("reports", "simulacion_montecarlo_resumen.csv")
    os.makedirs("reports", exist_ok=True)
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columnas)
        writer.writeheader()
        writer.writerows(filas)
    print(f"\n✅ Resumen guardado en '{ruta}' (semilla={SEMILLA}, reproducible).")

if __name__ == "__main__":
    main()
