# 🩺 Sistema Inteligente de Gestión de Citas Médicas

Proyecto Productivo IIIA — Instituto Continental (2026)
**Carrera:** Ciencia de Datos e Inteligencia Artificial

Sistema de predicción de **inasistencia a citas médicas (no-show)** con arquitectura de datos **Medallion** (Bronze → Silver → Gold) sobre **Supabase**, modelos de Machine Learning comparados por benchmark, y aplicación **Streamlit** para el personal de admisión.

El sistema permite a una clínica anticipar qué pacientes tienen alta probabilidad de faltar a su cita y ejecutar una acción preventiva diferenciada: 📞 llamada telefónica (riesgo alto), 💬 WhatsApp/SMS interactivo (riesgo medio) o ✉️ correo estándar (riesgo bajo).

---

## 🏗️ Arquitectura

```
                        ┌─────────────────────────────────────────────┐
                        │              SUPABASE (Postgres)            │
 data/raw/              │                                             │
 citas_medicas_raw.csv  │  ┌────────┐   ┌────────┐   ┌─────────────┐  │
        │               │  │ BRONZE │──▶│ SILVER │──▶│    GOLD     │  │
        ▼               │  │        │   │  (ETL) │   │ (ML + apps) │  │
 ┌──────────────┐       │  └────────┘   └────────┘   └─────────────┘  │
 │ 01_ingest_   ├──────▶│  citas_raw   citas_cleaned   ml_train_      │
 │ bronze.py    │       │  (inmutable) +features       dataset        │
 └──────────────┘       │                             score_input     │
 ┌──────────────┐       │  ┌────────┐                 score_output    │
 │ 02_process_  ├──────▶│  │  META  │   log_procesos   feedback_loop  │
 │ silver.py    │       │  └────────┘   (auditoría)                  │
 └──────────────┘       │        ▲                                    │
 ┌──────────────┐       │________│___________________________________│
 │ 05_build_    ├──────▶│ MODELO ESTRELLA (Gold analítico):           │
 │ star_schema  │       │  dim_barrio · dim_paciente · dim_tiempo     │
 └──────────────┘       │   → fact_citas (110.527 hechos)             │
 ┌──────────────┐       └─────────────────────────────────────────────┘
 │ 03_compare_  ├──────▶ Entrena 4 modelos → benchmark
 │ train_models │
 └──────────────┘      ┌─────────────────────────────────────────────┐
 ┌──────────────┐      │ 06_tune_threshold.py                        │
 │ 06_tune_     ├──────▶ Calibra hiperparámetros (RandomizedSearchCV)│
 │ threshold.py │      │ + Threshold Tuning → champion_model.pkl     │
 └──────────────┘      └─────────────────────────────────────────────┘
 ┌──────────────┐      ┌─────────────────────────────────────────────┐
 │ 04_predict_  ├──────▶ scoring batch → gold.score_output (riesgo)  │
 │ scoring.py   │      └─────────────────────────────────────────────┘
 └──────────────┘      ┌─────────────────────────────────────────────┐
 ┌──────────────┐      │ 07_simulacion_montecarlo.py                 │
 │   app.py     ├────▶│ Viabilidad económica: citas recuperadas y   │
 └──────────────┘      │ ahorro por 1.000 citas (10.000 iteraciones) │
   Streamlit (web)     └─────────────────────────────────────────────┘
```

| Capa | Contenido | Principio |
|---|---|---|
| **Bronze** | `citas_raw` — datos crudos tal cual llegan del CSV | Carga inmutable |
| **Silver** | `citas_cleaned` — datos limpios con *feature engineering* | Calidad y trazabilidad |
| **Gold** | `ml_train_dataset`, `score_input`, `score_output`, `feedback_loop` | Listo para consumo (ML y app) |
| **Meta** | `log_procesos` — auditoría de cada ejecución ETL | Observabilidad |

**Features generadas en Silver:**
- `lead_time` — días de anticipación entre la programación y la cita.
- `grupo_edad` — Niño / Joven / Adulto / Adulto Mayor.
- `historical_noshows` — inasistencias acumuladas del paciente **sin data leakage** (cumsum desplazado: no incluye la cita actual).
- `no_show_target` — variable objetivo (1 = faltó, 0 = asistió).
- `alerta_calidad_logica` — bandera de auditoría para edades negativas o lead times negativos.

---

## 📁 Estructura del repositorio

```
├── app.py                          # Aplicación Streamlit (5 módulos)
├── requirements.txt                # Dependencias con versiones fijadas
├── .env.example                    # Plantilla de credenciales (sin secretos)
├── sql/
│   └── schema_medallion.sql        # DDL completo: esquemas, 11 tablas, RLS, grants
├── src/
│   ├── config.py                   # Cliente Supabase (lee .env)
│   ├── 01_ingest_bronze.py         # CSV → bronze.citas_raw (lotes de 1000)
│   ├── 02_process_silver.py        # Bronze → Silver (limpieza + features)
│   ├── 03_compare_train_models.py  # Silver → Gold + benchmark de 4 modelos
│   ├── 04_predict_scoring.py       # Inferencia batch → gold.score_output
│   ├── 05_build_star_schema.py     # Silver → dim_* + fact_citas (Gold analítico)
│   ├── 06_tune_threshold.py        # Hiperparámetros + threshold tuning del campeón
│   └── 07_simulacion_montecarlo.py # Simulación de viabilidad económica
├── models/
│   └── champion_model.pkl          # Modelo campeón versionado (despliegue)
├── data/raw/
│   └── citas_medicas_raw.csv       # Dataset crudo (110.527 registros)
├── reports/
│   └── simulacion_montecarlo_resumen.csv
└── notebook/
    └── 00_eda_exploratorio_bronze.ipynb
```

---

## 📊 Dataset

- **Fuente:** [*Medical Appointment No Shows* (Kaggle)](https://www.kaggle.com/datasets/joniaro/medical-appointment-no-shows) — datos reales de citas de clínicas públicas de **Vitória, Espírito Santo (Brasil), 2016**.
- **Tamaño:** 110.527 citas, sin valores nulos.
- El CSV crudo está versionado en `data/raw/` para que el proyecto sea **100% replicable desde el clon**.

| Columna original | Descripción |
|---|---|
| `PatientId` | Identificador del paciente |
| `AppointmentID` | Identificador de la cita |
| `Gender` | F / M |
| `ScheduledDay` | Fecha-hora en que se programó la cita |
| `AppointmentDay` | Fecha de la cita |
| `Age` | Edad del paciente |
| `Neighbourhood` | Barrio de la clínica |
| `Scholarship` | 1 si recibe apoyo social (Bolsa Família) |
| `Hipertension`, `Diabetes`, `Alcoholism`, `Handcap` | Condiciones clínicas (0/1) |
| `SMS_received` | 1 si recibió recordatorio por SMS |
| `No-show` | **Target:** Yes (faltó) / No (asistió) |

---

## 🚀 Guía de replicación completa

### 1. Requisitos previos

- **Python 3.10+** (probado con Python 3.14)
- Cuenta gratuita en [Supabase](https://supabase.com)

### 2. Clonar e instalar dependencias

```bash
git clone https://github.com/Yaseli-M/PROYECTO_PRODUCTIVO_IIIA_CITAS_MEDICAS.git
cd PROYECTO_PRODUCTIVO_IIIA_CITAS_MEDICAS

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Crear tu propia instancia de Supabase

> ⚠️ Crea **tu propio proyecto**; no reutilices el de otra persona para no alterar sus datos.

1. **Crear proyecto** en el dashboard de Supabase (elige nombre y región; espera ~2 min a que aprovisione).
2. **Crear el esquema de base de datos:** abre **SQL Editor → New query**, pega el contenido completo de [`sql/schema_medallion.sql`](sql/schema_medallion.sql) y ejecuta **Run**. Esto crea los 4 esquemas, las 11 tablas (7 MLOps + 4 del modelo estrella), índices, permisos y políticas RLS.
3. **Exponer los esquemas a la API** *(paso crítico que se suele olvidar):*
   - Ve a **Settings → API → "Exposed schemas in API"**.
   - Reemplaza el valor por defecto (`public`) por:
     ```
     public, bronze, silver, gold, meta
     ```
   - Guarda. Sin esto, los scripts fallan con `PGRST106: Invalid schema`.
4. **Obtener credenciales:** en **Settings → API** copia la *Project URL* y la clave (`anon` o `secret`).

### 4. Configurar las credenciales

```bash
cp .env.example .env
# Edita .env con tu SUPABASE_URL y SUPABASE_KEY
```

> 🔐 El `.env` está en `.gitignore` y **nunca** debe versionarse. La clave `sb_secret_...` solo para uso en scripts locales; la clave `anon` es la adecuada si despliegas la app públicamente.

### 5. Ejecutar el pipeline (en orden estricto)

```bash
python src/01_ingest_bronze.py        # ~3 min  → 110.527 filas a Bronze
python src/02_process_silver.py      # ~5 min  → ETL + features a Silver
python src/03_compare_train_models.py # ~3 min → benchmark + champion inicial
python src/05_build_star_schema.py   # ~5 min  → modelo estrella (dim_* + fact_citas)
python src/06_tune_threshold.py      # ~8 min  → hiperparámetros + threshold tuning
python src/04_predict_scoring.py     # ~1 min  → scoring batch con el campeón tuneado
python src/07_simulacion_montecarlo.py # ~5 s  → viabilidad económica (reports/)
```

Cada script ETL registra su ejecución (INICIADO / EXITOSO / ERROR) en `meta.log_procesos`.

> ⚠️ **Idempotencia:** los scripts `05` y `06` son re-ejecutables (usan upserts y regeneran el artefacto). En cambio, re-ejecutar `01`, `02` o `03` duplicará filas en su tabla destino: antes de repetirlos, limpia la tabla desde el SQL Editor (`TRUNCATE`). El script `04` limpia él mismo `score_output` antes de insertar.

### 6. Lanzar la aplicación

```bash
streamlit run app.py
```

Se abre en `http://localhost:8501`. La app detecta automáticamente si hay conexión a Supabase ("🟢 Supabase Conectado") o corre en modo local.

---

## 🤖 Modelo predictivo

### Benchmark (split estratificado 80/20, `random_state=42` → resultados reproducibles)

| Modelo | Recall (clase "faltó") | F1-Score | ROC-AUC |
|---|---|---|---|
| LogisticRegression (baseline) | 0.5802 | 0.4084 | 0.6738 |
| RandomForest (benchmark inicial) | 0.8219 | 0.4441 | 0.7351 |
| XGBoost | 0.7764 | 0.4413 | 0.7286 |
| LightGBM | 0.8125 | 0.4458 | 0.7353 |

- **Criterio de selección del campeón:** maximizar el **Recall de la clase "faltó"** — desde el punto de vista del negocio, es más costoso no contactar a un paciente que sí faltaría que contactar de más.
- **Desbalance de clases (~20% no-show):** `class_weight='balanced'` en sklearn/LightGBM y `scale_pos_weight` calculado en XGBoost.

### Calibración final del campeón (`06_tune_threshold.py`)

1. **Hiperparámetros** calibrados con `RandomizedSearchCV` (12 candidatos, CV=3, scoring ROC-AUC) sobre Random Forest: `n_estimators=200, max_depth=10, min_samples_split=10, min_samples_leaf=4`.
2. **Threshold Tuning** (calibración del umbral de decisión): de los 44 umbrales que cumplen la meta **Recall ≥ 0.85**, se elige el de mejor F1-Score:

| Métrica del campeón final (`RandomForest_Tuned`) | Valor |
|---|---|
| ROC-AUC (independiente del umbral) | 0.7352 |
| **Umbral de decisión optimizado** | **0.48** |
| **Recall con umbral optimizado** | **0.8600** ✅ meta SMART ≥ 0.85 cumplida |
| Precision con umbral optimizado | 0.2970 |
| F1 con umbral optimizado | 0.4415 |
| Falsos negativos en test (20% ≈ 22.106 citas) | 625 |

> 📌 **Trade-off documentado:** priorizar el Recall (minimizar pacientes en riesgo no detectados) eleva los falsos positivos (Precision 0.30). Es la decisión correcta para este caso de uso: una llamada "de más" cuesta minutos; una inasistencia no atendida cuesta una consulta completa.

### Viabilidad económica (`07_simulacion_montecarlo.py`)

Simulación de Montecarlo (10.000 iteraciones, semilla=42, supuestos documentados en el script) sobre una agenda de 10.000 citas/mes:

| Escenario | Citas recuperadas/mes (P5–P95) | Costo por 1.000 citas |
|---|---|---|
| Pesimista | 167 – 262 | S/ 10.096 → S/ 8.867 |
| **Base** | **252 – 381** | **S/ 10.096 → S/ 8.366** |
| Optimista | 360 – 531 | S/ 10.096 → S/ 7.733 |

Resultados detallados en [`reports/simulacion_montecarlo_resumen.csv`](reports/simulacion_montecarlo_resumen.csv).

### Reglas de negocio (bandas de riesgo operativas)

| Probabilidad de inasistencia | Clasificación | Acción recomendada |
|---|---|---|
| ≥ 70% | 🔴 Prioridad Alta | 📞 Llamada telefónica preventiva (24 h antes) |
| 40% – 69% | 🟡 Prioridad Media | 💬 Recordatorio interactivo WhatsApp/SMS |
| < 40% | 🟢 Prioridad Normal | ✉️ Recordatorio estándar |

> Las bandas de riesgo (70%/40%) son la **segmentación operativa** de la agenda; el umbral 0.48 es el **punto de corte de clasificación** del modelo, calibrado para la meta de Recall.

---

## 🖥️ Aplicación Streamlit

| Módulo | Función |
|---|---|
| 📊 **Resumen de la Agenda** | KPIs de citas y distribución de acciones sugeridas |
| 🔮 **Evaluador de Citas** | Formulario del paciente → probabilidad de inasistencia, nivel de riesgo, mensaje de WhatsApp prellenado (`wa.me`) y registro en `gold.score_output` |
| 🗺️ **Ubicación y Cobertura** | Análisis por barrio (Vitória, Brasil) con filtros interactivos |
| 🔄 **Registro de Atención** | Feedback post-consulta (duración, satisfacción, reprogramación) → `gold.feedback_loop` para reentrenamientos futuros |
| ℹ️ **Información y Ayuda** | Ficha institucional y guía de uso |

---

## 🛠️ Solución de problemas

| Síntoma | Causa | Solución |
|---|---|---|
| `PGRST106: Invalid schema: bronze` | Esquemas no expuestos a la API | Settings → API → *Exposed schemas* → `public, bronze, silver, gold, meta` |
| `PGRST002: Could not query the database for the schema cache` | Se modificó la configuración (o DDL) durante una ejecución ETL | Espera unos segundos y re-ejecuta el script fallido; limpia antes la tabla destino |
| `No se encontró el archivo en la ruta: data/raw/...` | Falta el CSV | Verifica que `data/raw/citas_medicas_raw.csv` exista |
| Filas duplicadas en Bronze/Silver/Gold | Re-ejecución de scripts no idempotentes | `TRUNCATE` de la tabla destino y volver a ejecutar solo ese paso |
| La app dice "El motor predictivo no está disponible" | Falta `models/champion_model.pkl` | Ejecuta `python src/03_compare_train_models.py` |
| `No se encontraron 'SUPABASE_URL' o 'SUPABASE_KEY'` | Falta el `.env` | Crea el `.env` a partir de `.env.example` |

---

## 👥 Equipo de desarrollo

1. **Alarcón Olmedo Víctor Efraín**
2. **Inuma Macedo Mariana Milagros**
3. **Méndez Macedo Karen Yaseli**
4. **Salinas Llana Jhim Anthony**

**Docente:** Orizano Salvador Sergio Víctor
**Institución:** Instituto Continental — Proyecto Productivo IIIA (2026)
