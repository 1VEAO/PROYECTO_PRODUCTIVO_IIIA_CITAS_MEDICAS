# Guion de Exposición — Sistema Inteligente de Predicción de Inasistencias a Citas Médicas

> **Supuesto de duración:** 20 minutos de exposición (≈ 6–7 min por expositor) + ronda de preguntas.
> Si el tiempo es de 15 min, usar la versión abreviada marcada con ✂️ en cada bloque.
> **Equipo de 4 integrantes, 3 expositores:** quien no exponga puede operar la demo (notebook/app/Supabase) y auxiliar en las preguntas.

---

## Reparto de bloques

| Expositor | Bloque | Minutos | Puntos del informe que cubre |
|---|---|---|---|
| **Expositor 1** | Contexto, problema, datos y arquitectura | 6 | 1, 2, 3, 6, 7 |
| **Expositor 2** | Metodología, modelado, métricas y comparación | 7 | 11, 12, 13, 14 |
| **Expositor 3** | Resultados de negocio, economía, conclusiones | 6 | 9, 10, 15, 16 |

---

## EXPOSITOR 1 — Contexto, problema, datos y arquitectura (6 min)

### Apertura (30 s) — memorizar la primera frase

> "Buenos días. Somos el equipo [nombres]. Presentamos un sistema de inteligencia artificial que predice con días de anticipación qué pacientes no asistirán a sus citas médicas, para que el centro de salud pueda recuperar esas consultas antes de que se pierdan. Todo lo que veremos hoy está calculado en vivo: cada cifra se puede verificar ejecutando el código."

### 1. El problema (90 s)

- Los centros de salud públicos pierden **~2 de cada 10 consultas por inasistencia no notificada**: en nuestro dataset real, **110.527 citas con 20,19% de ausentismo (22.319 citas perdidas)**.
- Consecuencias: consultorios ociosos, listas de espera más largas y barreras de acceso.
- Mostrar el gráfico de **lead time** (notebook, punto 2): mismo día 4,6% vs. más de 30 días 33,0%. Frase clave: *"la anticipación de la reserva es la variable que más predice la ausencia"*.
- Mencionar la **paradoja del SMS**: con recordatorio masivo el ausentismo es 27,6% vs 16,7% sin él → enviar mensajes a todos no funciona; hace falta segmentar con predicción.

### 2. Objetivo SMART (60 s)

- Construir un clasificador con **Recall ≥ 0,85** sobre la clase "no asistirá", segmentar en 3 niveles de riesgo y desplegar un MVP funcional.
- Adelantar el resultado: *"la meta se cumplió: Recall 0,86 con umbral calibrado"*. (El detalle lo desarrolla el Expositor 2.)

### 3. Arquitectura Medallion en Supabase (3 min) — DEMO

Mostrar en el notebook la **tabla de inventario de las 11 tablas** (celda del punto 1) mientras se explica:

- **Bronze:** `citas_raw`, 110.527 registros crudos e inmutables, con metadatos de origen.
- **Silver:** `citas_cleaned` — estandarización y *feature engineering*: `lead_time`, `grupo_edad`, e historial de ausentismo acumulado **sin fuga de datos**.
- **Gold:** dos niveles — (a) **modelo estrella** para analítica: `dim_barrio` (81 barrios), `dim_paciente` (62.299 pacientes), `dim_tiempo` (27 fechas), `fact_citas` (110.527 hechos); (b) tablas MLOps: `ml_train_dataset`, `score_input`, `score_output`, `feedback_loop`.
- **Auditoría:** cada script registra ejecución y volumen en `meta.log_procesos`.
- Si hay conexión: abrir el panel de Supabase 20 segundos y mostrar `fact_citas`.

### Cierre del bloque (30 s)

> "Con la base lista y verificada, la pregunta siguiente era qué algoritmo predice mejor. Eso lo presenta [nombre del Expositor 2]."

### ✂️ Versión abreviada (si tocan 15 min): solo problema + lead time + inventario de tablas, sin SMART.

---

## EXPOSITOR 2 — Metodología, modelado, métricas y comparación (7 min)

### Transición de entrada (15 s)

> "Gracias. Voy a mostrar cómo construimos y validamos el modelo, y por qué cada número que presentamos es reproducible."

### 1. Metodología CRISP-DM y preparación (90 s)

- Comprensión del negocio → EDA → preparación → modelado → evaluación → despliegue.
- 9 variables predictoras (`lead_time`, `age`, `scholarship`, `hipertension`, `diabetes`, `alcoholism`, `handcap`, `sms_received`, `historical_noshows`), objetivo binario.
- Split **estratificado 80/20 con semilla 42**: 88.421 entrenamiento / 22.106 prueba.

### 2. Benchmark de 4 modelos (2 min) — DEMO

Mostrar la **tabla del benchmark** (notebook, punto 14):

| Modelo | Recall | ROC-AUC |
|---|---|---|
| Línea base ingenua (todos asisten) | 0,00 | — |
| Regresión Logística | 0,5802 | 0,6738 |
| XGBoost | 0,7764 | 0,7286 |
| LightGBM | 0,8125 | 0,7353 |
| **Random Forest tuneado ⭐** | **0,8600** | **0,7352** |

- Criterio de selección: **Recall de la clase "no asistirá"** → Random Forest.
- Frase para la pregunta obvia: *"LightGBM empató en AUC (0,7353 vs 0,7351); la diferencia no es significativa y el criterio declarado fue el Recall, donde Random Forest fue superior tanto antes como después del tuning."*

### 3. Calibración del campeón (2 min)

- **RandomizedSearchCV**: 12 combinaciones, validación cruzada de 3 pliegues, criterio ROC-AUC → `n_estimators=200, max_depth=10, min_samples_split=10, min_samples_leaf=4, class_weight='balanced'`.
- **Threshold tuning**: de 44 umbrales que cumplían la meta se eligió **0,48** por mejor F1 → Recall sube de 0,8219 a **0,8600** (meta ≥ 0,85 cumplida).

### 4. Métricas verificadas en vivo (90 s) — DEMO

Mostrar la celda del punto 13 con la matriz de confusión:

- Test: **8.556 TN / 9.086 FP / 625 FN / 3.839 TP**.
- Lectura operativa: *"de cada 100 pacientes que van a faltar, detectamos 86; se nos escapan 14"*. Solo **625 de 4.464** ausentes reales no se detectaron.
- Trade-off declarado: Precision 0,297 → contactamos pacientes que al final asisten; cuesta S/ 0,80 de llamada, no una consulta de S/ 50.
- Verificación: las métricas recalculadas en el notebook **coinciden exactamente** con el artefacto desplegado `champion_model.pkl`.

### Cierre del bloque (30 s)

> "Resumen: el modelamiento supera a la regla de negocio actual en +86 puntos de captura y al baseline estadístico en +28 puntos de Recall. La pregunta siguiente es qué vale esto en soles. [nombre del Expositor 3]."

### ✂️ Versión abreviada: benchmark + matriz de confusión + una frase de tuning.

---

## EXPOSITOR 3 — Resultados de negocio, economía y conclusiones (6 min)

### Transición de entrada (15 s)

> "Gracias. Ya tenemos un modelo que detecta 86% de las inasistencias. Veamos cómo se convierte eso en citas recuperadas y soles."

### 1. El producto: app Streamlit (2 min) — DEMO (plan B: capturas)

- **Tablero de control:** agenda del día priorizada por riesgo (muestra de 50 citas: 8% alto, 56% medio, 36% bajo).
- **Evaluador individual:** probabilidad en tiempo real + mensaje de WhatsApp personalizado generado automáticamente.
- **Ciclo de retroalimentación:** registro post-consulta en `feedback_loop` → insumo para reentrenamientos.
- Regla operativa: Riesgo Alto (≥70%) = llamada telefónica · Medio (40–69%) = WhatsApp/SMS · Bajo (<40%) = correo.

### 2. Viabilidad económica — Montecarlo (2 min) — DEMO

Mostrar el **histograma del escenario base** (notebook, punto 15):

- Simulación **reproducible**: 10.000 iteraciones, semillas fijas, supuestos documentados en el código.
- Escenario base: **312 citas recuperadas/mes (P50)**, rango honesto **252–382 (P5–P95)**.
- Costo por 1.000 citas: de **S/ 10.094 a S/ 8.360**.
- Frase clave: *"cada sol invertido en contactación (≈ S/ 1.580/mes) evita entre S/ 7,7 y S/ 15 de pérdida según el escenario; ahorro anual estimado de S/ 145.000 a S/ 284.000"*. Incluso el pesimista se paga 7,7 veces.

### 3. Conclusiones y recomendaciones (90 s)

1. Meta predictiva **cumplida y verificada** (Recall 0,86, AUC 0,7352).
2. El **lead time** es la palanca de gestión principal.
3. La segmentación predictiva revierte la paradoja del SMS.
4. **Sostenible en todos los escenarios** (retorno mínimo 7,7:1).
5. Arquitectura **determinista, auditable y replicable** — cualquiera puede regenerar este informe ejecutando el notebook.
6. Recomendaciones: pilotar el scoring masivo diario, **piloto A/B** para calibrar efectividades con datos propios, reentrenamiento con `feedback_loop` y enriquecimiento de variables para superar el techo de AUC ≈ 0,735.

### Cierre final (15 s) — memorizar

> "El sistema demuestra que predecir la inasistencia no es un ejercicio académico: es recuperar más de 300 consultas al mes y hacer que cada sol invertido vuelva multiplicado. El informe completo, con cada cifra calculada en vivo, está en el repositorio. Muchas gracias — quedamos atentos a sus preguntas."

### ✂️ Versión abreviada: app con una captura + retorno 11:1 + cierre.

---

## Consejos generales para los tres

1. **Ensayar con cronómetro** al menos una vez completa; el riesgo #1 es pasarse de tiempo y que quiten el bloque de economía.
2. **Plan B de demo:** el HTML `informe_cpp2_ejecutable.html` tiene todas las salidas renderizadas — si falla internet o Supabase, se muestra el HTML y la exposición no se entera.
3. **No leer las diapositivas**: la tabla de benchmark y la matriz de confusión se explican con la mano, no con lectura.
4. **Cada expositor debe poder responder lo básico de los otros bloques** (ver `banco_preguntas.md` y `chuleta_cifras.md`).
5. Nombres de tablas y cifras exactas: nunca improvisar — están en la chuleta.

## Mapa de diapositivas sugerido (si arman PPT, 10 láminas)

1. Portada (título, equipo, curso)
2. El problema en 3 números: 110.527 citas · 20,19% ausentismo · 33% con cita lejana
3. Paradoja del SMS (por qué segmentar)
4. Arquitectura Medallion (diagrama Bronze→Silver→Gold)
5. Objetivo SMART y meta Recall ≥ 0,85
6. Benchmark de modelos (tabla del punto 14)
7. Matriz de confusión + trade-off Recall/Precision
8. App Streamlit (capturas de las 3 vistas)
9. Montecarlo: histograma + retorno 11:1
10. Conclusiones y recomendaciones
