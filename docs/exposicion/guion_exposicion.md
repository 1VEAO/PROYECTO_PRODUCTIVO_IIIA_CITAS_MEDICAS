# Guion de Exposición (10 minutos) — Sistema Inteligente de Predicción de Inasistencias

> **Duración máxima: 10 min.** 3 expositores × ~3 min. El cronómetro manda: si un bloque se alarga, se recorta con las marcas 🔻, nunca improvisando.
> Regla de oro: **una idea por minuto, una cifra por idea.** El detalle profundo se defiende en la ronda de preguntas (ver `banco_preguntas.md`).
> Quien no exponga opera la demo (HTML/notebook/app) y apoya en las preguntas.

---

## Reparto y tiempos

| Expositor | Bloque | Tiempo |
|---|---|---|
| **Expositor 1** | Problema, datos y arquitectura | 3:00 |
| **Expositor 2** | Modelo, métricas y comparación | 3:30 |
| **Expositor 3** | Producto, economía y cierre | 3:00 |
| | Transiciones incluidas | **≈ 9:30** |

---

## EXPOSITOR 1 — Problema, datos y arquitectura (3:00)

### Apertura (20 s) — primera frase memorizada

> "Somos [nombres]. Nuestro sistema predice con días de anticipación qué pacientes no asistirán a su cita médica, para recuperar esas consultas antes de perderlas. Todas las cifras que veremos se calculan en vivo desde el código."

### El problema (80 s) — DEMO: gráfico de lead time

- Dataset real: **110.527 citas**, **20,19% de ausentismo** → 22.319 consultas perdidas: consultorios ociosos y listas de espera.
- Mostrar el gráfico de lead time: *"citar el mismo día casi nunca falla (4,6%); citar con más de 30 días falla un tercio de las veces (33,0%)"*.
- Paradoja del SMS en una frase: *"enviar recordatorios a todos no funciona — con SMS el ausentismo es 27,6% vs 16,7% sin ellos. Hay que segmentar con predicción."*

### Arquitectura (70 s) — DEMO: tabla de inventario de 11 tablas

- **Medallion en Supabase, un solo punto de verdad:** Bronze crudo inmutable (110.527) → Silver con *features* derivadas (`lead_time`, historial sin fuga de datos) → Gold con **modelo estrella** (`fact_citas` 110.527 + `dim_barrio`, `dim_paciente` 62.299, `dim_tiempo`) y tablas MLOps.
- Auditoría de cada corrida en `meta.log_procesos`. Frase puente:

> "Con la base lista, la pregunta era qué algoritmo predice mejor. [Expositor 2]."

### 🔻 Recorte de emergencia: dejar solo problema + lead time; arquitectura en una frase ("Bronze→Silver→Gold en Supabase, 11 tablas auditadas").

---

## EXPOSITOR 2 — Modelo, métricas y comparación (3:30)

### Transición (10 s)

> "Gracias. En 3 minutos: qué modelos probamos, qué ganó y por qué creerle a la métrica."

### Benchmark (90 s) — DEMO: tabla del punto 14

| Modelo | Recall | ROC-AUC |
|---|---|---|
| Línea base ingenua (todos asisten) | 0,00 | — |
| Regresión Logística | 0,5802 | 0,6738 |
| XGBoost / LightGBM | 0,7764 / 0,8125 | 0,7286 / 0,7353 |
| **Random Forest tuneado ⭐** | **0,8600** | **0,7352** |

- 4 algoritmos, mismo split estratificado 80/20, semilla 42. Criterio de selección: **Recall de la clase "no asistirá"** → ganó Random Forest, calibrado con RandomizedSearchCV.
- Contra la regla de negocio actual: **+86 puntos de captura**; contra el baseline estadístico: **+28**.

### Matriz de confusión (60 s) — DEMO: celda del punto 13

- Prueba de 22.106 citas: **3.839 ausentes detectados, solo 625 se escaparon de 4.464** → **de cada 100 que faltan, detectamos 86** (meta ≥ 0,85 cumplida con umbral calibrado 0,48).
- Trade-off con una frase: *"Precision 0,297: contactamos pacientes con perfil de riesgo que asisten — cuesta S/ 0,80 de llamada, no una consulta de S/ 50."*

### Verificabilidad (40 s)

- *"Estas métricas no están en una diapositiva: el notebook recalcula el pipeline con la misma semilla y coincide exactamente con el modelo desplegado. Lo pueden ejecutar."*

> Puente: "Detectar 86% de las ausencias está bien; la pregunta es qué vale en soles. [Expositor 3]."

### 🔻 Recorte de emergencia: solo tabla de benchmark + una cifra de la matriz (625 FN de 4.464).

---

## EXPOSITOR 3 — Producto, economía y cierre (3:00)

### Transición (10 s)

> "Gracias. El modelo ya está en un producto y su beneficio se puede cuantificar."

### El producto: app Streamlit (70 s) — DEMO rápida (una captura de respaldo lista)

- **Agenda pendiente en vivo:** 200 citas de los próximos 5 días priorizadas por el modelo — **23 de alto riesgo** para llamada prioritaria, 177 de riesgo medio para mensajería.
- **Agendamiento inteligente:** la recepción registra la cita, el modelo la puntúa al instante + **mensaje de WhatsApp** personalizado generado automáticamente.
- **Fidelización:** el registro post-consulta alimenta los **KPIs del ciclo** — piloto de 150 encuestas con satisfacción media **4,2 ⭐** y **20% de citas recuperadas** por reprogramación.
- Regla operativa: Alto (≥70%) llamada · Medio (40–69%) WhatsApp/SMS · Bajo (<40%) correo.

### Economía (80 s) — DEMO: histograma Montecarlo

- Simulación reproducible (10.000 iteraciones, semillas fijas, supuestos en el código).
- Escenario base: **312 citas recuperadas/mes** (P50; rango 252–382); costo por 1.000 citas baja de **S/ 10.094 a S/ 8.360**.
- Frase clave: *"cada sol invertido en contactación evita hasta 15 soles de pérdida; incluso en el peor escenario el retorno es 7,7 a 1."*

### Cierre final (40 s) — memorizado

> "En resumen: meta predictiva cumplida y verificada, arquitectura replicable de punta a punta, y sostenibilidad económica demostrada en todos los escenarios. El informe completo —con cada cifra calculada en vivo— está en el repositorio. Muchas gracias; quedamos atentos a sus preguntas."

### 🔻 Recorte de emergencia: una vista de la app + retorno 11:1 + cierre.

---

## Consejos para los 10 minutos

1. **Ensayar con cronómetro dos veces.** El error #1 en 10 min es llegar sin tiempo al cierre — la economía y el cierre son lo que el profesor recuerda.
2. **Demo mínima y blindada:** mostrar el **HTML ya renderizado** (`informe_cpp2_ejecutable.html`), no ejecutar el notebook en vivo (toma minutos). Solo si sobra tiempo al final, correr una celda.
3. **Nadie lee diapositivas:** las tablas se explican con la mano.
4. **Las preguntas no consumen los 10 min:** la profundidad va en la ronda de preguntas — está preparada en `banco_preguntas.md` (30+ preguntas con respuesta).
5. Los 3 expositores dominan los **10 números de la chuleta** — el profesor puede preguntar a cualquiera.

## Mapa de diapositivas sugerido (7 láminas para 10 min)

1. Portada (título, equipo, curso)
2. El problema en 3 números: 110.527 · 20,19% · 33% con cita lejana
3. Arquitectura Medallion (diagrama Bronze→Silver→Gold)
4. Benchmark de 4 modelos (tabla del punto 14)
5. Matriz de confusión + trade-off (625 FN, Recall 0,86)
6. App Streamlit (captura del tablero + WhatsApp)
7. Montecarlo y cierre: 312 citas/mes · retorno 11:1
