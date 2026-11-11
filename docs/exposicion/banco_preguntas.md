# Banco de Preguntas de Evaluación — con respuestas preparadas

> Organizado por tema. Cada pregunta tiene: **Respuesta corta** (la que se memoriza) y **desarrollo** (si el profesor pide profundizar o es contra-pregunta).
> Marca ⚠️ = pregunta trampa frecuente. Al final: preguntas difíciles de alto riesgo.

---

## A. Sobre los datos

### A1. ¿Por qué ese dataset y qué garantías de calidad tiene?
**Corta:** Dataset histórico real de 110.527 citas del sistema público de Vitória (Brasil), con integridad perfecta: 0% de valores nulos, variables demográficas, clínicas y temporales, y la variable objetivo explícita.
**Desarrollo:** Se eligió por su volumen y porque representa exactamente el problema de negocio (agenda saturada + ausentismo). La verificación de nulos se hace en vivo en el notebook (punto 2). El CSV está versionado en `data/raw/` para que la replicación no dependa de descargas externas.

### A2. ⚠️ ¿El dataset está balanceado? ¿Qué implicó?
**Corta:** No — 20,19% de inasistencias. Por eso la exactitud es engañosa y usamos balanceo de clases y métricas robustas al desbalance.
**Desarrollo:** Un modelo ingenuo que diga "todos asisten" logra 79,8% de exactitud con 0% de captura de inasistentes (está en nuestra tabla de benchmark). Respuestas: `class_weight='balanced'` en todos los modelos, `scale_pos_weight` en XGBoost, y evaluar con Recall/F1/AUC en lugar de Accuracy.

### A7. ⚠️ "Sus citas son de 2016 — ¿cómo puede la app mostrar la agenda ACTUAL del centro?"
**Corta:** Con el modo operacional simulado (replay): la agenda de los próximos 5 días se construye con citas reales del histórico cuyas fechas se desplazan de forma coherente (la reserva queda en el pasado según el lead time real de cada cita), y se puntúan con el mismo job diario de scoring que usaría una clínica real.
**Desarrollo:** El script `src/09_simular_agenda_operativa.py` genera 200 citas pendientes (hoy → +4 días: 60/50/40/30/20) con horas de atención simuladas, las registra en `gold.score_input` —la tabla diseñada para "citas pendientes de evaluar"— y ejecuta el pipeline real (`src/04_predict_scoring.py`) que produce `gold.score_output`. La app lee esa agenda priorizada por día. La cadena es exactamente la que operaría en producción: solo cambiaría el origen de `score_input` (la agenda real del EMR en lugar del replay). Los nombres son sintéticos porque el dataset no contiene identificadores personales. En producción real el job correría cada madrugada sobre la agenda del día siguiente.

### A3. ⚠️ ¿Cómo calcularon el historial de ausentismo sin fuga de datos?
**Corta:** Acumulado temporal por paciente, restando siempre la fila actual: `cumsum() - valor_actual`, con orden por fecha de programación. La cita nunca "ve" su propio desenlace ni desenlaces futuros.
**Desarrollo:** Es la fuga de datos más común en este tipo de problema. Si calcularas el total histórico de ausencias del paciente incluyendo la cita actual, el modelo estaría "viendo la respuesta". Lo verificamos ordenando por `scheduled_day` dentro de cada `patient_id`. Sin este cuidado, el Recall sería artificialmente alto.

### A4. Si hay 0% de nulos, ¿para qué sirve la capa Silver?
**Corta:** Silver no es solo limpieza: es transformación — estandariza tipos, calcula `lead_time`, `grupo_edad` y `historical_noshows`, y deja los datos listos para ML sin duplicar lógica en los consumidores.
**Desarrollo:** La ausencia de nulos eliminó la necesidad de imputación, pero el *feature engineering* es el valor real de la capa. Además separa responsabilidades: Bronze nunca se toca (trazabilidad), Silver es el contrato de datos para Gold y para el modelo.

### A5. ¿Qué es el lead time y por qué es la variable más importante?
**Corta:** Días entre el agendamiento y la cita. Es el predictor dominante: 4,6% de ausentismo el mismo día vs. 33,0% con más de 30 días de anticipación.
**Desarrollo:** Tiene lectura causal directa: a más días de espera, más oportunidad de que la vida del paciente se interponga (mejora de salud, olvido, conflicto de horario). Es también la palanca operativa: la intensidad del recordatorio debería escalar con el lead time.

### A6. ⚠️ Según sus datos, quien recibe SMS falta MÁS (27,6% vs 16,7%). ¿Entonces los SMS son contraproducentes?
**Corta:** No — es confusión por una tercera variable: quienes tienen citas con más anticipación reciben más SMS, y las citas lejanas son las que más se ausentan. Correlación no es causalidad.
**Desarrollo:** El SMS actúa como *proxy* del lead time. La decisión de negocio correcta no es "enviar menos SMS" sino segmentar: el modelo usa ambas variables simultáneamente y prioriza la intervención humana en alto riesgo. Nuestra simulación asigna mayor efectividad a la llamada (22–30%) que al SMS (12–18%) por esto mismo. Si el profesor presiona: un experimento A/B con asignación aleatoria sería la única forma de medir causalidad — y de hecho lo recomendamos como trabajo futuro.

---

## B. Modelado y métricas

### B1. ¿Por qué compararon exactamente esos 4 modelos?
**Corta:** Regresión Logística como baseline lineal de referencia, más tres ensambles de árboles (Random Forest, XGBoost, LightGBM) que capturan no linealidades, bajo idéntico split estratificado y semilla para comparabilidad.
**Desarrollo:** La literatura del marco teórico (Hastie 2009; Breiman 2001; Topuz 2018) respalda comparar lineal vs. ensambles en datos tabulares de salud. Todos los modelos recibieron las mismas 9 features y balanceo de clases.

### B2. ⚠️ Si LightGBM tuvo el mayor ROC-AUC (0,7353), ¿por qué eligieron Random Forest?
**Corta:** Porque el criterio de selección declarado del proyecto fue el **Recall de la clase "no asistirá"** — meta SMART — y ahí Random Forest ganó (0,8219 vs 0,8125); la diferencia de AUC (0,0002) no es significativa.
**Desarrollo:** Después del tuning el orden se mantuvo (RF tuneado: Recall 0,86). Además, RF con profundidad limitada es muy estable ante sobreajuste y entrega importancias de variables interpretables para el comité de negocio. Si el profesor pregunta "¿y si hubieran tuneado LightGBM?": legítimo, sería un experimento razonable; el costo de cómputo era mayor y el margen ya era mínimo.

### B3. ¿Qué es RandomizedSearchCV y por qué no GridSearchCV?
**Corta:** Muestrea aleatoriamente 12 combinaciones del espacio de hiperparámetros con validación cruzada de 3 pliegues optimizando ROC-AUC; es mucho más barato que enumerar todas las combinaciones y suele encontrar configuraciones casi óptimas.
**Desarrollo:** GridSearch con la misma rejilla (n_estimators × max_depth × min_samples_split × min_samples_leaf) habría requerido cientos de ajustes. Resultado: `n_estimators=200, max_depth=10, min_samples_split=10, min_samples_leaf=4, class_weight='balanced'`.

### B4. ⚠️ ¿Por qué Recall y no Precision o F1 como métrica principal?
**Corta:** Por la asimetría de costos del negocio: un falso negativo (paciente que falta sin intervención) cuesta una consulta completa (S/ 50); un falso positivo cuesta una llamada (S/ 0,80). Minimizamos lo caro.
**Desarrollo:** La meta SMART se formuló como Recall ≥ 0,85. La Precision no se ignora: se documentó (0,297) y el umbral se eligió por mejor F1 entre los 44 que cumplían la meta. Es decir: Recall define la meta, F1 desempata los umbrales.

### B5. ⚠️ Una Precision de 0,297 significa que 7 de cada 10 contactos son "inútiles". ¿Es aceptable?
**Corta:** Sí — el contacto "inútil" cuesta S/ 0,80; el ausente no detectado cuesta S/ 50. Con 625 ausentes escapándose de 4.464, subir Precision sacrificando Recall destruiría valor.
**Desarrollo:** Matemáticamente: cada 1.000 citas hay ~202 ausentes; detectar 86% captura ~174 consultas en riesgo (≈ S/ 8.700 en valor de agenda recuperable) a un costo de contactación de ~S/ 158/mes por 1.000 citas. Además el FP no es "inútil": refuerza la confirmación de un paciente con perfil de riesgo.

### B6. ¿Por qué el umbral es 0,48 y no el clásico 0,50?
**Corta:** Threshold tuning: el 0,5 por defecto daba Recall 0,8219, por debajo de la meta; de los 44 umbrales que la cumplían, 0,48 maximiza el F1 (0,4415).
**Desarrollo:** La probabilidad del modelo es un *score*, no una decisión; el umbral es un hiperparámetro de negocio que se calibra según el costo relativo de FP vs FN. Bajarlo 0,02 elevó el Recall a 0,8600 a cambio de algo más de FPs.

### B7. ¿Un ROC-AUC de 0,735 es suficiente? Parece bajo.
**Corta:** Es el techo empírico de este dataset con 9 variables sin fuga de datos — consistente con la literatura del dominio. Superarlo exige más datos, no más algoritmo.
**Desarrollo:** El benchmark demuestra que 4 algoritmos distintos convergen a 0,728–0,735: el límite está en la información de los datos. Las variables que faltan y superarían el techo: distancia real al centro de salud, clima del día, historial de cancelaciones, variables socioeconómicas, día de la semana. Declarar el techo con evidencia es mejor método que prometer 0,9.

### B8. ¿Cómo controlaron el sobreajuste?
**Corta:** Hiperparámetros conservadores del ensamble (`max_depth=10`, `min_samples_leaf=4`), validación cruzada en el tuning, y evaluación final en 20% holdout que el modelo nunca vio.
**Desarrollo:** La señal de que no hay sobreajuste es que las métricas de test (Recall 0,86, AUC 0,7352) son reproducibles de forma determinista (semilla 42) y coinciden con el artefacto desplegado — lo verificamos en vivo en el punto 13 del notebook.

### B9. ¿Qué dice exactamente su matriz de confusión?
**Corta:** En las 22.106 citas de prueba: 8.556 verdaderos negativos, 9.086 falsos positivos, **625 falsos negativos** y 3.839 verdaderos positivos. De 4.464 ausentes reales, se detectan 3.839 (86%).
**Desarrollo:** En lenguaje del centro de salud: cada día, la lista de alto riesgo contiene casi 9 de cada 10 pacientes que realmente faltarán, al precio de contactar también pacientes con perfil de riesgo que asisten.

### B10. ¿Cómo están definidos los 3 niveles de riesgo?
**Corta:** Riesgo Alto: probabilidad ≥ 70% → llamada telefónica; Medio: 40–69% → WhatsApp/SMS; Bajo: < 40% → correo estándar.
**Desarrollo:** La intensidad de la intervención escala con el riesgo y con el costo: lo humano y caro se reserva para donde el modelo concentra la probabilidad de pérdida. En la agenda operativa simulada (200 citas pendientes de 5 días): 23 de riesgo alto (11.5%) y 177 de riesgo medio (88.5%) — al filtrar citas con lead time ≥ 5 días, la agenda pendiente concentra solo casos con acción preventiva real.

---

## C. Arquitectura y base de datos

### C1. ¿Por qué una arquitectura Medallion?
**Corta:** Separa responsabilidades: Bronze guarda el crudo inmutable (trazabilidad/auditoría), Silver estandariza y deriva features, Gold sirve datos de consumo (modelo estrella + MLOps). Cada capa tiene un contrato claro.
**Desarrollo:** Si un cálculo sale mal, se reprocesa desde Bronze sin re-ingresar; si cambia el modelo, Gold se reconstruye sin tocar Silver. Todo el pipeline está registrado en `meta.log_procesos` con volúmenes y marcas de tiempo.

### C2. ¿Por qué Supabase y no otro motor?
**Corta:** PostgreSQL gestionado en la nube, capa gratuita suficiente para 110.527 filas, RLS nativo y accesible desde Python y Streamlit — replicable por cualquier estudiante sin infraestructura propia.
**Desarrollo:** El DDL completo (`sql/schema_medallion.sql`) es idempotente: cualquiera con su propia instancia de Supabase ejecuta un script y tiene el esquema idéntico con políticas de seguridad.

### C3. ¿Qué es el modelo estrella y qué aporta?
**Corta:** Una tabla de hechos (`fact_citas`, 110.527) rodeada de dimensiones conformes (`dim_barrio` 81, `dim_paciente` 62.299, `dim_tiempo` 27) — el estándar para analítica porque hace las consultas de gestión simples y rápidas.
**Desarrollo:** Ejemplos: ausentismo por barrio, por día de la semana, por perfil etario — sin recalcular el pipeline ni agregaciones costosas sobre Silver. Las dimensiones se construyeron con claves subrogadas y `dim_paciente` guarda agregados históricos del paciente.

### C4. ¿Qué es RLS y por qué importa en salud?
**Corta:** Row Level Security: políticas de acceso a nivel de fila en PostgreSQL. Garantiza que una aplicación conectada con la clave `anon` solo vea lo que las políticas permiten, nunca el esquema completo.
**Desarrollo:** Para el MVP definimos RLS en las 11 tablas dentro del DDL. Datos de salud exigen minimizar exposición; además la clave secreta nunca sale del `.env` (fuera del repositorio) — el despliegue público usaría solo la clave `anon`.

### C5. ¿El scoring es en tiempo real o por lotes?
**Corta:** Ambos, según el caso de uso: scoring masivo de la agenda en lote (batch) hacia `score_output`, y el evaluador individual de la app responde en vivo (< 1 segundo) con el mismo artefacto.
**Desarrollo:** Es la arquitectura estándar en producción: el batch alimenta el tablero diario de admisión; la predicción individual atiende consultas puntuales del personal. `feedback_loop` cierra el ciclo registrando el desenlace real.

---

## D. Economía y simulación

### D1. ¿De dónde salen los supuestos económicos?
**Corta:** Todos documentados en el código: tasa de ausentismo como distribución Beta derivada del dataset real (media 20,19%), consulta S/ 50, tarifas de contacto (llamada S/ 0,80 · WhatsApp/SMS S/ 0,10 · correo S/ 0,02), efectividades por rango tomadas de la literatura de recordatorios en salud.
**Desarrollo:** La efectividad de la llamada (22–30%) es mayor que la del mensaje (12–18%) siguiendo la evidencia de intervención humana vs automatizada. Al ser supuestos, se modelaron como rangos y se reportan tres escenarios.

### D2. ¿Por qué Montecarlo y no un cálculo simple?
**Corta:** Porque los insumos son inciertos: Montecarlo propaga esa incertidumbre y entrega un rango (P5–P95), no un número falso-preciso. 10.000 iteraciones con semillas deterministas.
**Desarrollo:** El resultado base — 312 citas recuperadas/mes con rango 252–382 — es honesto: el centro de salud puede planificar con el P5 (peor caso razonable). Y es reproducible: mismo CSV `reports/simulacion_montecarlo_resumen.csv` en cualquier máquina.

### D3. ¿Qué resultado económico defendemos?
**Corta:** Escenario base: costo por 1.000 citas baja de S/ 10.094 a S/ 8.360; ahorro ≈ S/ 17.331/mes (≈ S/ 208.000/año); retorno de 11 soles por cada sol invertido. Rango completo: 7,7:1 (pesimista) a 15:1 (optimista).
**Desarrollo:** La contactación completa cuesta ≈ S/ 1.580/mes sobre 10.000 citas — el sistema se paga incluso en el peor escenario. El marco lógico (punto 9) usa exactamente estos indicadores verificables.

### D4. ⚠️ ¿No es muy optimista asumir que van a ejecutar todas las llamadas?
**Corta:** La simulación no asume 100% de ejecución: la efectividad (22–30% de las inasistencias del segmento alto) ya modela la fracción real que se logra recuperar, y el escenario pesimista baja ese rango a 15–22%.
**Desarrollo:** Aun así, el siguiente paso que proponemos es el piloto A/B con asignación aleatoria para sustituir la efectividad de literatura por la efectividad observada propia del centro — el modelo de simulación está construido para recalibrarse con esos datos.

---

## E. Despliegue, producción y ética

### E1. ¿Cómo llevarían esto a producción real?
**Corta:** Tres pasos: (1) app desplegada (Streamlit Cloud con clave `anon` + RLS), (2) scoring batch diario programado, (3) monitoreo de deriva y reentrenamiento periódico con `feedback_loop`.
**Desarrollo:** Para escala mayor: contenedorerizar la inferencia, mover el batch a un job scheduler, alertar cuando la tasa de no-show observada se desvíe del 20,19% histórico (señal de drift).

### E2. ¿Qué limitaciones reconocen?
**Corta:** Cuatro declaradas en el informe: sin variables externas (clima, tráfico, distancia real), scoring masivo en lote (no streaming), MVP sin integración con Historia Clínica Electrónica, y techo de AUC ≈ 0,735 con las variables actuales.
**Desarrollo:** La honestidad sobre límites es parte del método: cada límite está emparejado con una recomendación concreta (enriquecer features, API EMR, A/B de efectividades).

### E3. ¿Y la privacidad y el sesgo con pacientes vulnerables?
**Corta:** Los datos se manejan sin identificadores personales en los artefactos, con RLS y secretos fuera del repositorio. El sesgo por condición socioeconómica (Bolsa Família: 23,74% vs 19,81%) se usa para dar MÁS apoyo al grupo vulnerable, no para excluirlo — la intervención prioriza su asistencia.
**Desarrollo:** Un sistema así debe monitorearse para que las intervenciones no generen carga desproporcionada sobre ningún grupo; el registro en `feedback_loop` permitiría auditar la cobertura de contacto por segmento.

### E4. ¿Por qué Streamlit y no una aplicación web "seria"?
**Corta:** Streamlit es la elección correcta para un MVP: ciclos de desarrollo de horas, integración directa con scikit-learn y Supabase, y suficiente para validar la adopción por el personal antes de invertir en un frontend propio.
**Desarrollo:** La arquitectura separa modelo (artefacto .pkl) y datos (Supabase) de la interfaz: cambiar Streamlit por una API + React no toca nada del pipeline.

---

## F. Meta-preguntas (alto riesgo)

### F1. ⚠️ "¿Cómo sé que esas cifras no las inventaron?"
**Corta:** No lo crea por fe: el notebook `01_informe_cpp2_ejecutable.ipynb` recalcula todas las cifras en vivo — el EDA desde el CSV versionado, los conteos desde Supabase, el benchmark re-entrenado con la misma semilla, y el Montecarlo re-simulado. Ejecútelo y saldrá lo mismo.
**Desarrollo:** Además: semillas fijas en todos los scripts, `meta.log_procesos` audita cada corrida, y las métricas del notebook coinciden exactamente con el artefacto desplegado `champion_model.pkl` (verificado en el punto 13). La simulación económica es determinista: semillas 42+1/2/3.

### F2. ⚠️ "Si el profesor pide ejecutar algo ahora, ¿qué se ejecuta?"
**Plan acordado:** abrir el notebook ya ejecutado en Jupyter y correr "Restart & Run All" (~2 min) — o mostrar el HTML si no hay entorno. Si hay internet: mostrar el panel de Supabase y la app. Nunca prometer un reentrenamiento en vivo completo (toma minutos).

### F3. "¿Qué harían diferente con más tiempo?"
**Corta:** Tres cosas: piloto A/B para calibrar efectividades reales, enriquecer variables (distancia, clima, cancelaciones) para romper el techo de AUC, e integración con el EMR del centro.
**Desarrollo:** También: revisar la bandera `alerta_calidad_logica` de Silver como rutina de gobernanza y definir políticas de retención de datos para el Free Tier.

### F4. "¿Cuál fue el mayor error o dificultad del proyecto?"
**Respuesta honesta sugerida:** La fuga de datos potencial en `historical_noshows` — es el error clásico que infla métricas, y detectarlo y resolverlo con corte temporal estricto fue el aprendizaje metodológico principal. (Secundario: la paradoja del SMS nos obligó a distinguir correlación de causalidad en el análisis.)

### F5. "¿Por qué 6 semanas era plausible?"
**Corta:** Porque el alcance fue disciplinado: pipeline determinista de 7 scripts, un solo dataset, MVP funcional en lugar de producto, y simulación en lugar de experimento en vivo. Cada semana tenía entregable verificable (ver medios de verificación del marco lógico).

### F6. ⚠️ "El título del proyecto dice 'fidelización de pacientes' — ¿dónde está esa parte?"
**Corta:** En el ciclo de retroalimentación: la app registra tras cada consulta la duración real y la satisfacción (1–5 ⭐) en `gold.feedback_loop`, muestra los **KPIs del ciclo de fidelización** (satisfacción media, % de citas recuperadas por reprogramación, duración media) y el mensaje de WhatsApp ya incluye el canal de reprogramación.
**Desarrollo:** El piloto de encuestas (150 encuestas deterministas, semilla 42, **vinculadas a citas reales del dataset**: 120 de atenciones completadas y 30 de pacientes ausentes recuperados vía reprogramación) arroja satisfacción media **4,17/5** y **20% de citas recuperadas** — insumo declarado para los reentrenamientos. Si el profesor pregunta si son encuestas reales: honestidad — **son un piloto simulado para demostrar el ciclo**; el flujo de captura real se ejecuta desde la vista "Registro de Atención" y cualquier encuesta guardada ahí actualiza los KPIs al instante. Trabajo futuro: módulo de acciones proactivas de recuperación (lista de ausentes con contacto personalizado).

---

## Los 10 números que TODOS deben saber de memoria

| # | Dato | Valor |
|---|---|---|
| 1 | Citas del dataset | 110.527 |
| 2 | Tasa de ausentismo | 20,19% (22.319 citas) |
| 3 | Ausentismo por lead time | 4,6% mismo día → 33,0% con >30 días |
| 4 | Modelos comparados | 4 (LogReg, RF, XGBoost, LightGBM) |
| 5 | Campeón | Random Forest tuneado, umbral 0,48 |
| 6 | Meta SMART | Recall ≥ 0,85 → **logrado 0,86** |
| 7 | ROC-AUC | 0,7352 (baseline lineal: 0,6738) |
| 8 | Matriz de prueba | 3.839 TP · 625 FN (de 4.464 ausentes) |
| 9 | Montecarlo base | 312 citas/mes recuperadas (P50); rango 252–382 |
| 10 | Retorno económico | S/ 8.360 vs S/ 10.094 por 1.000 citas; 11:1 |
