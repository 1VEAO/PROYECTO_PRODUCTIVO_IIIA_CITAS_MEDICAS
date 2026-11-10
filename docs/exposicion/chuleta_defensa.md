# Chuleta de Defensa — imprimir o tener en el celular

## 1. Comandos para la demo (en orden de uso)

```bash
# PLAN A para 10 min: HTML con TODAS las salidas ya renderizadas
xdg-open informe_cpp2_ejecutable.html

# Solo si sobra tiempo al final: notebook ejecutable (cada celda corre en vivo)
jupyter notebook notebook/01_informe_cpp2_ejecutable.ipynb

# Mostrar la app (terminal aparte; requiere .env con credenciales)
streamlit run app.py

# Mostrar el resumen económico oficial (por si preguntan de dónde sale)
cat reports/simulacion_montecarlo_resumen.csv
```

**Momentos de demo por expositor:**
- **Exp. 1:** notebook → tabla inventario 11 tablas (punto 1) + gráfico lead time (punto 2)
- **Exp. 2:** notebook → tabla benchmark (punto 14) + matriz de confusión (punto 13)
- **Exp. 3:** app Streamlit (3 vistas) + histograma Montecarlo (punto 15)

## 2. Los números base (memorizar sí o sí)

| Dato | Valor |
|---|---|
| Dataset | 110.527 citas · 20,19% ausentismo · 0% nulos |
| Split | 80/20 estratificado, semilla 42 (88.421 / 22.106) |
| Features | 9 (lead_time, age, scholarship, hipertension, diabetes, alcoholism, handcap, sms_received, historical_noshows) |
| Campeón | Random Forest: n_est=200, depth=10, min_split=10, min_leaf=4, balanced |
| Umbral | 0,48 (de 44 umbrales válidos, mejor F1) |
| Recall | 0,8600 (meta 0,85 ✅) · Precision 0,297 · F1 0,4415 · AUC 0,7352 |
| Matriz test | TN 8.556 · FP 9.086 · FN 625 · TP 3.839 |
| Benchmark | LR 0,5802/0,6738 · RF 0,8219/0,7351 · XGB 0,7764/0,7286 · LGBM 0,8125/0,7353 |
| Lead time | 4,6 / 22,9 / 27,3 / 32,6 / 33,0 (%) |
| SMS paradoja | 27,57% con SMS vs 16,70% sin |
| Tablas | 11 · dim_paciente 62.299 · dim_barrio 81 · dim_tiempo 27 · fact_citas 110.527 |
| Muestra app (50 citas) | 8% alto · 56% medio · 36% bajo |
| Montecarlo | 10.000 iter, semilla 42 · Base P50=312 (P5–P95: 252–382) |
| Economía | S/ 10.094 → S/ 8.360 por 1.000 citas · ahorro S/ 17.331/mes (base) · retorno 7,7:1 a 15:1 (base 11:1) |
| Contactación | ≈ S/ 1.580/mes · consulta perdida S/ 50 · llamada S/ 0,80 · SMS S/ 0,10 |

## 3. Frases salvavidas (cuando la pregunta aprieta)

- **"No tenemos ese dato exacto, pero lo verificable es…"** → nunca inventar cifras; redirigir a una que se sepa.
- **"Es una decisión declarada en el informe: el criterio fue Recall"** → cuando cuestionen la elección de modelo/umbral.
- **"Es correlación, no causalidad — para causalidad proponemos el piloto A/B"** → ante la trampa del SMS.
- **"El techo de 0,735 es del dataset, no del algoritmo: cuatro modelos convergen ahí"** → ante "el AUC es bajo".
- **"Ese costo de 80 centimos es aceptable frente a una consulta de 50 soles perdida"** → ante "la precision es baja".
- **"Está en el notebook: se puede ejecutar en vivo y sale lo mismo"** → ante cualquier duda de veracidad.

## 4. Reparto de temas para la ronda de preguntas

- **Exp. 1 responde:** datos, calidad, arquitectura Medallion, Supabase, modelo estrella.
- **Exp. 2 responde:** modelado, hiperparámetros, métricas, umbral, fuga de datos, overfitting.
- **Exp. 3 responde:** economía, simulación, app, despliegue, limitaciones, ética.
- Regla: quien NO es dueño de la pregunta igual puede complementar — nunca dejar un silencio.

## 5. Checklist noche anterior

- [ ] Ensayo cronometrado completo (10 min estrictos)
- [ ] HTML abierto y listo (demo principal); notebook como respaldo
- [ ] `.venv` activado y `streamlit run app.py` probado hoy
- [ ] Proyector probado con el gráfico de lead time (ver contraste)
- [ ] Chuleta impresa / en el celular
- [ ] Cada expositor memorizó sus 10 números y su frase de apertura/cierre
