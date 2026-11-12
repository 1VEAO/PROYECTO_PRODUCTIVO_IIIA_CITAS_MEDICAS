import os
import sys
import pickle
import urllib.parse
import datetime
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================================
# 1. CONFIGURACIÓN DE PÁGINA Y ESTILOS UI
# ============================================================================
st.set_page_config(
    page_title="Gestión Inteligente de Citas Médicas",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS Limpios y Profesionales
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    
    .header-banner {
        background: linear-gradient(135deg, #0d6efd 0%, #0a58ca 100%);
        color: white;
        padding: 22px;
        border-radius: 12px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(13, 110, 253, 0.15);
    }
    
    .card-metric {
        background: #ffffff;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        border-left: 5px solid #0d6efd;
        text-align: left;
    }
    .card-metric-title {
        font-size: 13px;
        color: #6c757d;
        font-weight: 700;
        text-transform: uppercase;
    }
    .card-metric-val {
        font-size: 28px;
        font-weight: 800;
        color: #1e293b;
        margin-top: 4px;
    }
    
    .badge-status {
        padding: 14px;
        border-radius: 10px;
        color: white;
        font-weight: bold;
        text-align: center;
        font-size: 16px;
    }
    .bg-high { background: linear-gradient(135deg, #e03131 0%, #c92a2a 100%); }
    .bg-medium { background: linear-gradient(135deg, #f59f00 0%, #e67700 100%); }
    .bg-low { background: linear-gradient(135deg, #2f9e44 0%, #2b8a3e 100%); }

    .message-box {
        background-color: #e7f8e7;
        border-left: 5px solid #25d366;
        padding: 16px;
        border-radius: 8px;
        color: #111b21;
        font-size: 14px;
        margin-top: 10px;
        line-height: 1.6;
    }
    
    .info-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        margin-bottom: 15px;
    }

    .btn-whatsapp {
        background-color: #25D366;
        color: white !important;
        border: none;
        padding: 10px 18px;
        border-radius: 8px;
        font-weight: bold;
        cursor: pointer;
        display: inline-block;
        margin-top: 12px;
        text-decoration: none !important;
    }
    .btn-whatsapp:hover {
        background-color: #1eb954;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# Importar conexión a Supabase
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
try:
    from config import supabase
except Exception:
    supabase = None

# Cargar Modelo Predictivo
@st.cache_resource
def cargar_modelo():
    rutas = ["models/champion_model.pkl", "models/model.pkl"]
    for ruta in rutas:
        if os.path.exists(ruta):
            with open(ruta, "rb") as f:
                return pickle.load(f)
    return None

artefacto_ml = cargar_modelo()

# Estados para los Presets del Agendamiento Inteligente
if "sim_lead_time" not in st.session_state: st.session_state.sim_lead_time = 14
if "sim_age" not in st.session_state: st.session_state.sim_age = 28
if "sim_noshows" not in st.session_state: st.session_state.sim_noshows = 1
if "sim_sms" not in st.session_state: st.session_state.sim_sms = 0
if "sim_nombre" not in st.session_state: st.session_state.sim_nombre = "Ana María Silva"
if "sim_patient_id" not in st.session_state: st.session_state.sim_patient_id = 9988001
if "sim_appointment_id" not in st.session_state: st.session_state.sim_appointment_id = 5750001

# ============================================================================
# 2. NAVEGACIÓN Y MENÚ LATERAL (SIDEBAR)
# ============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/isometric/512/hospital.png", width=65)
    st.title("Asistente de Citas")
    st.caption("Plataforma de Gestión Preventiva")
    st.markdown("---")
    
    menu_opcion = st.radio(
        "📌 Menú Principal",
        [
            "📊 Resumen de la Agenda",
            "📅 Agendamiento Inteligente",
            "🗺️ Ubicación y Cobertura",
            "🔄 Registro de Atención",
            "ℹ️ Información y Ayuda"
        ]
    )
    
    st.markdown("---")
    st.caption("🟢 Supabase Conectado" if supabase else "⚪ Modo de Prueba Local")

# ============================================================================
# 3. CONTENIDO SEGÚN LA OPCIÓN SELECCIONADA
# ============================================================================

# ----------------------------------------------------------------------------
# OPCIÓN 1: RESUMEN DE LA AGENDA
# ----------------------------------------------------------------------------
if menu_opcion == "📊 Resumen de la Agenda":
    st.markdown("""
        <div class="header-banner">
            <h2 style='margin:0;'>📊 Agenda Operativa del Centro de Salud</h2>
            <p style='margin:5px 0 0 0; opacity:0.9;'>Citas pendientes de los próximos días, priorizadas por el modelo de riesgo de inasistencia</p>
        </div>
    """, unsafe_allow_html=True)
    
    cant_citas, cant_evaluadas = 110527, 50
    if supabase:
        try:
            r_silver = supabase.schema("silver").table("citas_cleaned").select("id", count="exact").execute()
            if r_silver.count: cant_citas = r_silver.count
            r_score = supabase.schema("gold").table("score_output").select("id", count="exact").execute()
            if r_score.count: cant_evaluadas = r_score.count
        except Exception:
            pass

    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="card-metric"><div class="card-metric-title">Total Citas Registradas</div><div class="card-metric-val">{cant_citas:,}</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="card-metric"><div class="card-metric-title">Citas Pendientes Evaluadas</div><div class="card-metric-val">{cant_evaluadas:,}</div></div>', unsafe_allow_html=True)

    # ── 📅 Agenda pendiente de los próximos días (score_output × score_input) ──
    df_agenda = pd.DataFrame()
    if supabase:
        try:
            r_out = supabase.schema("gold").table("score_output").select(
                "appointment_id, patient_id, probability_noshow, risk_level, recommended_action").execute()
            r_in = supabase.schema("gold").table("score_input").select(
                "appointment_id, appointment_day, age, lead_time").execute()
            df_out, df_in = pd.DataFrame(r_out.data), pd.DataFrame(r_in.data)
            if not df_out.empty and not df_in.empty:
                df_agenda = df_out.merge(df_in, on="appointment_id", how="left")
        except Exception:
            pass

    if not df_agenda.empty:
        dt_cita = pd.to_datetime(df_agenda["appointment_day"], utc=True)
        hoy = pd.Timestamp.now(tz="UTC").normalize()
        df_agenda["fecha"] = dt_cita.dt.date
        df_agenda["hora"] = dt_cita.dt.strftime("%H:%M")
        dias_diff = (dt_cita.dt.normalize() - hoy).dt.days
        df_agenda["etiqueta_dia"] = dias_diff.map(lambda n: "Hoy" if n == 0 else ("Mañana" if n == 1 else f"En {n} días"))
        # Nombres sintéticos deterministas (el dataset no contiene identificadores personales)
        NOMBRES = ["Ana María", "Carlos Eduardo", "Fernanda", "João Pedro", "Luiza", "Marcos Vinícius",
                   "Juliana", "Rafael", "Camila", "Bruno", "Patrícia", "André Luís", "Beatriz",
                   "Gabriel", "Larissa", "Thiago", "Vanessa", "Eduardo", "Renata", "Felipe"]
        APELLIDOS = ["Silva", "Santos", "Oliveira", "Souza", "Rodrigues", "Ferreira", "Almeida",
                     "Costa", "Gomes", "Martins", "Araújo", "Barbosa", "Ribeiro", "Carvalho",
                     "Lima", "Pereira", "Nogueira", "Moraes", "Cardoso", "Teixeira"]
        pid = df_agenda["patient_id"].astype("int64")
        df_agenda["paciente"] = [f"{NOMBRES[p % 20]} {APELLIDOS[(p // 13) % 20]}" for p in pid]
    else:
        dias_diff = pd.Series(dtype=int)

    n_alto_riesgo = int(df_agenda["risk_level"].str.contains("Alto", na=False).sum()) if not df_agenda.empty else 0
    c3.markdown(f'<div class="card-metric"><div class="card-metric-title">🔴 Alto Riesgo Pendiente</div><div class="card-metric-val">{n_alto_riesgo}</div></div>', unsafe_allow_html=True)

    if not df_agenda.empty:
        dias_disponibles = df_agenda[["etiqueta_dia", "fecha"]].drop_duplicates().sort_values("fecha")
        opciones = ["Todas las citas"] + [f"{r.etiqueta_dia} — {r.fecha:%a %d %b}" for r in dias_disponibles.itertuples()]
        sel_dia = st.selectbox("🗓️ Mostrar agenda de:", opciones)
        df_vista = df_agenda if sel_dia == "Todas las citas" else df_agenda[df_agenda["etiqueta_dia"] == sel_dia.split(" — ")[0]]

        df_tabla = (df_vista.sort_values(["fecha", "hora"])
                    .assign(probabilidad=lambda d: (d["probability_noshow"] * 100).round(1))
                    [["hora", "paciente", "age", "probabilidad", "risk_level", "recommended_action", "fecha"]]
                    .rename(columns={"hora": "Hora", "paciente": "Paciente", "age": "Edad",
                                     "probabilidad": "Prob. de Falta (%)", "risk_level": "Nivel de Riesgo",
                                     "recommended_action": "Acción Recomendada"}))
        st.dataframe(df_tabla, use_container_width=True, hide_index=True)
        st.caption(f"⚙️ **Modo operacional simulado (replay):** agenda de los próximos días construida con citas reales del histórico (fechas desplazadas de forma coherente), puntuada por el modelo campeón mediante el job diario de scoring (`src/04`). Los nombres son sintéticos: el dataset no contiene identificadores personales.")
    else:
        st.info("📭 No hay citas pendientes en `score_output`. Ejecuta `python src/09_simular_agenda_operativa.py --force` para construir la agenda operativa.")

    st.markdown("<br>", unsafe_allow_html=True)
    
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("##### 🍩 Distribución de Citas por Acción Recomendada")
        cant_alto, cant_medio, cant_bajo = 15, 25, 10
        if supabase:
            try:
                res_out = supabase.schema("gold").table("score_output").select("risk_level").execute()
                if res_out.data:
                    df_out = pd.DataFrame(res_out.data)
                    cant_alto = df_out['risk_level'].str.contains('Alto', na=False).sum()
                    cant_medio = df_out['risk_level'].str.contains('Medio', na=False).sum()
                    cant_bajo = df_out['risk_level'].str.contains('Bajo', na=False).sum()
            except Exception:
                pass

        df_pie = pd.DataFrame({
            "Acción Sugerida": ["📞 Llamada Telefónica", "💬 WhatsApp / SMS", "✉️ Correo Estándar"],
            "Cantidad": [int(cant_alto), int(cant_medio), int(cant_bajo)]
        })
        fig_pie = px.pie(df_pie, names="Acción Sugerida", values="Cantidad", hole=0.5,
                         color="Acción Sugerida",
                         color_discrete_map={"📞 Llamada Telefónica": "#e03131", "💬 WhatsApp / SMS": "#f59f00", "✉️ Correo Estándar": "#2f9e44"})
        fig_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': True})

    with g2:
        st.markdown("##### 📈 Tasa de Ausentismo según Días de Anticipación")
        
        # Dataset interactivo para filtrado
        df_lead = pd.DataFrame({
            "Anticipación": ["Mismo Día (0d)", "1 a 3 Días", "4 a 14 Días", "15 a 30 Días", "Más de 30 Días"],
            "Inasistencia (%)": [4.6, 22.9, 27.3, 32.6, 33.0]
        })
        
        # Filtro de rango de visión
        rangos_sel = st.multiselect("Filtrar intervalos de tiempo:", options=df_lead["Anticipación"].tolist(), default=df_lead["Anticipación"].tolist())
        df_lead_filt = df_lead[df_lead["Anticipación"].isin(rangos_sel)]
        
        if not df_lead_filt.empty:
            fig_bar = px.bar(df_lead_filt, x="Anticipación", y="Inasistencia (%)", color="Inasistencia (%)",
                             color_continuous_scale="Reds", text="Inasistencia (%)")
            fig_bar.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_bar.update_layout(margin=dict(t=30, b=10, l=10, r=10), yaxis=dict(range=[0, 40]))
            st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': True})
        else:
            st.info("Seleccione al menos un intervalo para visualizar los datos.")

# ----------------------------------------------------------------------------
# OPCIÓN 2: AGENDAMIENTO INTELIGENTE (REGISTRA + PUNTÚA AL INSTANTE)
# ----------------------------------------------------------------------------
elif menu_opcion == "📅 Agendamiento Inteligente":
    st.markdown("""
        <div class="header-banner">
            <h2 style='margin:0;'>📅 Agendamiento Inteligente</h2>
            <p style='margin:5px 0 0 0; opacity:0.9;'>Registra la cita del paciente, la puntúa al instante con el modelo campeón y la envía a la agenda priorizada</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("##### ⚡ Cargar perfiles de prueba rápida:")
    b1, b2, b3 = st.columns(3)
    if b1.button("🔴 Paciente con Alto Riesgo de Falta", use_container_width=True):
        st.session_state.sim_lead_time = 45; st.session_state.sim_age = 22; st.session_state.sim_noshows = 3; st.session_state.sim_sms = 0
        st.session_state.sim_nombre = "Carlos Eduardo Mendoza"
        st.session_state.sim_patient_id = 9988102; st.session_state.sim_appointment_id = 5750045
        st.rerun()
    if b2.button("🟡 Paciente con Riesgo Moderado", use_container_width=True):
        st.session_state.sim_lead_time = 12; st.session_state.sim_age = 35; st.session_state.sim_noshows = 1; st.session_state.sim_sms = 1
        st.session_state.sim_nombre = "Mariana Milagros López"
        st.session_state.sim_patient_id = 9988203; st.session_state.sim_appointment_id = 5750088
        st.rerun()
    if b3.button("🟢 Paciente Frecuente y Cumplidor", use_container_width=True):
        st.session_state.sim_lead_time = 0; st.session_state.sim_age = 65; st.session_state.sim_noshows = 0; st.session_state.sim_sms = 1
        st.session_state.sim_nombre = "Roberto Alarcón Gómez"
        st.session_state.sim_patient_id = 9988304; st.session_state.sim_appointment_id = 5750100
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    if artefacto_ml is None:
        st.error("❌ El motor predictivo no está disponible temporalmente.")
    else:
        with st.form("form_evaluador"):
            st.markdown("##### 👤 Identificación del Paciente y Datos de la Cita:")
            c_a, c_b, c_c = st.columns(3)
            
            with c_a:
                appointment_id = st.number_input("Número / Código de Cita", min_value=1000000, value=int(st.session_state.sim_appointment_id), step=1)
                patient_id = st.number_input("ID / Código del Paciente", min_value=1000000, value=int(st.session_state.sim_patient_id), step=1)
                nombre_paciente = st.text_input("Nombre Completo del Paciente", value=st.session_state.sim_nombre)
                telefono_paciente = st.text_input("Teléfono del Paciente", value="+51 987 654 321")

            with c_b:
                especialidad = st.selectbox("Especialidad Médica", ["Cardiología", "Medicina General", "Pediatría", "Ginecología", "Oftalmología", "Traumatología"])
                fecha_cita = st.date_input("Fecha Programada de la Cita", value=datetime.date.today() + datetime.timedelta(days=int(st.session_state.sim_lead_time)))
                hora_cita = st.time_input("Hora Programada", value=datetime.time(10, 30))
                barrio_paciente = st.selectbox("Barrio / Distrito", ["JARDIM DA PENHA", "MARIA ORTIZ", "RESISTÊNCIA", "JARDIM CAMBURI", "ITARARÉ", "SANTA MARTHA"])

            with c_c:
                age = st.slider("Edad del Paciente", 0, 100, int(st.session_state.sim_age))
                lead_time = st.slider("Días de Anticipación (Lead Time)", 0, 90, int(st.session_state.sim_lead_time))
                historical_noshows = st.number_input("Inasistencias Anteriores Registradas", min_value=0, value=int(st.session_state.sim_noshows), step=1)
                scholarship = st.selectbox("¿Apoyo Social (Programa)?", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")

            st.markdown("---")
            st.markdown("##### 🩺 Antecedentes Clínicos del Paciente:")
            d1, d2, d3, d4 = st.columns(4)
            with d1: hipertension = st.selectbox("Hipertensión", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
            with d2: diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
            with d3: alcoholism = st.selectbox("Alcoholismo", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
            with d4:
                handcap = st.selectbox("Discapacidad", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
                sms_received = st.selectbox("¿SMS Enviado?", [0, 1], index=int(st.session_state.sim_sms), format_func=lambda x: "Sí" if x == 1 else "No")

            btn_evaluar = st.form_submit_button("🚀 Agendar Cita y Evaluar Riesgo al Instante", use_container_width=True)

        if btn_evaluar:
            model = artefacto_ml["model"]
            features = artefacto_ml["features"]
            
            df_in = pd.DataFrame([{
                'lead_time': lead_time, 'age': age, 'scholarship': scholarship,
                'hipertension': hipertension, 'diabetes': diabetes, 'alcoholism': alcoholism,
                'handcap': handcap, 'sms_received': sms_received, 'historical_noshows': historical_noshows
            }])[features]
            
            prob_raw = model.predict_proba(df_in)
            if hasattr(prob_raw, "ndim") and prob_raw.ndim > 1:
                prob_noshow = float(prob_raw[0][1])
            else:
                prob_noshow = float(prob_raw[1]) if len(prob_raw) > 1 else float(prob_raw[0])
            
            st.markdown("---")
            st.markdown("#### 💡 Resultado y Recomendación del Sistema:")
            
            col_res1, col_res2 = st.columns([1, 1.2])
            
            tel_centro = "(027) 3333-7000"
            fecha_str = fecha_cita.strftime("%d/%m/%Y")
            hora_str = hora_cita.strftime("%I:%M %p")
            
            # Formato exacto requerido para el mensaje
            msg_txt = f"Estimado/a {nombre_paciente}, le recordamos su cita en la especialidad de {especialidad} programada para el {fecha_str} a las {hora_str} (en {lead_time} días). Para confirmar responda '1' a este WhatsApp o llame al {tel_centro}. Si requiere reprogramar o cancelar, por favor comuníquese al mismo número. ¡Le esperamos!"
            
            if prob_noshow >= 0.70:
                nivel_txt = "🔴 Prioridad Alta - Alto Riesgo de Inasistencia"
                nivel_db = "🔴 Riesgo Alto (≥ 70%)"
                accion_txt = f"📞 Realizar Llamada Telefónica Preventiva al paciente {nombre_paciente} ({telefono_paciente})"
                class_bg = "bg-high"
            elif prob_noshow >= 0.40:
                nivel_txt = "🟡 Prioridad Media - Riesgo Moderado"
                nivel_db = "🟡 Riesgo Medio (40%-69%)"
                accion_txt = f"💬 Enviar Recordatorio Interactivo de WhatsApp al paciente {nombre_paciente}"
                class_bg = "bg-medium"
            else:
                nivel_txt = "🟢 Prioridad Normal - Bajo Riesgo"
                nivel_db = "🟢 Riesgo Bajo (< 40%)"
                accion_txt = f"✉️ Enviar Recordatorio Estándar al paciente {nombre_paciente}"
                class_bg = "bg-low"

            # Enlace codificado para abrir directamente en WhatsApp
            msg_encoded = urllib.parse.quote(msg_txt)
            link_wa = f"https://wa.me/?text={msg_encoded}"

            with col_res1:
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob_noshow * 100,
                    number={'suffix': "%", 'font': {'size': 26}},
                    title={'text': "Nivel Estimado de Inasistencia", 'font': {'size': 15}},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#1e293b"},
                        'steps': [
                            {'range': [0, 40], 'color': "#d3f9d8"},
                            {'range': [40, 70], 'color': "#fff3bf"},
                            {'range': [70, 100], 'color': "#ffe3e3"}
                        ],
                    }
                ))
                fig_g.update_layout(height=260, margin=dict(t=50, b=20, l=30, r=30))
                st.plotly_chart(fig_g, use_container_width=True)

            with col_res2:
                st.markdown(f'<div class="badge-status {class_bg}">{nivel_txt}</div>', unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.info(f"**Acción Sugerida:** {accion_txt}")
                
                st.markdown(f'''
                    <div class="message-box">
                        <b>💬 Notificación Personalizada Generada para {nombre_paciente}:</b><br><br>
                        <i>"{msg_txt}"</i>
                        <br><br>
                        <a href="{link_wa}" target="_blank" class="btn-whatsapp">
                            🟢 Abrir en WhatsApp con mensaje listo
                        </a>
                    </div>
                ''', unsafe_allow_html=True)

            if supabase:
                try:
                    # 1) La cita evaluada entra como pendiente en score_input
                    #    (mismo contrato que el pipeline batch) → aparece en la agenda
                    fecha_reserva = fecha_cita - datetime.timedelta(days=int(lead_time))
                    input_data = {
                        "appointment_id": int(appointment_id), "patient_id": int(patient_id),
                        "scheduled_day": fecha_reserva.isoformat() + "T09:00:00Z",
                        "appointment_day": fecha_cita.isoformat() + "T" + hora_cita.strftime("%H:%M:%S") + "Z",
                        "lead_time": int(lead_time), "age": int(age),
                        "scholarship": int(scholarship), "hipertension": int(hipertension),
                        "diabetes": int(diabetes), "alcoholism": int(alcoholism),
                        "handcap": int(handcap), "sms_received": int(sms_received),
                        "historical_noshows": int(historical_noshows)
                    }
                    supabase.schema("gold").table("score_input").insert(input_data).execute()

                    # 2) El resultado queda en score_output con el mismo vocabulario
                    #    del pipeline ("Riesgo Alto/Medio/Bajo") para el tablero
                    score_data = {
                        "appointment_id": int(appointment_id), "patient_id": int(patient_id),
                        "probability_noshow": round(float(prob_noshow), 4),
                        "risk_level": nivel_db, "recommended_action": accion_txt
                    }
                    supabase.schema("gold").table("score_output").insert(score_data).execute()
                    st.success("✅ Evaluación registrada en `gold.score_input` y `gold.score_output` — la cita ya aparece en la Agenda Pendiente.")
                except Exception as e:
                    st.error(f"⚠️ No se pudo registrar la evaluación en la base de datos: {e}")

# ----------------------------------------------------------------------------
# OPCIÓN 3: UBICACIÓN Y COBERTURA (DATOS GEOGRÁFICOS BRASIL)
# ----------------------------------------------------------------------------
elif menu_opcion == "🗺️ Ubicación y Cobertura":
    st.markdown("""
        <div class="header-banner">
            <h2 style='margin:0;'>🗺️ Cobertura Geográfica del Servicio</h2>
            <p style='margin:5px 0 0 0; opacity:0.9;'>Análisis espacial interactivo del origen de los pacientes en Vitória, Espírito Santo (Brasil)</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Dataset Base Geográfico
    df_barrios_base = pd.DataFrame({
        "Barrio / Distrito": ["JARDIM DA PENHA", "MARIA ORTIZ", "RESISTÊNCIA", "JARDIM CAMBURI", "ITARARÉ", "SANTA MARTHA", "CENTRO", "TABUZEIRO"],
        "Total Citas": [4351, 3805, 3050, 2871, 2514, 2110, 1845, 1620],
        "Asistencia Promedio (%)": [83.7, 79.0, 79.5, 81.0, 78.2, 80.1, 82.4, 77.9]
    })
    
    # Filtros Interactivos
    st.markdown("##### 🔍 Filtros Interactivos del Territorio:")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        barrios_sel = st.multiselect("Seleccionar Barrios a Inspeccionar:", options=df_barrios_base["Barrio / Distrito"].tolist(), default=df_barrios_base["Barrio / Distrito"].tolist()[:6])
    with col_f2:
        asistencia_min = st.slider("Filtrar por Asistencia Promedio Mínima (%):", 70.0, 90.0, 75.0, step=0.5)

    # Aplicación de Filtros
    df_filtrado = df_barrios_base[
        (df_barrios_base["Barrio / Distrito"].isin(barrios_sel)) & 
        (df_barrios_base["Asistencia Promedio (%)"] >= asistencia_min)
    ]

    st.markdown("---")
    col_m1, col_m2 = st.columns([1.4, 1])
    
    with col_m1:
        st.markdown("##### 📍 Gráfico Interactivo de Citas por Barrio:")
        if not df_filtrado.empty:
            fig_barrios = px.bar(
                df_filtrado, x="Total Citas", y="Barrio / Distrito", orientation='h',
                color="Asistencia Promedio (%)", color_continuous_scale="Blues",
                text="Total Citas",
                title="Volumen de Citas Médicas por Barrio Filtrado"
            )
            fig_barrios.update_traces(textposition='outside')
            fig_barrios.update_layout(
                yaxis=dict(autorange="reversed"),
                margin=dict(t=50, b=40, l=120, r=40),
                height=380
            )
            st.plotly_chart(fig_barrios, use_container_width=True, config={'displayModeBar': True})
        else:
            st.warning("⚠️ No hay barrios que coincidan con los filtros seleccionados.")

    with col_m2:
        st.markdown("##### 🇧🇷 Ficha Geográfica y Resultados Filtrados:")
        st.dataframe(df_filtrado, use_container_width=True)
        
        st.markdown("""
            <div class="info-box">
                <b>🏙️ Ciudad:</b> Vitória | <b>Estado:</b> Espírito Santo (Brasil)<br>
                <b>🏥 Citas en Dataset:</b> 110,527 Registros<br>
                💡 <i>Usa la barra de herramientas sobre el gráfico para hacer Zoom, Panorámica o descargar la imagen en PNG.</i>
            </div>
        """, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# OPCIÓN 4: REGISTRO DE ATENCIÓN (POST-CONSULTA)
# ----------------------------------------------------------------------------
elif menu_opcion == "🔄 Registro de Atención":
    st.markdown("""
        <div class="header-banner">
            <h2 style='margin:0;'>🔄 Registro Post-Consulta</h2>
            <p style='margin:5px 0 0 0; opacity:0.9;'>Registra la duración real y la satisfacción del paciente, y monitorea los KPIs del ciclo de fidelización</p>
        </div>
    """, unsafe_allow_html=True)

    # ── KPIs del Ciclo de Fidelización (leídos de gold.feedback_loop) ──
    st.markdown("##### 💙 KPIs del Ciclo de Fidelización")
    df_fb = pd.DataFrame()
    if supabase:
        try:
            r_fb = supabase.schema("gold").table("feedback_loop").select("*").execute()
            df_fb = pd.DataFrame(r_fb.data)
        except Exception:
            pass

    if not df_fb.empty:
        sat_media = df_fb["satisfaction_score"].mean()
        dur_media = df_fb["actual_duration_min"].mean()
        pct_recuperadas = df_fb["rescheduled_status"].str.contains("Reprogramada", na=False).mean() * 100

        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(f'<div class="card-metric"><div class="card-metric-title">Encuestas Registradas</div><div class="card-metric-val">{len(df_fb):,}</div></div>', unsafe_allow_html=True)
        k2.markdown(f'<div class="card-metric"><div class="card-metric-title">Satisfacción Media</div><div class="card-metric-val">{sat_media:.2f} ⭐</div></div>', unsafe_allow_html=True)
        k3.markdown(f'<div class="card-metric"><div class="card-metric-title">Citas Recuperadas</div><div class="card-metric-val">{pct_recuperadas:.1f}%</div></div>', unsafe_allow_html=True)
        k4.markdown(f'<div class="card-metric"><div class="card-metric-title">Duración Media</div><div class="card-metric-val">{dur_media:.1f} min</div></div>', unsafe_allow_html=True)

        f1, f2 = st.columns(2)
        with f1:
            st.markdown("##### ⭐ Distribución de Calificaciones")
            df_sat = df_fb["satisfaction_score"].value_counts().sort_index().reset_index()
            df_sat.columns = ["Calificación", "Encuestas"]
            fig_sat = px.bar(df_sat, x="Calificación", y="Encuestas", color="Encuestas",
                             color_continuous_scale="Blues", text="Encuestas")
            fig_sat.update_layout(margin=dict(t=10, b=10, l=10, r=10), xaxis=dict(dtick=1))
            st.plotly_chart(fig_sat, use_container_width=True, config={'displayModeBar': False})
        with f2:
            st.markdown("##### 🎯 Estado Final de las Citas Encuestadas")
            df_estado = df_fb["rescheduled_status"].value_counts().reset_index()
            df_estado.columns = ["Estado Final", "Encuestas"]
            fig_est = px.pie(df_estado, names="Estado Final", values="Encuestas", hole=0.5)
            fig_est.update_layout(margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_est, use_container_width=True, config={'displayModeBar': False})
    elif supabase:
        st.info("📭 Aún no hay encuestas en el ciclo de retroalimentación. La primera aparecerá aquí al guardar el formulario de abajo.")
    else:
        st.info("⚪ Modo local: los KPIs de fidelización se muestran al conectar Supabase.")

    with st.form("form_registro_atencion"):
        st.markdown("##### 📋 Datos de la Cita Atendida:")
        c_r1, c_r2 = st.columns(2)
        
        with c_r1:
            nombre_pac_atendido = st.text_input("Nombre Completo del Paciente", value="Ana María Silva")
            patient_id_atendido = st.number_input("ID / Código del Paciente", min_value=1000000, value=9988001, step=1)
            fb_appointment_id = st.number_input("Número / Código de Cita Médica", min_value=1000000, value=5750001, step=1)
            especialidad_atendida = st.selectbox("Especialidad Atendida", ["Cardiología", "Medicina General", "Pediatría", "Ginecología", "Oftalmología", "Traumatología"])

        with c_r2:
            actual_duration = st.slider("Duración Real de la Consulta (Minutos)", 5, 60, 20)
            satisfaction_score = st.select_slider("Calificación de Satisfacción del Paciente", options=[1, 2, 3, 4, 5], value=5, format_func=lambda x: f"{x} {'⭐' * x}")
            rescheduled_status = st.selectbox("Estado Final de la Cita", [
                "Atención Completada Normal",
                "Reprogramada Mismo Día",
                "Reprogramada Otra Fecha",
                "Cancelada por el Paciente"
            ])

        btn_guardar_rec = st.form_submit_button("💾 Guardar Registro de Atención en el Sistema", use_container_width=True)

    if btn_guardar_rec:
        if supabase:
            try:
                fb_data = {
                    "appointment_id": fb_appointment_id,
                    "actual_duration_min": actual_duration,
                    "satisfaction_score": satisfaction_score,
                    "rescheduled_status": rescheduled_status
                }
                supabase.schema("gold").table("feedback_loop").insert(fb_data).execute()
                st.balloons()
                st.success(f"🎉 Atención registrada con éxito para {nombre_pac_atendido} (Cita #{fb_appointment_id} - {especialidad_atendida}).")
            except Exception as e:
                st.error(f"Error al guardar registro: {e}")
        else:
            st.success(f"🎉 Atención registrada localmente para {nombre_pac_atendido} (Cita #{fb_appointment_id}).")

# ----------------------------------------------------------------------------
# OPCIÓN 5: INFORMACIÓN INSTITUCIONAL, BUENAS PRÁCTICAS Y AYUDA
# ----------------------------------------------------------------------------
elif menu_opcion == "ℹ️ Información y Ayuda":
    st.markdown("""
        <div class="header-banner">
            <h2 style='margin:0;'>ℹ️ Ficha del Proyecto y Guía de Uso</h2>
            <p style='margin:5px 0 0 0; opacity:0.9;'>Información académica, equipo de desarrollo y buenas prácticas de uso</p>
        </div>
    """, unsafe_allow_html=True)
    
    col_i1, col_i2 = st.columns(2)
    
    with col_i1:
        st.markdown("##### 🎓 Datos Académicos e Institucionales")
        st.markdown("""
            <div class="info-box">
                <b>🏫 Instituto:</b> Instituto Continental<br>
                <b>📚 Curso:</b> Proyecto Productivo IIIA<br>
                <b>💼 Carrera:</b> Ciencia de Datos e Inteligencia Artificial<br>
                <b>👨‍🏫 Docente del Curso:</b> Orizano Salvador Sergio Víctor<br>
                <b>📅 Año Académico:</b> 2026
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("##### 👥 Equipo de Desarrollo:")
        st.markdown("""
            <div class="info-box">
                1. <b>Alarcón Olmedo Víctor Efraín</b><br>
                2. <b>Inuma Macedo Mariana Milagros</b><br>
                3. <b>Méndez Macedo Karen Yaseli</b><br>
                4. <b>Salinas Llana Jhim Anthony</b>
            </div>
        """, unsafe_allow_html=True)

    with col_i2:
        st.markdown("##### 💡 Buenas Prácticas de Uso")
        st.markdown("""
            <div class="info-box">
                1. <b>Atención Prioritaria:</b> Para citas clasificadas como 🔴 <b>Prioridad Alta</b>, realice una llamada telefónica preventiva desde admisión al menos 24 horas antes.<br><br>
                2. <b>Notificación Automática:</b> Presione el botón verde <i>"🟢 Abrir en WhatsApp"</i> para enviar el recordatorio personalizado con nombre, fecha y hora exacta.<br><br>
                3. <b>Feedback Continuo:</b> Al finalizar la consulta, complete la sección <i>"Registro de Atención"</i> para alimentar la Capa Gold y reentrenar la IA.
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("##### ❓ Preguntas Frecuentes (FAQ)")
        with st.expander("¿Para qué sirve esta plataforma?"):
            st.write("Ayuda al personal de admisión a predecir qué pacientes tienen mayor probabilidad de faltar a su cita médica, recomendando la mejor acción preventiva para asegurar su asistencia.")
        with st.expander("¿Quiénes deben usar esta aplicación?"):
            st.write("Está diseñada para el personal administrativo, recepcionistas y equipo médico de cualquier centro de salud.")