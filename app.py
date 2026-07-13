"""
============================================================
 Tech Challenge – Fase 04 | POSTECH Data Analytics
 Aplicação Streamlit: Sistema Preditivo + Dashboard Analítico
 de Obesidade
============================================================
 Para rodar:
   pip install streamlit pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn plotly
   streamlit run app.py
============================================================
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────
# CONFIGURAÇÃO DA PÁGINA
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Predição de Obesidade | POSTECH",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────
# PALETA VERDE PASTEL
# ──────────────────────────────────────────────────────────
# Escala única de verde: do mais claro (peso insuficiente) ao mais escuro (obesidade III)
COLOR_MAP = {
    "Insufficient_Weight":  "#c8e6c9",  # verde muito claro
    "Normal_Weight":        "#a5d6a7",  # verde claro
    "Overweight_Level_I":   "#81c784",  # verde médio-claro
    "Overweight_Level_II":  "#66bb6a",  # verde médio
    "Obesity_Type_I":       "#43a047",  # verde médio-escuro
    "Obesity_Type_II":      "#2e7d32",  # verde escuro
    "Obesity_Type_III":     "#1b5e20",  # verde muito escuro
}

COLOR_MAP_DASH = {
    "Insufficient_Weight":  "#81c784",  # verde
    "Normal_Weight":        "#aed581",  # verde claro
    "Overweight_Level_I":   "#fff176",  # amarelo
    "Overweight_Level_II":  "#ffb74d",  # laranja
    "Obesity_Type_I":       "#ff8a65",  # laranja avermelhado
    "Obesity_Type_II":      "#ef5350",  # vermelho
    "Obesity_Type_III":     "#c62828",  # vermelho escuro
}

COLOR_MAP_STEPS = {
            "Insufficient_Weight": "#acd6ae",
            "Normal_Weight":       "#c8df7a",
            "Overweight_Level_I":  "#fff38a",
            "Overweight_Level_II": "#ffc96b",
            "Obesity_Type_I":      "#ffa285",
            "Obesity_Type_II":     "#ec8482",
            "Obesity_Type_III":    "#db6666",
}


COLOR_F = "#FD3DB5"
COLOR_M = '#0000FF'
#Cor de destaque para gênero e outros bináries — ainda dentro do verde
# COLOR_F  = "#b2dfdb"   # verde-água feminino
# COLOR_M  = "#80cbc4"   # verde-água masculino



# ──────────────────────────────────────────────────────────
# ESTILOS CUSTOMIZADOS
# ──────────────────────────────────────────────────────────
st.markdown("""
<style>
.main { background-color: #f1f8f2; }

.metric-card {
    background: white;
    border-radius: 12px;
    padding: 20px 24px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    text-align: center;
    border-top: 4px solid #66bb6a;
}
.metric-card h2 { margin: 0; font-size: 2rem; color: #2e7d32; }
.metric-card p  { margin: 4px 0 0; color: #5f6368; font-size: 0.9rem; }

.pred-box {
    border-radius: 14px;
    padding: 28px 32px;
    text-align: center;
    margin: 16px 0;
    background: #e8f5e9;
    color: #1b5e20;
}
.pred-box h1 { font-size: 2.2rem; margin: 0; }
.pred-box p  { font-size: 1rem; margin: 6px 0 0; opacity: 0.85; }

[data-testid="stSidebar"] { background-color: #1b5e20; }
[data-testid="stSidebar"] * { color: white !important; }

.section-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #2e7d32;
    border-left: 4px solid #66bb6a;
    padding-left: 12px;
    margin-bottom: 16px;
}

.insight-box {
    background: #f1f8f2;
    border-left: 4px solid #66bb6a;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 10px 0;
    color: #1b5e20;
    font-size: 0.92rem;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
# CONSTANTES
# ──────────────────────────────────────────────────────────
ORDER = [
    "Insufficient_Weight", "Normal_Weight",
    "Overweight_Level_I", "Overweight_Level_II",
    "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III",
]

LABEL_PT = {
    "Insufficient_Weight":  "Abaixo do Peso",
    "Normal_Weight":        "Peso Normal",
    "Overweight_Level_I":   "Sobrepeso Grau I",
    "Overweight_Level_II":  "Sobrepeso Grau II",
    "Obesity_Type_I":       "Obesidade Tipo I",
    "Obesity_Type_II":      "Obesidade Tipo II",
    "Obesity_Type_III":     "Obesidade Tipo III",
}

RECOMMENDATIONS = {
    "Insufficient_Weight": [
        "Consultar nutricionista para plano de ganho de peso saudável.",
        "Aumentar ingestão calórica com alimentos nutritivos.",
        "Avaliar causas subjacentes (distúrbios alimentares, hipertireoidismo).",
        "Praticar musculação para ganho de massa magra.",
    ],
    "Normal_Weight": [
        "Manter hábitos atuais de alimentação e atividade física.",
        "Realizar check-up anual preventivo.",
        "Monitorar IMC periodicamente.",
        "Continuar com hidratação adequada (≥ 2L/dia).",
    ],
    "Overweight_Level_I": [
        "Reduzir consumo de alimentos ultraprocessados e calóricos.",
        "Iniciar prática regular de atividade física (3×/semana).",
        "Monitorar ingestão calórica diária.",
        "Consultar médico para avaliação cardiometabólica.",
    ],
    "Overweight_Level_II": [
        "Intervenção nutricional urgente com acompanhamento profissional.",
        "Aumentar frequência de atividade física para ≥ 4×/semana.",
        "Reduzir consumo de álcool e alimentos calóricos.",
        "Avaliar risco de síndrome metabólica, hipertensão e diabetes.",
    ],
    "Obesity_Type_I": [
        "Encaminhar para equipe multidisciplinar (médico, nutricionista, psicólogo).",
        "Programa estruturado de perda de peso (déficit calórico controlado).",
        "Exames: glicemia, perfil lipídico, pressão arterial.",
        "Considerar atividade física de baixo impacto (natação, caminhada).",
    ],
    "Obesity_Type_II": [
        "Avaliação médica imediata para comorbidades associadas.",
        "Considerar tratamento farmacológico adjuvante.",
        "Programa intensivo de mudança de estilo de vida.",
        "Monitoramento contínuo de pressão arterial, glicemia e função hepática.",
    ],
    "Obesity_Type_III": [
        "Avaliação para cirurgia bariátrica (IMC ≥ 40).",
        "Acompanhamento médico intensivo e multidisciplinar.",
        "Rastreamento de apneia do sono, diabetes tipo 2, doenças cardiovasculares.",
        "Suporte psicológico e grupos de apoio.",
    ],
}


def fmt_num(n: int) -> str:
    """Formata inteiro com separador de milhar por ponto (padrão brasileiro)."""
    return f"{n:,}".replace(",", ".")


# ──────────────────────────────────────────────────────────
# CARREGAMENTO DE DADOS E MODELO
# ──────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    paths = ["Obesity.csv", "data/Obesity.csv",
             os.path.join(os.path.dirname(__file__), "Obesity.csv")]
    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p)
    st.error("❌ Arquivo Obesity.csv não encontrado. Coloque-o na mesma pasta do app.py.")
    st.stop()


@st.cache_resource
def load_model():
    model_dir = "model"
    required = ["best_model.pkl", "scaler.pkl", "label_encoder.pkl", "metadata.pkl"]
    if all(os.path.exists(os.path.join(model_dir, f)) for f in required):
        with open(f"{model_dir}/best_model.pkl",    "rb") as f: model  = pickle.load(f)
        with open(f"{model_dir}/scaler.pkl",        "rb") as f: scaler = pickle.load(f)
        with open(f"{model_dir}/label_encoder.pkl", "rb") as f: le     = pickle.load(f)
        with open(f"{model_dir}/metadata.pkl",      "rb") as f: meta   = pickle.load(f)
        return model, scaler, le, meta
    else:
        return _train_fallback_model()


def _train_fallback_model(): # O prefixo _ indica que essa é uma função de uso interno do programa  (Uma convenão em Python, não uma regra)
    import lightgbm as lgb
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    df = load_data()
    dw = df.copy()
    for col in ["FCVC", "NCP", "CH2O", "FAF", "TUE"]:
        dw[col] = dw[col].round().astype(int)
    dw["BMI"] = dw["Weight"] / (dw["Height"] ** 2) # cria coluna BMI (IMC)
    dw["risk_score"] = ( #Essa coluna é uma engenharia de atributos (feature engineering).Ela cria uma pontuação baseada em hábitos considerados de risco.
        (dw["FAVC"] == "yes").astype(int) * 2 +
        (dw["family_history"] == "yes").astype(int) * 2 +
        (dw["SMOKE"] == "yes").astype(int) +
        (dw["SCC"] == "no").astype(int) +
        (dw["CAEC"] == "Always").astype(int) * 2 +
        (dw["CAEC"] == "Frequently").astype(int) +
        (dw["CALC"] == "Always").astype(int) * 2 +
        (dw["CALC"] == "Frequently").astype(int)
    )
    dw["healthy_score"] = ( # conta hábitos saudáveis
        (dw["FAF"] >= 2).astype(int) * 2 +
        (dw["FCVC"] >= 2).astype(int) +
        (dw["CH2O"] >= 2).astype(int) +
        (dw["SCC"] == "yes").astype(int) +
        dw["MTRANS"].isin(["Walking", "Bike"]).astype(int)
    )
    for col in ["Gender", "family_history", "FAVC", "SMOKE", "SCC"]: # Transforma texto em números
        dw[col] = dw[col].map({"Male": 1, "Female": 0, "yes": 1, "no": 0}).fillna(0).astype(int)
    dw["CAEC"] = dw["CAEC"].map({"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3})
    dw["CALC"] = dw["CALC"].map({"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3})
    dw = pd.get_dummies(dw, columns=["MTRANS"], drop_first=False)

    label_order = [ # define as ordem das classses
        "Insufficient_Weight", "Normal_Weight", "Overweight_Level_I",
        "Overweight_Level_II", "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III",
    ]
    le = LabelEncoder()
    le.fit(label_order) # Aprende a correspondência ex: insullficient: 0, Normal:1...
    X = dw.drop(columns=["Obesity"])
    y = le.transform(dw["Obesity"])
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, random_state=42, verbose=-1)
    model.fit(X_sc, y)
    meta = {
        "best_model_name": "LightGBM (auto-treinado)",
        "feature_columns": list(X.columns),
        "scaled": True,
        "accuracy": None,
    }
    return model, scaler, le, meta

# Essa função prepara os dados de entrada exatamente da mesma forma que foi feito durante o treinamento do modelo.
def engineer_features(input_df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    df = input_df.copy()
    for col in ["FCVC", "NCP", "CH2O", "FAF", "TUE"]:
        if col in df.columns:
            df[col] = df[col].round().astype(int)
    df["BMI"] = df["Weight"] / (df["Height"] ** 2)
    df["risk_score"] = (
        (df["FAVC"] == "yes").astype(int) * 2 +
        (df["family_history"] == "yes").astype(int) * 2 +
        (df["SMOKE"] == "yes").astype(int) +
        (df["SCC"] == "no").astype(int) +
        (df["CAEC"] == "Always").astype(int) * 2 +
        (df["CAEC"] == "Frequently").astype(int) +
        (df["CALC"] == "Always").astype(int) * 2 +
        (df["CALC"] == "Frequently").astype(int)
    )
    df["healthy_score"] = (
        (df["FAF"] >= 2).astype(int) * 2 +
        (df["FCVC"] >= 2).astype(int) +
        (df["CH2O"] >= 2).astype(int) +
        (df["SCC"] == "yes").astype(int) +
        df["MTRANS"].isin(["Walking", "Bike"]).astype(int)
    )
    for col in ["Gender", "family_history", "FAVC", "SMOKE", "SCC"]:
        df[col] = df[col].map({"Male": 1, "Female": 0, "yes": 1, "no": 0}).fillna(0).astype(int)
    df["CAEC"] = df["CAEC"].map({"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3})
    df["CALC"] = df["CALC"].map({"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3})
    df = pd.get_dummies(df, columns=["MTRANS"], drop_first=False)
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    return df[feature_columns]


def predict_obesity(inputs: dict, model, scaler, le, meta) -> tuple:
    input_df = pd.DataFrame([inputs])
    X = engineer_features(input_df, meta["feature_columns"])
    X_proc = scaler.transform(X) if meta.get("scaled", False) else X.values
    pred_idx = model.predict(X_proc)[0]
    pred_class = le.inverse_transform([pred_idx])[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba_arr = model.predict_proba(X_proc)[0]
        proba = {le.inverse_transform([i])[0]: float(p) for i, p in enumerate(proba_arr)}
    return pred_class, proba


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Converte cor hex para string rgba válida para o Plotly."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ──────────────────────────────────────────────────────────
# INICIALIZAÇÃO
# ──────────────────────────────────────────────────────────
df_raw = load_data()
model, scaler, le, meta = load_model()

df_dash = df_raw.copy()
df_dash["BMI"] = df_dash["Weight"] / (df_dash["Height"] ** 2)
df_dash["Obesity_PT"] = df_dash["Obesity"].map(LABEL_PT)


# ──────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 Predição de Obesidade")
    st.markdown("**POSTECH – Tech Challenge Fase 04**")
    st.markdown("---")
    page = st.radio(
        "Navegação",
        ["🔮 Sistema Preditivo", "📊 Dashboard Analítico"],
        index=0,
    )
    st.markdown("---")
    st.markdown(f"**Modelo:** {meta.get('best_model_name', 'LightGBM')}")
    if meta.get("accuracy"):
        st.markdown(f"**Acurácia:** {meta['accuracy']*100:.2f}%")
    st.markdown(f"**Base de dados:** {fmt_num(len(df_raw))} registros")
    st.markdown("---")
    st.markdown("*Desenvolvido para apoiar a equipe médica na tomada de decisão clínica.*")


# ══════════════════════════════════════════════════════════
# PÁGINA 1 – SISTEMA PREDITIVO
# ══════════════════════════════════════════════════════════
if page == "🔮 Sistema Preditivo":

    st.markdown("# 🔮 Sistema Preditivo de Obesidade")
    st.markdown(
        "Preencha os dados do paciente abaixo para obter a predição do nível de obesidade "
        "e recomendações clínicas personalizadas."
    )
    st.markdown("---")
    

    with st.form("prediction_form"):
        st.markdown('<div class="section-title">👤 Dados Antropométricos e Demográficos</div>',
                    unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        gender  = c1.selectbox("Gênero", ["Female", "Male"],
                                format_func=lambda x: "Feminino" if x == "Female" else "Masculino")
        age     = c2.number_input("Idade (anos)", min_value=10, max_value=100, value=25, step=1)
        height  = c3.number_input("Altura (m)", min_value=1.40, max_value=2.20,
                                   value=1.70, step=0.01, format="%.2f")
        c4, c5 = st.columns(2)
        weight  = c4.number_input("Peso (kg)", min_value=30.0, max_value=250.0,
                                   value=70.0, step=0.5, format="%.1f")
        bmi_preview = weight / (height ** 2)
        delta_label = "Normal" if 18.5 <= bmi_preview < 25 else ("Abaixo do esperado" if bmi_preview < 18.5 else "Acima do esperado")
        c5.metric("IMC Calculado", f"{bmi_preview:.1f} kg/m²", delta=delta_label)

        st.markdown("---")
        st.markdown('<div class="section-title">🍽️ Hábitos Alimentares</div>',
                    unsafe_allow_html=True)
        c6, c7, c8 = st.columns(3)
        favc = c6.selectbox("Consome alimentos calóricos frequentemente?",
                             ["yes", "no"],
                             format_func=lambda x: "Sim" if x == "yes" else "Não")
        fcvc = c7.slider("Frequência de consumo de vegetais (1=Raramente / 3=Sempre)", 1, 3, 2)
        ncp  = c8.slider("Número de refeições principais por dia", 1, 4, 3)
        c9, c10 = st.columns(2)
        caec = c9.selectbox("Come entre as refeições?",
                             ["no", "Sometimes", "Frequently", "Always"],
                             format_func=lambda x: {
                                 "no": "Não", "Sometimes": "Às vezes",
                                 "Frequently": "Frequentemente", "Always": "Sempre"
                             }[x])
        ch2o = c10.slider("Consumo diário de água (1=Menos de 1L / 2=Entre 1–2L / 3=Mais de 2L)", 1, 3, 2)

        st.markdown("---")
        st.markdown('<div class="section-title">🏃 Estilo de Vida</div>',
                    unsafe_allow_html=True)
        c11, c12, c13 = st.columns(3)
        faf   = c11.slider("Frequência de atividade física por semana (0=Nenhuma / 3=Diária)", 0, 3, 1)
        tue   = c12.slider("Tempo em dispositivos eletrônicos (0=Até 2h / 1=3 a 5h / 2=Mais de 5h)", 0, 2, 1)
        smoke = c13.selectbox("Fuma?", ["no", "yes"],
                               format_func=lambda x: "Não" if x == "no" else "Sim")
        c14, c15, c16 = st.columns(3)
        scc   = c14.selectbox("Monitora calorias ingeridas?", ["no", "yes"],
                               format_func=lambda x: "Não" if x == "no" else "Sim")
        calc  = c15.selectbox("Frequência de consumo de álcool",
                               ["no", "Sometimes", "Frequently", "Always"],
                               format_func=lambda x: {
                                   "no": "Não consome", "Sometimes": "Às vezes",
                                   "Frequently": "Frequentemente", "Always": "Sempre"
                               }[x])
        mtrans = c16.selectbox("Meio de transporte habitual",
                                ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"],
                                format_func=lambda x: {
                                    "Public_Transportation": "Transporte Público",
                                    "Walking": "A pé", "Automobile": "Automóvel",
                                    "Motorbike": "Moto", "Bike": "Bicicleta"
                                }[x])

        st.markdown("---")
        st.markdown('<div class="section-title">🧬 Histórico de Saúde</div>',
                    unsafe_allow_html=True)
        family_history = st.selectbox(
            "Algum familiar tem ou teve excesso de peso?", ["no", "yes"],
            format_func=lambda x: "Não" if x == "no" else "Sim"
        )

        submitted = st.form_submit_button("🔮 Realizar Predição", use_container_width=True)

    # ── RESULTADO ──────────────────────────────────────
    if submitted:
        inputs = dict(
            Gender=gender, Age=age, Height=height, Weight=weight,
            family_history=family_history, FAVC=favc, FCVC=fcvc,
            NCP=ncp, CAEC=caec, SMOKE=smoke, CH2O=ch2o, SCC=scc,
            FAF=faf, TUE=tue, CALC=calc, MTRANS=mtrans,
        )
        bmi_calc = weight / (height ** 2)

        with st.spinner("Analisando dados do paciente..."):
            pred_class, proba = predict_obesity(inputs, model, scaler, le, meta)

        cor_fundo = COLOR_MAP_STEPS[pred_class]
        cor_letra = COLOR_MAP_DASH[pred_class]
        st.markdown(f"""
        <div class="pred-box" style="background:{cor_fundo}22; border: 2px solid {cor_fundo}; color:{cor_letra};">
            <h1>🩺 {LABEL_PT[pred_class]}</h1>
            <p>Diagnóstico preditivo baseado nos dados informados</p>
        </div>
        """, unsafe_allow_html=True)

        # Probabilidades
        if proba:
            st.markdown("#### 📊 Probabilidade por Classe")
            prob_df = pd.DataFrame([
                {"Classe": LABEL_PT[k], "Probabilidade": v}
                for k, v in sorted(proba.items(), key=lambda x: ORDER.index(x[0]))
            ])
            color_list = [COLOR_MAP[k] for k in sorted(proba.keys(), key=lambda x: ORDER.index(x))]
            fig_prob = px.bar(
                prob_df, x="Probabilidade", y="Classe", orientation="h",
                color="Classe",
                color_discrete_sequence=color_list,
                text=prob_df["Probabilidade"].apply(lambda v: f"{v*100:.1f}%"),
                height=320,
            )
            fig_prob.update_layout(
                showlegend=False,
                margin=dict(l=10, r=10, t=10, b=10),
                yaxis=dict(categoryorder="array",
                           categoryarray=[LABEL_PT[c] for c in ORDER]),
                #plot_bgcolor="#f1f8f2",
                #paper_bgcolor="#f1f8f2",
            )
            fig_prob.update_traces(textposition="outside")
            st.plotly_chart(fig_prob, use_container_width=True)

        col_g1, col_g2 = st.columns(2)

        COLOR_MAP_STEPS = {
            "Insufficient_Weight": "#b7ddb8",
            "Normal_Weight":       "#d4e68a",
            "Overweight_Level_I":  "#fff59d",
            "Overweight_Level_II": "#ffd180",
            "Obesity_Type_I":      "#ffab91",
            "Obesity_Type_II":     "#ef9a9a",
            "Obesity_Type_III":    "#ef9a9a",
        }

        # Barra (mais escura)
        COLOR_MAP_BAR = {
            "Insufficient_Weight": "#388e3c",
            "Normal_Weight":       "#689f38",
            "Overweight_Level_I":  "#fbc02d",
            "Overweight_Level_II": "#f57c00",
            "Obesity_Type_I":      "#e64a19",
            "Obesity_Type_II":     "#c62828",
            "Obesity_Type_III":    "#8e0000",
        }
        # Gauge IMC
        with col_g1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(bmi_calc, 1),
                title={"text": "IMC do Paciente"},
                gauge={
                    "axis": {"range": [10, 50]},
                    "bar": {"color": COLOR_MAP_BAR[pred_class]},
                    "steps": [
                        {"range": [10, 18.5], "color": "#acd6ae"},
                        {"range": [18.5, 25],  "color": "#c8df7a"},
                        {"range": [25, 30],    "color": "#fff38a"},
                        {"range": [30, 35],    "color": "#ffc96b"},
                        {"range": [35, 40],    "color": "#ffa285"},
                        {"range": [40, 50],    "color": "#ec8482"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 3},
                        "thickness": 0.9,
                        "value": bmi_calc,
                    },
                },
                number={"suffix": " kg/m²", "font": {"size": 28}},
            ))
            fig_gauge.update_layout(
                height=260,
                margin=dict(l=20, r=20, t=40, b=10),
                #paper_bgcolor="#f1f8f2",
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        # Radar de perfil — fillcolor convertido corretamente via função helper
        with col_g2:
            risk = (
                (favc == "yes") * 2 +
                (family_history == "yes") * 2 +
                (smoke == "yes") +
                (scc == "no") +
                (caec == "Always") * 2 +
                (caec == "Frequently") +
                (calc == "Always") * 2 +
                (calc == "Frequently")
            )
            healthy = (
                (faf >= 2) * 2 +
                (fcvc >= 2) +
                (ch2o >= 2) +
                (scc == "yes") +
                (mtrans in ["Walking", "Bike"])
            )
            categories = [
                "Risco<br>Comportamental", "Hábitos<br>Saudáveis",
                "Ativ.<br>Física", "Hidratação", "Dieta",
            ]
            values = [
                min(risk / 10, 1),
                min(healthy / 6, 1),
                faf / 3,
                (ch2o - 1) / 2,
                (fcvc - 1) / 2,
            ]
            line_color   = COLOR_MAP[pred_class]
            fill_color   = hex_to_rgba(COLOR_MAP[pred_class], 0.35)

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill="toself",
                fillcolor=fill_color,
                line=dict(color=line_color, width=2),
                name="Paciente",
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=False,
                height=260,
                margin=dict(l=40, r=40, t=40, b=20),
                title="Perfil do Paciente",
                #paper_bgcolor="#f1f8f2",
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # Recomendações clínicas
        st.markdown("---")
        st.markdown("#### 💊 Recomendações Clínicas")
        for rec in RECOMMENDATIONS[pred_class]:
            st.markdown(
                f'<div class="insight-box">▶ {rec}</div>',
                unsafe_allow_html=True,
            )

        # Resumo dos dados
        with st.expander("📋 Resumo dos Dados Informados"):
            resumo = pd.DataFrame({
                "Campo": [
                    "Gênero", "Idade", "Altura", "Peso", "IMC",
                    "Histórico familiar", "Alimentos calóricos", "Consumo de vegetais",
                    "Refeições por dia", "Come entre refeições", "Fumante",
                    "Água por dia", "Monitora calorias", "Atividade física",
                    "Eletrônicos", "Álcool", "Transporte",
                ],
                "Valor": [
                    "Feminino" if gender == "Female" else "Masculino",
                    f"{age} anos", f"{height:.2f} m", f"{weight:.1f} kg",
                    f"{bmi_calc:.1f}",
                    "Sim" if family_history == "yes" else "Não",
                    "Sim" if favc == "yes" else "Não",
                    f"{fcvc}/3", f"{ncp}",
                    {"no": "Não", "Sometimes": "Às vezes",
                     "Frequently": "Frequentemente", "Always": "Sempre"}[caec],
                    "Sim" if smoke == "yes" else "Não",
                    f"{ch2o}/3",
                    "Sim" if scc == "yes" else "Não",
                    f"{faf}/3", f"{tue}/2",
                    {"no": "Não consome", "Sometimes": "Às vezes",
                     "Frequently": "Frequentemente", "Always": "Sempre"}[calc],
                    {"Public_Transportation": "Transporte Público",
                     "Walking": "A pé", "Automobile": "Automóvel",
                     "Motorbike": "Moto", "Bike": "Bicicleta"}[mtrans],
                ]
            })
            st.dataframe(resumo, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════
# PÁGINA 2 – DASHBOARD ANALÍTICO
# ══════════════════════════════════════════════════════════
else:
    st.markdown("# 📊 Dashboard Analítico – Obesidade")
    st.markdown(
        "Visão analítica dos dados para apoio à equipe médica na compreensão dos "
        "fatores associados à obesidade."
    )
    st.markdown("---")

    # ── KPIs ──────────────────────────────────────────
    total     = len(df_dash)
    obesos    = df_dash["Obesity"].isin(["Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"]).sum()
    sobrepeso = df_dash["Obesity"].isin(["Overweight_Level_I", "Overweight_Level_II"]).sum()
    normal    = df_dash["Obesity"].isin(["Normal_Weight"]).sum()
    imc_medio = df_dash["BMI"].mean()

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.markdown(f'<div class="metric-card"><h2>{fmt_num(total)}</h2><p>Total de Pacientes</p></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="metric-card"><h2>{fmt_num(obesos)}</h2><p>Com Obesidade ({obesos/total*100:.0f}%)</p></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="metric-card"><h2>{fmt_num(sobrepeso)}</h2><p>Sobrepeso ({sobrepeso/total*100:.0f}%)</p></div>', unsafe_allow_html=True)
    k4.markdown(f'<div class="metric-card"><h2>{fmt_num(normal)}</h2><p>Peso Normal ({normal/total*100:.0f}%)</p></div>', unsafe_allow_html=True)
    k5.markdown(f'<div class="metric-card"><h2>{imc_medio:.1f}</h2><p>IMC Médio (kg/m²)</p></div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Filtros ────────────────────────────────────────
    with st.expander("🔧 Filtros", expanded=False):
        col_f1, col_f2, col_f3 = st.columns(3)
        gen_filter = col_f1.multiselect(
            "Gênero", ["Female", "Male"], default=["Female", "Male"],
            format_func=lambda x: "Feminino" if x == "Female" else "Masculino"
        )
        age_filter = col_f2.slider("Faixa etária", 14, 61, (14, 61))
        cls_filter = col_f3.multiselect(
            "Nível de obesidade", ORDER, default=ORDER,
            format_func=lambda x: LABEL_PT[x]
        )

    df_f = df_dash[
        df_dash["Gender"].isin(gen_filter) &
        df_dash["Age"].between(*age_filter) &
        df_dash["Obesity"].isin(cls_filter)
    ]
    st.markdown(f"*{fmt_num(len(df_f))} registros após filtros aplicados*")
    st.markdown("---")

    LAYOUT_BASE = dict(
        plot_bgcolor="#f1f8f2",
        paper_bgcolor="#081709",
        font=dict(color="#1b5e20"),
    )

    # ── ROW 1: Distribuição ──────────────────────────
    st.markdown('<div class="section-title">Distribuição e Perfil Demográfico</div>',
                unsafe_allow_html=True)
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        dist = df_f["Obesity"].value_counts().reindex(ORDER).fillna(0).reset_index()
        dist.columns = ["Obesity", "Quantidade"]
        dist["Nível"] = dist["Obesity"].map(LABEL_PT)
        fig1 = px.bar(
            dist, x="Nível", y="Quantidade",
            color="Obesity", color_discrete_map=COLOR_MAP,
            text="Quantidade",
            title="Distribuição por Nível de Obesidade",
            labels={"Nível": "", "Quantidade": "Pacientes"},
        )
        fig1.update_traces(textposition="outside")
        fig1.update_layout(showlegend=False, margin=dict(t=40, b=60))
        fig1.update_xaxes(tickangle=-30)
        fig1.update_yaxes(showgrid=False, showticklabels = False)
        st.plotly_chart(fig1, use_container_width=True)

    with r1c2:
        gd = df_f.groupby(["Obesity", "Gender"]).size().reset_index(name="Quantidade")
        gd["Nível"] = gd["Obesity"].map(LABEL_PT)
        gd["Gênero"] = gd["Gender"].map({"Female": "Feminino", "Male": "Masculino"})
        fig2 = px.bar(
            gd, x="Nível", y="Quantidade", color="Gênero",
            barmode="group",
            title="Distribuição por Gênero e Nível de Obesidade",
            labels={"Nível": "", "Quantidade": "Pacientes"},
            color_discrete_map={"Feminino": COLOR_F, "Masculino": COLOR_M},
        )
        fig2.update_layout(margin=dict(t=40, b=60))
        fig2.update_xaxes(tickangle=-30)
        st.plotly_chart(fig2, use_container_width=True)

    # ── ROW 2: IMC ──────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-title">IMC e Idade por Nível de Obesidade</div>',
                unsafe_allow_html=True)
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        bmi_box = df_f.copy()
        bmi_box["Nível"] = bmi_box["Obesity"].map(LABEL_PT)
        fig3 = px.box(
            bmi_box, x="Nível", y="BMI",
            color="Obesity", color_discrete_map=COLOR_MAP_DASH,
            title="Distribuição do IMC por Nível de Obesidade",
            labels={"Nível": "", "BMI": "IMC (kg/m²)"},
            category_orders={"Nível": [LABEL_PT[o] for o in ORDER]},
        )
        fig3.update_layout(showlegend=False, margin=dict(t=40, b=60))
        fig3.update_xaxes(tickangle=-30)
        st.plotly_chart(fig3, use_container_width=True)

    with r2c2:
        sc_df = df_f.copy()
        sc_df["Nível"] = sc_df["Obesity"].map(LABEL_PT)
        fig4 = px.scatter(
            sc_df.sample(min(600, len(sc_df)), random_state=42),
            x="Age", y="BMI",
            color="Obesity", color_discrete_map=COLOR_MAP_DASH,
            opacity=0.65,
            title="Relação Idade × IMC",
            labels={"Age": "Idade (anos)", "BMI": "IMC"},
            hover_data=["Weight", "Height"],
        )
        fig4.update_layout(legend_title="Nível", margin=dict(t=40))
        st.plotly_chart(fig4, use_container_width=True)

    # ── ROW 3: Fatores de risco ───────────────────────
    st.markdown("---")
    st.markdown('<div class="section-title">Fatores de Risco e Hábitos de Vida</div>',
                unsafe_allow_html=True)
    r3c1, r3c2, r3c3 = st.columns(3)

    with r3c1:
        fp = (
            df_f.groupby("Obesity")["family_history"]
            .apply(lambda x: (x == "yes").mean() * 100)
            .reindex(ORDER).reset_index()
        )
        fp.columns = ["Obesity", "Percentual"]
        fp["Nível"] = fp["Obesity"].map(LABEL_PT)
        fig5 = px.bar(
            fp, x="Nível", y="Percentual",
            color="Obesity", color_discrete_map=COLOR_MAP,
            title="% com Histórico Familiar",
            labels={"Nível": "", "Percentual": ""},
            text=fp["Percentual"].round(1).astype(str) + "%",
        )
        fig5.update_layout(showlegend=False, margin=dict(t=40, b=60))
        fig5.update_xaxes(tickangle=-30)
        fig5.update_yaxes(showgrid= False, showticklabels = False)
        st.plotly_chart(fig5, use_container_width=True)

    with r3c2:
        fm = df_f.groupby("Obesity")["FAF"].mean().reindex(ORDER).reset_index()
        fm["Nível"] = fm["Obesity"].map(LABEL_PT)
        fig6 = px.bar(
            fm, x="Nível", y="FAF",
            color="Obesity", color_discrete_map=COLOR_MAP,
            title="Frequência Média de Atividade Física",
            labels={"Nível": "", "FAF": "FAF médio (0–3)"},
            text=fm["FAF"].round(2),
        )
        fig6.update_traces(textposition="outside")
        fig6.update_layout(showlegend=False, margin=dict(t=40, b=60))
        fig6.update_xaxes(tickangle=-30)
        fig6.update_yaxes(showgrid = False, showticklabels = False)
        st.plotly_chart(fig6, use_container_width=True)

    with r3c3:
        fc = (
            df_f.groupby("Obesity")["FAVC"]
            .apply(lambda x: (x == "yes").mean() * 100)
            .reindex(ORDER).reset_index()
        )
        fc.columns = ["Obesity", "Percentual"]
        fc["Nível"] = fc["Obesity"].map(LABEL_PT)
        fig7 = px.bar(
            fc, x="Nível", y="Percentual",
            color="Obesity", color_discrete_map=COLOR_MAP,
            title="% que Consome Alimentos Calóricos",
            labels={"Nível": "", "Percentual": ""},
            text=fc["Percentual"].round(1).astype(str) + "%",
        )
        fig7.update_layout(showlegend=False, margin=dict(t=40, b=60))
        fig7.update_xaxes(tickangle=-30)
        fig7.update_yaxes(showgrid = False, showticklabels = False)
        st.plotly_chart(fig7, use_container_width=True)

    # ── ROW 4: Transporte e correlação ───────────────
    st.markdown("---")
    st.markdown('<div class="section-title">Transporte e Correlação entre Variáveis</div>',
                unsafe_allow_html=True)
    r4c1, r4c2 = st.columns(2)

    with r4c1:
        mt = df_f[df_f["Obesity"].str.contains("Obesity")].copy()

        mt["Transporte"] = mt["MTRANS"].map({
            "Public_Transportation": "Transp. Público",
            "Walking": "A pé",
            "Automobile": "Automóvel",
            "Motorbike": "Moto",
            "Bike": "Bicicleta",
        })

        fig8 = px.histogram(
            mt,
            x="Transporte",
            text_auto=True,
            title="Obesidade por tipos de transporte",
        )

        # Define todas as barras como verdes
        fig8.update_traces(
            marker_color="green"
        )

        fig8.update_layout(
            xaxis_title="",
            yaxis_title="",
        )
        fig8.update_layout(margin=dict(t=50))
        fig8.update_yaxes(showgrid = False, showticklabels = False)
        
        st.plotly_chart(fig8, use_container_width=True)
       
    with r4c2:
        num_cols = ["Age", "BMI", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
        corr_df  = df_f[num_cols].corr().round(2)
        fig9 = px.imshow(
            corr_df,
            text_auto=True,
            color_continuous_scale=["#e8f5e9", "#a5d6a7", "#2e7d32"],
            zmin=-1, zmax=1,
            title="Correlação entre Variáveis Numéricas",
            aspect="auto",
        )
        fig9.update_layout(margin=dict(t=50))
        st.plotly_chart(fig9, use_container_width=True)
    
    # ── ROW 5: Insights ──────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-title">💡 Principais Insights para a Equipe Médica</div>',
                unsafe_allow_html=True)

    obesos_mask = df_f["Obesity"].isin(["Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"])
    pct_fam   = df_f[obesos_mask]["family_history"].value_counts(normalize=True).get("yes", 0) * 100
    faf_ob3   = df_f[df_f["Obesity"] == "Obesity_Type_III"]["FAF"].mean()
    faf_norm  = df_f[df_f["Obesity"] == "Normal_Weight"]["FAF"].mean()
    pct_favc  = df_f[obesos_mask]["FAVC"].value_counts(normalize=True).get("yes", 0) * 100
    pct_scc   = df_f[df_f["SCC"] == "yes"].shape[0] / len(df_f) * 100
    pct_trans = df_f[df_f["MTRANS"].isin(["Automobile", "Motorbike"])].shape[0] / len(df_f) * 100
    ch2o_norm = df_f[df_f["Obesity"] == "Normal_Weight"]["CH2O"].mean()
    ch2o_ob3  = df_f[df_f["Obesity"] == "Obesity_Type_III"]["CH2O"].mean()

    insights = [
        ("🧬 Fator Genético",
         f"{pct_fam:.0f}% dos pacientes obesos possuem histórico familiar de excesso de peso — o fator hereditário é determinante no risco."),
        ("🏃 Sedentarismo",
         f"Pacientes com Obesidade Tipo III praticam em média {faf_ob3:.1f}/3 de atividade física semanal, contra {faf_norm:.1f}/3 no grupo de Peso Normal."),
        ("🍔 Alimentação Calórica",
         f"{pct_favc:.0f}% dos pacientes obesos consomem alimentos altamente calóricos com frequência."),
        ("📊 Monitoramento Calórico",
         f"Apenas {pct_scc:.1f}% dos pacientes monitoram sua ingestão calórica — intervenção educativa de alto impacto potencial."),
        ("🚗 Transporte Passivo",
         f"{pct_trans:.0f}% dos pacientes utilizam transporte motorizado, reduzindo a atividade física diária."),
        ("💧 Hidratação",
         f"O consumo adequado de água é maior no grupo de Peso Normal ({ch2o_norm:.1f}/3) em comparação à Obesidade Tipo III ({ch2o_ob3:.1f}/3)."),
    ]

    cols_ins = st.columns(2)
    for i, (title, text) in enumerate(insights):
        with cols_ins[i % 2]:
            st.markdown(
                f'<div class="insight-box"><strong>{title}</strong><br>{text}</div>',
                unsafe_allow_html=True,
            )


# =============TESTE=============
