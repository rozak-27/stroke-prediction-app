import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
import io

warnings.filterwarnings("ignore")

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Deteksi Dini Risiko Stroke",
    page_icon=":brain:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"], [data-testid], p, div, span, h1, h2, h3, h4, label, button {
        font-family: 'DM Sans', sans-serif !important;
    }

    /* Main background — soft sky gradient */
    .stApp {
        background: linear-gradient(160deg, #e8f4fd 0%, #dbeafe 40%, #eff6ff 100%);
        min-height: 100vh;
    }

    /* ── Sidebar — glassy sky blue ── */
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.55) !important;
        backdrop-filter: blur(24px) !important;
        -webkit-backdrop-filter: blur(24px) !important;
        border-right: 1px solid rgba(147,197,253,0.45) !important;
        min-width: 300px !important;
        width: 300px !important;
        box-shadow: 4px 0 24px rgba(59,130,246,0.06);
    }
    [data-testid="stSidebar"] * {
        color: #1e3a5f !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    [data-testid="stSidebar"] .stRadio > div {
        gap: 6px;
        padding: 0 10px;
    }
    /* Bigger nav items */
    [data-testid="stSidebar"] .stRadio label {
        font-size: 17px !important;
        font-weight: 500 !important;
        padding: 16px 22px !important;
        border-radius: 14px !important;
        background: rgba(255,255,255,0.5) !important;
        border: 1px solid rgba(147,197,253,0.35) !important;
        width: 100% !important;
        display: block !important;
        transition: background 0.15s ease, box-shadow 0.15s ease;
        cursor: pointer;
        letter-spacing: 0px;
        line-height: 1.4;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(255,255,255,0.85) !important;
        border-color: rgba(59,130,246,0.45) !important;
        box-shadow: 0 3px 12px rgba(59,130,246,0.10);
    }

    /* ── Metric cards — glassy ── */
    .metric-card {
        background: rgba(255,255,255,0.55);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(147,197,253,0.45);
        border-radius: 16px;
        padding: 24px 16px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(59,130,246,0.07);
    }
    .metric-card h2 { color: #1d4ed8; font-size: 2rem; margin: 0; font-weight: 700; }
    .metric-card p  { color: #3b82f6; margin: 6px 0 0 0; font-size: 13px; font-weight: 500; }

    /* ── Hero ── */
    .hero {
        background: rgba(255,255,255,0.50);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        padding: 52px 36px;
        border-radius: 22px;
        border: 1px solid rgba(147,197,253,0.5);
        text-align: center;
        margin-bottom: 28px;
        box-shadow: 0 8px 36px rgba(59,130,246,0.09);
    }
    .hero h1 { color: #1d4ed8; font-size: 2.5rem; margin: 0; font-weight: 700; letter-spacing: -0.6px; }
    .hero h3 { color: #3b82f6; font-weight: 400; margin: 12px 0 0 0; font-size: 1.1rem; letter-spacing: 0.1px; }

    /* ── Result boxes ── */
    .result-high {
        background: rgba(254,226,226,0.75);
        backdrop-filter: blur(12px);
        border: 1.5px solid #ef4444;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        color: #7f1d1d;
    }
    .result-medium {
        background: rgba(255,251,235,0.75);
        backdrop-filter: blur(12px);
        border: 1.5px solid #f59e0b;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        color: #78350f;
    }
    .result-low {
        background: rgba(220,252,231,0.75);
        backdrop-filter: blur(12px);
        border: 1.5px solid #22c55e;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        color: #14532d;
    }

    /* ── Warn card ── */
    .warn-card {
        background: rgba(255,251,235,0.75);
        backdrop-filter: blur(10px);
        border-left: 4px solid #f59e0b;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 10px 0;
        color: #78350f;
    }

    /* ── Section header ── */
    .section-header {
        background: linear-gradient(90deg, rgba(37,99,235,0.85), rgba(96,165,250,0.85));
        backdrop-filter: blur(8px);
        color: white;
        padding: 11px 22px;
        border-radius: 10px;
        margin: 22px 0 16px 0;
        font-size: 15px;
        font-weight: 600;
        letter-spacing: 0.2px;
    }

    /* ── Team card — glassy ── */
    .team-card {
        background: rgba(255,255,255,0.60);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(147,197,253,0.45);
        border-radius: 18px;
        padding: 32px 24px;
        text-align: center;
        min-height: 130px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-shadow: 0 4px 20px rgba(59,130,246,0.07);
    }
    .team-card .name { color: #1d4ed8; font-weight: 600; font-size: 16px; margin-top: 4px; }
    .team-card .nim  { color: #3b82f6; font-size: 12px; margin-top: 4px; }

    /* ── Badge ── */
    .badge {
        display: inline-block;
        background: linear-gradient(90deg, #2563eb, #7dd3fc);
        color: white;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
        margin-top: 8px;
        letter-spacing: 0.3px;
    }

    /* ── Hide "Navigasi" radio label ── */
    [data-testid="stSidebar"] .stRadio > label {
        display: none !important;
    }

    /* ── Fix expander icon overlap ── */
    [data-testid="stExpander"] summary {
        font-size: 15px !important;
        font-weight: 500 !important;
        padding: 14px 16px !important;
    }
    [data-testid="stExpander"] summary p {
        font-size: 15px !important;
        font-weight: 500 !important;
    }
    [data-testid="stExpander"] details {
        border: 1px solid rgba(147,197,253,0.35) !important;
        border-radius: 10px !important;
        margin-bottom: 8px !important;
        background: rgba(255,255,255,0.45) !important;
        backdrop-filter: blur(10px) !important;
    }

    /* ── Fix file uploader double text ── */
    [data-testid="stFileUploaderDropzoneInstructions"] div span:last-child {
        display: none !important;
    }
    [data-testid="stFileUploader"] label {
        font-size: 14px !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar Navigation ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 36px 12px 32px 12px;'>
        <div style='
            width:56px; height:56px;
            background: linear-gradient(135deg, #2563eb, #7dd3fc);
            border-radius:16px; margin:0 auto;
            display:flex; align-items:center; justify-content:center;
            box-shadow: 0 4px 18px rgba(37,99,235,0.22);'>
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96-.46 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.84A2.5 2.5 0 0 1 9.5 2"/>
                <path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96-.46 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.84A2.5 2.5 0 0 0 14.5 2"/>
            </svg>
        </div>
        <div style='color:#1d4ed8; font-size:16px; font-weight:700; margin-top:16px; line-height:1.5;
                    letter-spacing:-0.3px; font-family: DM Sans, sans-serif;'>
            Deteksi Dini<br>Risiko Stroke
        </div>
        <div style='color:#60a5fa; font-size:11px; margin-top:5px; font-weight:500; letter-spacing:0.5px;
                    text-transform:uppercase;'>
            Data Mining · CRISP-DM
        </div>
    </div>
    <hr style='border:none; border-top:1px solid rgba(147,197,253,0.4); margin:0 12px 18px 12px;'>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigasi",
        ["Home",
         "Dataset Overview",
         "Prediction / Analysis",
         "Visualization",
         "About"],
        label_visibility="hidden"
    )

    st.markdown("""
    <hr style='border:none; border-top:1px solid rgba(147,197,253,0.4); margin:16px 12px 12px 12px;'>
    <div style='font-size:12px; color:#60a5fa; text-align:center; padding:0 8px;
                font-weight:500; line-height:1.7;'>
        Upload dataset CSV di<br>halaman Dataset Overview
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═════════════════════════════════════════════════════════════════════════════
if "df" not in st.session_state:           st.session_state.df = None
if "model_ready" not in st.session_state:  st.session_state.model_ready = False
if "rf_model" not in st.session_state:     st.session_state.rf_model = None
if "kmeans_model" not in st.session_state: st.session_state.kmeans_model = None
if "scaler" not in st.session_state:       st.session_state.scaler = None
if "feature_names" not in st.session_state: st.session_state.feature_names = None


# ═════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════
def load_sample_data():
    np.random.seed(42)
    n = 5110
    gender = np.random.choice(["Male", "Female"], n, p=[0.41, 0.59])
    age = np.random.beta(2.5, 5, n) * 100
    hypertension = np.random.choice([0, 1], n, p=[0.90, 0.10])
    heart_disease = np.random.choice([0, 1], n, p=[0.946, 0.054])
    ever_married = np.where(age > 35, np.random.choice(["Yes", "No"], n, p=[0.85, 0.15]),
                            np.random.choice(["Yes", "No"], n, p=[0.3, 0.7]))
    work_type = np.random.choice(
        ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
        n, p=[0.57, 0.17, 0.13, 0.12, 0.01])
    residence = np.random.choice(["Urban", "Rural"], n, p=[0.51, 0.49])
    glucose = np.concatenate([
        np.random.normal(91, 15, int(n * 0.7)),
        np.random.normal(200, 30, int(n * 0.3))])[:n]
    glucose = np.clip(glucose, 55, 300)
    bmi = np.random.normal(28.5, 7, n)
    bmi = np.where(np.random.random(n) < 0.04, np.nan, np.clip(bmi, 10, 55))
    smoke = np.random.choice(
        ["never smoked", "formerly smoked", "smokes", "Unknown"],
        n, p=[0.37, 0.17, 0.15, 0.31])
    stroke_prob = 0.02 + 0.003*(age > 60) + 0.04*hypertension + \
                  0.03*heart_disease + 0.001*(glucose > 150)
    stroke_prob = np.clip(stroke_prob, 0, 0.25)
    stroke = np.random.binomial(1, stroke_prob)
    return pd.DataFrame({
        "id": range(1, n+1), "gender": gender, "age": age.round(1),
        "hypertension": hypertension, "heart_disease": heart_disease,
        "ever_married": ever_married, "work_type": work_type,
        "Residence_type": residence, "avg_glucose_level": glucose.round(2),
        "bmi": bmi.round(1), "smoking_status": smoke, "stroke": stroke,
    })


def prepare_data(df_raw):
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import SMOTE
    df = df_raw.copy()
    df = df.drop(columns=["id"], errors="ignore")
    df = df[df["gender"] != "Other"].reset_index(drop=True)
    df["bmi"] = df["bmi"].fillna(df["bmi"].median())
    cat_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
    feature_names = [c for c in df.columns if c != "stroke"]
    X = df[feature_names]; y = df["stroke"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_sc, y_train)
    return X_train_sm, X_test_sc, y_train_sm, y_test, scaler, le_dict, feature_names


def train_models(X_train_sm, y_train_sm):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.cluster import KMeans
    rf = RandomForestClassifier(n_estimators=100, max_depth=10,
                                min_samples_split=5, min_samples_leaf=2,
                                random_state=42, n_jobs=-1)
    rf.fit(X_train_sm, y_train_sm)
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    km.fit(X_train_sm)
    return rf, km


def get_risk_level(prob):
    if prob >= 0.6:   return "TINGGI",   "#ef4444", "result-high"
    elif prob >= 0.3: return "MENENGAH", "#f59e0b", "result-medium"
    else:             return "RENDAH",   "#22c55e", "result-low"


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ═════════════════════════════════════════════════════════════════════════════
if page == "Home":
    st.markdown("""
    <div class='hero'>
        <h1>Deteksi Dini Risiko Stroke</h1>
        <h3>Analisis Pola dan Prediksi Menggunakan Data Mining</h3>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("<div class='metric-card'><h2>15M</h2><p>Kasus Stroke / Tahun (WHO)</p></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='metric-card'><h2>#2</h2><p>Penyebab Kematian Global</p></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='metric-card'><h2>80%</h2><p>Kasus Dapat Dicegah</p></div>", unsafe_allow_html=True)
    with c4:
        st.markdown("<div class='metric-card'><h2>5110</h2><p>Records Dataset</p></div>", unsafe_allow_html=True)

    st.markdown("---")

    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.markdown("### Deskripsi Proyek")
        st.markdown("""
        Proyek ini menerapkan teknik **Data Mining** untuk mendeteksi dini risiko stroke
        pada pasien berdasarkan profil kesehatan mereka. Stroke adalah kondisi medis darurat
        yang terjadi ketika aliran darah ke otak terhenti — mayoritas kasus **dapat dicegah**
        jika faktor risiko terdeteksi lebih awal.

        **Framework:** CRISP-DM *(Cross-Industry Standard Process for Data Mining)*

        **Dataset:** Stroke Prediction Dataset — Kaggle (fedesoriano)
        """)

        st.markdown("### Metode yang Digunakan")
        methods = [
            ("K-Means Clustering", "Segmentasi pasien menjadi kelompok risiko rendah / menengah / tinggi"),
            ("Random Forest",      "Prediksi probabilitas risiko stroke (model utama)"),
            ("Decision Tree",      "Model pembanding untuk justifikasi ensemble"),
            ("SHAP Explainable AI","Penjelasan mengapa model membuat prediksi tertentu"),
        ]
        for name, desc in methods:
            st.markdown(f"**{name}** — {desc}")

    with col_r:
        st.markdown("### Pertanyaan Riset")
        for i, q in enumerate([
            "Faktor kesehatan apa yang **paling berpengaruh** terhadap risiko stroke?",
            "Bagaimana **pola segmentasi** pasien berdasarkan profil kesehatan?",
            "Seberapa akurat model dalam **mendeteksi** kasus stroke (recall)?",
            "Apa perbedaan performa **Random Forest vs Decision Tree**?",
        ], 1):
            st.markdown(f"**{i}.** {q}")

        st.markdown("### Target Output")
        for o in [
            "Kelompok cluster pasien berdasarkan profil kesehatan",
            "Probabilitas risiko stroke (0–100%)",
            "Fitur-fitur paling berpengaruh terhadap stroke",
            "Perbandingan performa dua model ML",
        ]:
            st.markdown(f"✔ {o}")

    st.markdown("---")
    st.markdown("### Identitas Tim")

    members = [
        ("Nama Anggota 1", "NIM: 22XXXXXXX", "Ketua"),
        ("Nama Anggota 2", "NIM: 22XXXXXXX", "Anggota"),
    ]
    cols = st.columns(2)
    for col, (name, nim, role) in zip(cols, members):
        with col:
            st.markdown(f"""
            <div class='team-card'>
                <div class='name'>{name}</div>
                <div class='nim'>{nim}</div>
                <span class='badge'>{role}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align:center; color:#3b82f6; margin-top:24px; font-size:13px; font-weight:500;'>
        Mata Kuliah: Data Mining &nbsp;|&nbsp; Semester: Genap 2024/2025 &nbsp;|&nbsp; Framework: CRISP-DM
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DATASET OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Dataset Overview":
    st.title("Dataset Overview")
    st.markdown("Upload file CSV dataset Anda, atau gunakan **sample data** untuk demo.")

    uploaded = st.file_uploader(
        "Upload `healthcare-dataset-stroke-data.csv`", type=["csv"],
        help="Dataset dari Kaggle: fedesoriano/stroke-prediction-dataset")

    col_btn1, col_btn2, _ = st.columns([1, 1, 4])
    with col_btn1:
        use_sample = st.button("Gunakan Sample Data", type="secondary")
    with col_btn2:
        if st.session_state.df is not None:
            if st.button("Reset", type="secondary"):
                st.session_state.df = None
                st.session_state.model_ready = False
                st.rerun()

    if uploaded:
        st.session_state.df = pd.read_csv(uploaded)
        st.success(f"Dataset berhasil diupload: {len(st.session_state.df):,} baris")
    elif use_sample:
        st.session_state.df = load_sample_data()
        st.info("Menggunakan sample data sintetis (struktur identik dengan dataset asli Kaggle)")

    if st.session_state.df is not None:
        df = st.session_state.df

        st.markdown("<div class='section-header'>Informasi Dataset</div>", unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Records", f"{len(df):,}")
        c2.metric("Total Fitur", f"{df.shape[1]}")
        c3.metric("Kasus Stroke", f"{df['stroke'].sum():,}" if 'stroke' in df.columns else "N/A")
        c4.metric("Missing Values", f"{df.isnull().sum().sum():,}")
        c5.metric("Duplikat", f"{df.duplicated().sum():,}")

        with st.expander("Kamus Data (Data Dictionary)", expanded=False):
            dict_data = {
                "Kolom": ["id","gender","age","hypertension","heart_disease",
                          "ever_married","work_type","Residence_type",
                          "avg_glucose_level","bmi","smoking_status","stroke"],
                "Tipe": ["int","kategorik","numerik","biner","biner",
                         "kategorik","kategorik","kategorik","numerik","numerik","kategorik","biner"],
                "Keterangan": [
                    "ID unik pasien","Male / Female / Other","Usia pasien (tahun)",
                    "1 = ada riwayat hipertensi","1 = ada riwayat penyakit jantung",
                    "Yes / No","Jenis pekerjaan","Urban / Rural",
                    "Rata-rata kadar gula darah (mg/dL)","Body Mass Index",
                    "Status merokok","TARGET — 1 = Stroke"],
                "Status": ["Dihapus","Other dihapus","Penting","Faktor Risiko",
                           "Faktor Risiko","OK","OK","OK","Penting","201 missing","OK","Target"],
            }
            st.dataframe(pd.DataFrame(dict_data), use_container_width=True, hide_index=True)

        st.markdown("<div class='section-header'>5 Baris Pertama Dataset</div>", unsafe_allow_html=True)
        st.dataframe(df.head(), use_container_width=True)

        st.markdown("<div class='section-header'>Statistik Deskriptif</div>", unsafe_allow_html=True)
        st.dataframe(df.describe().round(3), use_container_width=True)

        st.markdown("<div class='section-header'>Analisis Missing Values & Distribusi</div>", unsafe_allow_html=True)
        col_mv, col_target = st.columns(2)

        with col_mv:
            mv = df.isnull().sum(); mv = mv[mv > 0]
            if len(mv) > 0:
                fig_mv = px.bar(x=mv.index, y=mv.values,
                    labels={"x":"Kolom","y":"Jumlah Missing"},
                    title="Missing Values per Kolom",
                    color=mv.values, color_continuous_scale="Blues")
                fig_mv.update_layout(showlegend=False, height=300, margin=dict(t=40,b=20,l=10,r=10))
                st.plotly_chart(fig_mv, use_container_width=True)
            else:
                st.success("Tidak ada missing values pada dataset ini.")

        with col_target:
            if "stroke" in df.columns:
                vc = df["stroke"].value_counts()
                fig_pie = px.pie(values=vc.values, names=["Tidak Stroke","Stroke"],
                    title="Distribusi Target Variable (Stroke)",
                    color_discrete_sequence=["#3b82f6","#ef4444"], hole=0.4)
                fig_pie.update_traces(textinfo="percent+label")
                fig_pie.update_layout(height=300, margin=dict(t=40,b=10,l=10,r=10))
                st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("<div class='section-header'>Heatmap Korelasi</div>", unsafe_allow_html=True)
        df_num = df.copy()
        from sklearn.preprocessing import LabelEncoder
        for c in df_num.select_dtypes("object").columns:
            df_num[c] = LabelEncoder().fit_transform(df_num[c].astype(str))
        df_num = df_num.drop(columns=["id"], errors="ignore")
        corr = df_num.corr().round(2)
        fig_heat = px.imshow(corr, text_auto=True, aspect="auto",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Korelasi Antar Fitur")
        fig_heat.update_layout(height=500, margin=dict(t=40,b=10,l=10,r=10))
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("---")
        st.markdown("### Latih Model")
        st.markdown("Klik tombol di bawah untuk melatih model Random Forest & K-Means menggunakan dataset yang sudah diupload.")

        if st.button("Latih Model Sekarang", type="primary", use_container_width=True):
            with st.spinner("Memproses data dan melatih model... (30-60 detik)"):
                try:
                    X_tr, X_te, y_tr, y_te, scaler, le_dict, feat_names = prepare_data(df)
                    rf_model, km_model = train_models(X_tr, y_tr)
                    st.session_state.rf_model      = rf_model
                    st.session_state.kmeans_model  = km_model
                    st.session_state.scaler        = scaler
                    st.session_state.le_dict       = le_dict
                    st.session_state.feature_names = feat_names
                    st.session_state.X_test        = X_te
                    st.session_state.y_test        = y_te
                    st.session_state.model_ready   = True
                    st.success("Model berhasil dilatih! Silakan gunakan menu Prediction & Visualization.")
                except Exception as e:
                    st.error(f"Error saat melatih model: {str(e)}")
    else:
        st.markdown("""
        <div class='warn-card'>
            <b>Belum ada dataset.</b><br>
            Upload file CSV atau klik <b>Gunakan Sample Data</b> untuk memulai.
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PREDICTION / ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Prediction / Analysis":
    st.title("Prediction / Analysis")

    if not st.session_state.model_ready:
        st.warning("Model belum dilatih. Silakan pergi ke **Dataset Overview** dan latih model terlebih dahulu.")
        st.stop()

    st.markdown("Masukkan data profil kesehatan pasien untuk mendapatkan prediksi risiko stroke.")
    st.markdown("---")

    with st.form("prediction_form"):
        st.markdown("### Form Input Data Pasien")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Informasi Demografis**")
            gender       = st.selectbox("Jenis Kelamin", ["Male", "Female"])
            age          = st.slider("Usia (tahun)", 1, 100, 45)
            ever_married = st.selectbox("Status Pernikahan", ["Yes", "No"])
            residence    = st.selectbox("Tipe Tempat Tinggal", ["Urban", "Rural"])

        with col2:
            st.markdown("**Riwayat Medis**")
            hypertension  = st.selectbox("Riwayat Hipertensi", [0, 1],
                format_func=lambda x: "Ya (1)" if x == 1 else "Tidak (0)")
            heart_disease = st.selectbox("Riwayat Penyakit Jantung", [0, 1],
                format_func=lambda x: "Ya (1)" if x == 1 else "Tidak (0)")
            avg_glucose   = st.number_input("Rata-rata Kadar Gula Darah (mg/dL)",
                min_value=55.0, max_value=300.0, value=100.0, step=0.1)
            bmi_val       = st.number_input("BMI (Body Mass Index)",
                min_value=10.0, max_value=60.0, value=25.0, step=0.1)

        with col3:
            st.markdown("**Gaya Hidup**")
            work_type = st.selectbox("Jenis Pekerjaan",
                ["Private","Self-employed","Govt_job","children","Never_worked"])
            smoking   = st.selectbox("Status Merokok",
                ["never smoked","formerly smoked","smokes","Unknown"])
            st.markdown("**Referensi Nilai Normal**")
            st.info("Gula Normal: 70–130 mg/dL\n\nBMI Normal: 18.5–24.9")

        submitted = st.form_submit_button("Analisis Risiko Stroke", type="primary", use_container_width=True)

    if submitted:
        from sklearn.preprocessing import LabelEncoder
        rf_model      = st.session_state.rf_model
        km_model      = st.session_state.kmeans_model
        scaler        = st.session_state.scaler
        le_dict       = st.session_state.le_dict
        feature_names = st.session_state.feature_names

        pasien = {
            "gender": gender, "age": age, "hypertension": hypertension,
            "heart_disease": heart_disease, "ever_married": ever_married,
            "work_type": work_type, "Residence_type": residence,
            "avg_glucose_level": avg_glucose, "bmi": bmi_val,
            "smoking_status": smoking,
        }
        inp = pd.DataFrame([pasien])
        for col, le in le_dict.items():
            if col in inp.columns:
                try:    inp[col] = le.transform(inp[col].astype(str))
                except: inp[col] = 0
        inp    = inp[feature_names]
        inp_sc = scaler.transform(inp)

        pred_prob   = rf_model.predict_proba(inp_sc)[0]
        cluster_p   = km_model.predict(inp_sc)[0]
        prob_stroke = pred_prob[1]

        risk_level, risk_color, risk_class = get_risk_level(prob_stroke)

        st.markdown("---")
        st.markdown("### Hasil Prediksi")

        col_res, col_detail = st.columns([2, 3])
        with col_res:
            st.markdown(f"""
            <div class='{risk_class}'>
                <h2 style='margin:0 0 8px 0;'>RISIKO {risk_level}</h2>
                <h3 style='margin:0;'>Probabilitas Stroke: {prob_stroke:.1%}</h3>
                <p style='margin:8px 0 0 0; opacity:0.75;'>Cluster: Kelompok {cluster_p}</p>
            </div>
            """, unsafe_allow_html=True)

        with col_detail:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_stroke * 100,
                domain={"x": [0,1], "y": [0,1]},
                title={"text": "Probabilitas Risiko Stroke (%)", "font": {"size": 14}},
                number={"suffix": "%", "font": {"size": 28}},
                gauge={
                    "axis": {"range": [0,100], "tickwidth": 1},
                    "bar": {"color": risk_color},
                    "steps": [
                        {"range": [0,30],   "color": "#dcfce7"},
                        {"range": [30,60],  "color": "#fef9c3"},
                        {"range": [60,100], "color": "#fee2e2"},
                    ],
                    "threshold": {"line": {"color":"#1e3a5f","width":3},
                                  "thickness":0.75, "value": prob_stroke*100},
                },
            ))
            fig_gauge.update_layout(height=250, margin=dict(t=30,b=10,l=20,r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown("### Kontribusi Faktor Risiko (Feature Importance)")
        fi = dict(zip(feature_names, rf_model.feature_importances_))
        fi_sorted   = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:8]
        feat_labels = [f[0] for f in fi_sorted]
        feat_vals   = [f[1] for f in fi_sorted]
        risky = []
        if age > 60:          risky.append("age")
        if hypertension == 1: risky.append("hypertension")
        if heart_disease == 1: risky.append("heart_disease")
        if avg_glucose > 150: risky.append("avg_glucose_level")
        if bmi_val > 30:      risky.append("bmi")
        bar_colors = ["#ef4444" if f in risky else "#3b82f6" for f in feat_labels]
        fig_fi = go.Figure(go.Bar(x=feat_vals, y=feat_labels, orientation="h", marker_color=bar_colors))
        fig_fi.update_layout(title="Feature Importance (Merah = Faktor Risiko Pasien Ini)",
                             xaxis_title="Importance Score", height=320,
                             margin=dict(t=40,b=20,l=10,r=20))
        st.plotly_chart(fig_fi, use_container_width=True)

        st.markdown("### Rekomendasi")
        recs = []
        if age > 60:          recs.append("**Usia > 60 tahun** — lakukan pemeriksaan kesehatan rutin minimal setahun sekali.")
        if hypertension == 1: recs.append("**Hipertensi** — pantau tekanan darah secara rutin dan patuhi pengobatan.")
        if heart_disease == 1: recs.append("**Penyakit Jantung** — konsultasikan dengan dokter spesialis jantung.")
        if avg_glucose > 150: recs.append("**Kadar Gula Tinggi** — kurangi konsumsi gula, evaluasi kemungkinan diabetes.")
        if bmi_val > 30:      recs.append("**BMI Obesitas** — terapkan pola makan sehat dan olahraga rutin.")
        if smoking in ["smokes","formerly smoked"]:
            recs.append("**Status Merokok** — berhenti merokok dapat menurunkan risiko stroke secara signifikan.")
        if not recs:
            st.success("Profil kesehatan Anda tergolong baik. Pertahankan gaya hidup sehat!")
        else:
            for r in recs: st.markdown(f"- {r}")

        st.markdown("""
        <div class='warn-card'>
            <b>Disclaimer:</b> Prediksi ini hanya bersifat informatif dan tidak menggantikan
            diagnosis medis profesional. Konsultasikan dengan dokter untuk evaluasi lebih lanjut.
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Visualization":
    st.title("Visualization")

    if st.session_state.df is None:
        st.warning("Belum ada dataset. Pergi ke **Dataset Overview** untuk upload atau gunakan sample data.")
        st.stop()

    df = st.session_state.df
    tab1, tab2, tab3, tab4 = st.tabs(["EDA", "Clustering", "Klasifikasi", "Evaluasi Model"])

    with tab1:
        st.markdown("#### Distribusi Fitur Numerik berdasarkan Status Stroke")
        num_col = st.selectbox("Pilih fitur numerik:", ["age","avg_glucose_level","bmi"])
        if num_col in df.columns and "stroke" in df.columns:
            fig_dist = px.histogram(df, x=num_col, color="stroke", barmode="overlay", nbins=40,
                color_discrete_map={0:"#3b82f6", 1:"#ef4444"},
                labels={"stroke":"Status Stroke", num_col: num_col},
                title=f"Distribusi {num_col} berdasarkan Status Stroke")
            fig_dist.update_layout(height=350, margin=dict(t=40,b=20))
            st.plotly_chart(fig_dist, use_container_width=True)

        st.markdown("#### Distribusi Fitur Kategorik vs Stroke Rate")
        cat_col = st.selectbox("Pilih fitur kategorik:",
            ["gender","hypertension","heart_disease","ever_married","work_type","smoking_status"])
        if cat_col in df.columns and "stroke" in df.columns:
            ct = df.groupby(cat_col)["stroke"].agg(["sum","count"]).reset_index()
            ct.columns = [cat_col,"stroke_count","total"]
            ct["stroke_rate"] = (ct["stroke_count"]/ct["total"]*100).round(2)
            fig_cat = px.bar(ct, x=cat_col, y="stroke_rate", color="stroke_rate",
                color_continuous_scale="Blues",
                title=f"Stroke Rate (%) per Kategori {cat_col}",
                labels={"stroke_rate":"Stroke Rate (%)"}, text="stroke_rate")
            fig_cat.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_cat.update_layout(height=350, margin=dict(t=40,b=20), showlegend=False)
            st.plotly_chart(fig_cat, use_container_width=True)

        st.markdown("#### Scatter Plot: Usia vs Glukosa")
        if all(c in df.columns for c in ["age","avg_glucose_level","stroke"]):
            sample = df.sample(min(2000, len(df)), random_state=42)
            fig_sc = px.scatter(sample, x="age", y="avg_glucose_level",
                color=sample["stroke"].map({0:"Tidak Stroke",1:"Stroke"}),
                color_discrete_map={"Tidak Stroke":"#3b82f6","Stroke":"#ef4444"},
                opacity=0.6, title="Usia vs Rata-rata Kadar Gula (Stroke vs Tidak)",
                labels={"avg_glucose_level":"Rata-rata Gula (mg/dL)","age":"Usia"})
            fig_sc.update_layout(height=400, margin=dict(t=40,b=20))
            st.plotly_chart(fig_sc, use_container_width=True)

    with tab2:
        if not st.session_state.model_ready:
            st.info("Latih model terlebih dahulu di halaman Dataset Overview.")
        else:
            df2 = df.copy()
            from sklearn.preprocessing import LabelEncoder, StandardScaler
            for c in df2.select_dtypes("object").columns:
                df2[c] = LabelEncoder().fit_transform(df2[c].astype(str))
            df2 = df2.drop(columns=["id","stroke"], errors="ignore").fillna(df2.median())
            sc2 = st.session_state.scaler
            km2 = st.session_state.kmeans_model
            fn2 = st.session_state.feature_names
            df2 = df2[fn2] if all(c in df2.columns for c in fn2) else df2.iloc[:, :len(fn2)]
            X_sc2  = sc2.transform(df2)
            labels = km2.predict(X_sc2)

            from sklearn.decomposition import PCA
            pca    = PCA(n_components=2, random_state=42)
            pcs    = pca.fit_transform(X_sc2)
            pca_df = pd.DataFrame({"PC1": pcs[:,0], "PC2": pcs[:,1], "Cluster": labels.astype(str)})

            st.markdown("#### Visualisasi Cluster (PCA 2D)")
            fig_pca = px.scatter(pca_df, x="PC1", y="PC2", color="Cluster",
                color_discrete_sequence=["#3b82f6","#ef4444","#22c55e"], opacity=0.6,
                title=f"K-Means Clustering (k=3) — PCA 2D ({pca.explained_variance_ratio_.sum()*100:.1f}% variance)")
            fig_pca.update_layout(height=400, margin=dict(t=40,b=20))
            st.plotly_chart(fig_pca, use_container_width=True)

            st.markdown("#### Statistik per Cluster")
            orig_df = df.drop(columns=["id"], errors="ignore").copy()
            for c in orig_df.select_dtypes("object").columns:
                orig_df[c] = LabelEncoder().fit_transform(orig_df[c].astype(str))
            orig_df = orig_df.fillna(orig_df.median())
            orig_df["Cluster"] = labels
            num_feats     = [c for c in ["age","avg_glucose_level","bmi","hypertension","heart_disease"] if c in orig_df.columns]
            cluster_stats = orig_df.groupby("Cluster")[num_feats].mean().round(2)
            if "stroke" in df.columns:
                cluster_stats["Stroke Rate (%)"] = (orig_df.groupby("Cluster")["stroke"].mean()*100).round(2)
            st.dataframe(cluster_stats, use_container_width=True)

            st.markdown("#### Radar Chart Profil Cluster")
            radar_feats = [c for c in ["age","avg_glucose_level","bmi","hypertension","heart_disease"] if c in orig_df.columns]
            c_avg = orig_df.groupby("Cluster")[radar_feats].mean().copy()
            for col in radar_feats:
                rng = c_avg[col].max() - c_avg[col].min() + 1e-10
                c_avg[col] = (c_avg[col] - c_avg[col].min()) / rng
            angles    = radar_feats + [radar_feats[0]]
            fig_radar = go.Figure()
            for i, row in c_avg.iterrows():
                vals = row.values.tolist() + [row.values[0]]
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals, theta=angles, fill="toself",
                    name=f"Cluster {i}", line_color=["#3b82f6","#ef4444","#22c55e"][i], opacity=0.7))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                title="Radar Chart — Profil Cluster (Ternormalisasi 0-1)",
                height=420, margin=dict(t=60,b=20))
            st.plotly_chart(fig_radar, use_container_width=True)

    with tab3:
        if not st.session_state.model_ready:
            st.info("Latih model terlebih dahulu di halaman Dataset Overview.")
        else:
            rf_model      = st.session_state.rf_model
            feature_names = st.session_state.feature_names
            fi    = dict(zip(feature_names, rf_model.feature_importances_))
            fi_df = pd.DataFrame(fi.items(), columns=["Fitur","Importance"]).sort_values("Importance", ascending=True)
            fig_fi = px.bar(fi_df, x="Importance", y="Fitur", orientation="h",
                color="Importance", color_continuous_scale="Blues",
                title="Feature Importance — Random Forest",
                labels={"Importance":"Importance Score"})
            fig_fi.update_layout(height=400, showlegend=False, margin=dict(t=40,b=20,l=10,r=20))
            st.plotly_chart(fig_fi, use_container_width=True)

            if "stroke" in df.columns:
                df_corr2 = df.copy()
                from sklearn.preprocessing import LabelEncoder
                for c in df_corr2.select_dtypes("object").columns:
                    df_corr2[c] = LabelEncoder().fit_transform(df_corr2[c].astype(str))
                df_corr2 = df_corr2.drop(columns=["id"], errors="ignore").fillna(df_corr2.median())
                corr_stroke = df_corr2.corr()["stroke"].drop("stroke").abs().sort_values(ascending=False)
                comp_df = pd.DataFrame({
                    "Fitur": corr_stroke.index,
                    "Korelasi (|r|)": corr_stroke.values,
                    "RF Importance": [fi.get(f,0) for f in corr_stroke.index],
                })
                fig_comp = go.Figure()
                fig_comp.add_bar(name="Korelasi (|r|)", x=comp_df["Fitur"], y=comp_df["Korelasi (|r|)"], marker_color="#60a5fa")
                fig_comp.add_bar(name="RF Importance",  x=comp_df["Fitur"], y=comp_df["RF Importance"],  marker_color="#2563eb")
                fig_comp.update_layout(barmode="group", height=380,
                    title="Korelasi vs RF Importance per Fitur", margin=dict(t=40,b=20))
                st.plotly_chart(fig_comp, use_container_width=True)

    with tab4:
        if not st.session_state.model_ready:
            st.info("Latih model terlebih dahulu di halaman Dataset Overview.")
        else:
            from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                                         recall_score, f1_score, roc_auc_score, roc_curve)
            from sklearn.tree import DecisionTreeClassifier
            rf_model = st.session_state.rf_model
            X_test   = st.session_state.X_test
            y_test   = st.session_state.y_test

            with st.spinner("Melatih Decision Tree untuk perbandingan..."):
                try:
                    df_prep = df.copy().drop(columns=["id"], errors="ignore")
                    for c in df_prep.select_dtypes("object").columns:
                        df_prep[c] = LabelEncoder().fit_transform(df_prep[c].astype(str))
                    df_prep = df_prep.fillna(df_prep.median())
                    feat_names = st.session_state.feature_names
                    X_all = df_prep[feat_names]; y_all = df_prep["stroke"]
                    from sklearn.model_selection import train_test_split
                    from imblearn.over_sampling import SMOTE
                    X_tr2, _, y_tr2, _ = train_test_split(X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)
                    X_tr2_sc = st.session_state.scaler.transform(X_tr2)
                    X_tr2_sm, y_tr2_sm = SMOTE(random_state=42).fit_resample(X_tr2_sc, y_tr2)
                    dt_model = DecisionTreeClassifier(max_depth=8, min_samples_split=10, min_samples_leaf=5, random_state=42)
                    dt_model.fit(X_tr2_sm, y_tr2_sm)
                except Exception: dt_model = None

            y_pred_rf = rf_model.predict(X_test)
            y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

            st.markdown("#### Confusion Matrix — Random Forest")
            cm_rf  = confusion_matrix(y_test, y_pred_rf)
            fig_cm = px.imshow(cm_rf, text_auto=True, aspect="auto",
                color_continuous_scale="Blues",
                labels={"x":"Prediksi","y":"Aktual"},
                x=["Tidak Stroke","Stroke"], y=["Tidak Stroke","Stroke"],
                title="Confusion Matrix — Random Forest")
            fig_cm.update_layout(height=350, margin=dict(t=40,b=20))
            st.plotly_chart(fig_cm, use_container_width=True)

            st.markdown("#### Metrik Evaluasi")
            metrics = {
                "Accuracy":  accuracy_score(y_test, y_pred_rf),
                "Precision": precision_score(y_test, y_pred_rf, zero_division=0),
                "Recall":    recall_score(y_test, y_pred_rf),
                "F1-Score":  f1_score(y_test, y_pred_rf, zero_division=0),
                "ROC-AUC":   roc_auc_score(y_test, y_prob_rf),
            }
            m_cols = st.columns(5)
            for (k,v), col in zip(metrics.items(), m_cols):
                col.metric(k, f"{v:.4f}", delta="Prioritas" if k=="Recall" else "")

            st.markdown("#### ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, y_prob_rf)
            auc_rf      = roc_auc_score(y_test, y_prob_rf)
            fig_roc     = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                name=f"Random Forest (AUC={auc_rf:.4f})", line=dict(color="#2563eb",width=3)))
            if dt_model:
                y_prob_dt   = dt_model.predict_proba(X_test)[:,1]
                fpr_dt, tpr_dt, _ = roc_curve(y_test, y_prob_dt)
                auc_dt      = roc_auc_score(y_test, y_prob_dt)
                fig_roc.add_trace(go.Scatter(x=fpr_dt, y=tpr_dt, mode="lines",
                    name=f"Decision Tree (AUC={auc_dt:.4f})", line=dict(color="#60a5fa",width=3,dash="dash")))
            fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                name="Random (AUC=0.50)", line=dict(color="#94a3b8",width=2,dash="dot")))
            fig_roc.update_layout(title="ROC Curve — Perbandingan Model",
                xaxis_title="False Positive Rate", yaxis_title="True Positive Rate (Recall)",
                height=400, margin=dict(t=40,b=20))
            st.plotly_chart(fig_roc, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5 — ABOUT
# ═════════════════════════════════════════════════════════════════════════════
elif page == "About":
    st.title("About")

    tab_method, tab_dataset, tab_project = st.tabs(["Metode", "Dataset", "Informasi Proyek"])

    with tab_method:
        st.markdown("### Penjelasan Metode Data Mining")

        with st.expander("K-Means Clustering (Unsupervised Learning)", expanded=True):
            st.markdown("""
            **K-Means Clustering** adalah algoritma unsupervised yang mengelompokkan data ke dalam
            *k* cluster berdasarkan kesamaan fitur, meminimalkan jarak intra-cluster (WCSS).

            **Cara Kerja:**
            1. Inisialisasi *k* centroid secara acak
            2. Setiap data point ditetapkan ke centroid terdekat (jarak Euclidean)
            3. Hitung ulang centroid berdasarkan rata-rata anggota cluster
            4. Ulangi langkah 2–3 hingga konvergen

            **Pemilihan K Optimal:** Elbow Method, Silhouette Score, Davies-Bouldin Score

            **Output di Proyek Ini:** 3 cluster pasien berdasarkan profil kesehatan
            """)

        with st.expander("Random Forest (Model Utama)", expanded=True):
            st.markdown("""
            **Random Forest** adalah algoritma ensemble yang membangun banyak Decision Tree secara
            paralel (bagging) dan menggabungkan hasil prediksi melalui voting mayoritas.

            | Parameter | Nilai | Alasan |
            |-----------|-------|--------|
            | n_estimators | 100 | Keseimbangan kecepatan-akurasi |
            | max_depth | 10 | Mencegah overfitting |
            | min_samples_split | 5 | Minimum sampel sebelum split |
            | min_samples_leaf | 2 | Minimum sampel di daun |
            """)

        with st.expander("Decision Tree (Model Pembanding)"):
            st.markdown("""
            **Decision Tree** membangun model berupa pohon aturan IF-THEN yang mudah diinterpretasi.

            | Aspek | Decision Tree | Random Forest |
            |-------|--------------|---------------|
            | Tipe | Single tree | Ensemble (100 trees) |
            | Interpretasi | Mudah (whitebox) | Sulit (blackbox) |
            | Overfitting | Rentan | Lebih robust |
            | Akurasi | Lebih rendah | Lebih tinggi |
            """)

        with st.expander("Penanganan Imbalanced Data — SMOTE"):
            st.markdown("""
            Dataset stroke sangat **tidak seimbang** (~95% tidak stroke, ~5% stroke).

            **SMOTE (Synthetic Minority Oversampling Technique):**
            - Membuat data sintetis untuk kelas minoritas (stroke)
            - Interpolasi antara sampel minoritas yang ada
            - Diterapkan **hanya pada training data** (mencegah data leakage)

            **Pipeline:**
            ```
            Train-Test Split → StandardScaler (fit train only) → SMOTE (train only)
            ```
            """)

        with st.expander("Metrik Evaluasi"):
            st.markdown("""
            | Metrik | Penjelasan | Prioritas |
            |--------|-----------|-----------|
            | **Accuracy** | Proporsi prediksi benar dari total | Misleading pada data imbalanced |
            | **Precision** | Dari yang diprediksi stroke, berapa yang benar? | Sedang |
            | **Recall** | Dari kasus stroke nyata, berapa yang terdeteksi? | **Tinggi** |
            | **F1-Score** | Harmonic mean Precision & Recall | Tinggi |
            | **ROC-AUC** | Kemampuan membedakan dua kelas (0-1) | Tinggi |

            > **Recall diprioritaskan** karena False Negative (stroke tidak terdeteksi) jauh lebih berbahaya dari False Positive.
            """)

    with tab_dataset:
        st.markdown("### Informasi Dataset")
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("""
            **Nama Dataset:** Stroke Prediction Dataset

            **Sumber:** Kaggle — fedesoriano/stroke-prediction-dataset

            **Jumlah Records:** 5.110 baris

            **Jumlah Fitur:** 12 kolom (11 fitur + 1 target)

            **Target Variable:** `stroke` (0 = tidak stroke, 1 = stroke)

            **Distribusi Target:**
            - Tidak Stroke: ~4.861 (95.1%)
            - Stroke: ~249 (4.9%)
            - Rasio ketidakseimbangan: ~20:1
            """)
        with col_r:
            st.markdown("""
            **Missing Values:**
            - Kolom `bmi`: 201 nilai kosong (3.9%)
            - Penanganan: diisi dengan **median**

            **Preprocessing Steps:**
            1. Hapus kolom `id`
            2. Hapus baris `gender='Other'`
            3. Imputasi `bmi` dengan median
            4. Label Encoding untuk fitur kategorik
            5. Train-Test Split (80:20, stratified)
            6. StandardScaler (fit hanya pada train)
            7. SMOTE oversampling (hanya pada train)

            **Lisensi:** Open Database License (ODbL)
            """)

    with tab_project:
        st.markdown("### Informasi Proyek")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Mata Kuliah:** Data Mining

            **Semester:** Genap 2024/2025

            **Program Studi:** Informatika / Ilmu Komputer

            **Framework:** CRISP-DM

            **Tools & Libraries:**
            - Python 3.x
            - Streamlit (Web App)
            - Scikit-learn (ML)
            - Imbalanced-learn (SMOTE)
            - Plotly (Visualisasi Interaktif)
            """)
        with col2:
            st.markdown("""
            **Checklist CRISP-DM:**

            | Fase | Status |
            |------|--------|
            | Business Understanding | Done |
            | Data Understanding | Done |
            | Data Preparation | Done |
            | Modeling — Clustering | Done |
            | Modeling — Classification | Done |
            | Evaluation | Done |
            | Deployment (Web App) | Done |
            """)

        st.markdown("---")
        st.markdown("### Tim Proyek")
        members = [
            ("M Ridho Aulia", "NIM: 24051214116", "Anggota— Modeling & Evaluasi"),
            ("Rozak M Limbong", "NIM: 24051214116", "Anggota — EDA, Visualisasi & Deployment"),
        ]
        cols = st.columns(2)
        for col, (name, nim, role) in zip(cols, members):
            with col:
                st.markdown(f"""
                <div class='team-card'>
                    <div class='name'>{name}</div>
                    <div class='nim'>{nim}</div>
                    <div class='nim' style='margin-top:4px; color:#2563eb; font-weight:500;'>{role}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("""
        <div style='text-align:center; color:#3b82f6; margin-top:28px; font-size:13px;
                    font-weight:500; padding:16px;'>
            Deteksi Dini Risiko Stroke — Proyek Data Mining 2024/2025<br>
            Dibuat menggunakan Streamlit + Scikit-learn + Plotly
        </div>
        """, unsafe_allow_html=True)