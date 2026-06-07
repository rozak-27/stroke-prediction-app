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

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Deteksi Dini Risiko Stroke",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    [data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        font-size: 15px;
        padding: 6px 0;
    }

    /* Cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        color: white;
    }
    .metric-card h2 { color: #e94560; font-size: 2rem; margin: 0; }
    .metric-card p  { color: #a8dadc; margin: 4px 0 0 0; font-size: 14px; }

    /* Hero banner */
    .hero {
        background: linear-gradient(135deg, #1a1a2e 0%, #0f3460 50%, #16213e 100%);
        padding: 40px 30px;
        border-radius: 16px;
        border: 1px solid #e94560;
        text-align: center;
        margin-bottom: 24px;
    }
    .hero h1 { color: #e94560; font-size: 2.4rem; margin: 0; }
    .hero h3 { color: #a8dadc; font-weight: 400; margin: 8px 0 0 0; }

    /* Result box */
    .result-high {
        background: linear-gradient(135deg, #3d0000, #7b0000);
        border: 2px solid #e74c3c;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        color: white;
    }
    .result-medium {
        background: linear-gradient(135deg, #3d2200, #7b4500);
        border: 2px solid #f39c12;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        color: white;
    }
    .result-low {
        background: linear-gradient(135deg, #003d1a, #007b35);
        border: 2px solid #2ecc71;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        color: white;
    }

    /* Info card */
    .info-card {
        background: #f8f9fa;
        border-left: 4px solid #3498db;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 10px 0;
    }
    .warn-card {
        background: #fff8e1;
        border-left: 4px solid #f39c12;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 10px 0;
    }

    /* Section header */
    .section-header {
        background: linear-gradient(90deg, #e94560, #0f3460);
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        margin: 20px 0 16px 0;
        font-size: 17px;
        font-weight: 600;
    }

    /* Team member card */
    .team-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        color: white;
        height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .team-card .avatar { font-size: 2.2rem; }
    .team-card .name   { color: #a8dadc; font-weight: 600; font-size: 15px; margin-top: 6px; }
    .team-card .nim    { color: #888; font-size: 12px; }

    /* Badge */
    .badge {
        display: inline-block;
        background: #e94560;
        color: white;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar Navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 24px 0;'>
        <div style='font-size: 3rem;'>🧠</div>
        <div style='color:#e94560; font-size:16px; font-weight:700; margin-top:8px;'>
            Deteksi Dini Risiko Stroke
        </div>
        <div style='color:#888; font-size:12px;'>Data Mining · CRISP-DM</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigasi",
        ["🏠  Home",
         "📊  Dataset Overview",
         "🔮  Prediction / Analysis",
         "📈  Visualization",
         "ℹ️  About"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px; color:#666; text-align:center;'>
        📌 Upload dataset CSV di<br>halaman Dataset Overview
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE — simpan dataset & model
# ══════════════════════════════════════════════════════════════════════════════
if "df" not in st.session_state:
    st.session_state.df = None
if "model_ready" not in st.session_state:
    st.session_state.model_ready = False
if "rf_model" not in st.session_state:
    st.session_state.rf_model = None
if "kmeans_model" not in st.session_state:
    st.session_state.kmeans_model = None
if "scaler" not in st.session_state:
    st.session_state.scaler = None
if "feature_names" not in st.session_state:
    st.session_state.feature_names = None


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def load_sample_data():
    """Generate synthetic sample data that mimics the Kaggle stroke dataset."""
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
        n, p=[0.57, 0.17, 0.13, 0.12, 0.01]
    )
    residence = np.random.choice(["Urban", "Rural"], n, p=[0.51, 0.49])
    glucose = np.concatenate([
        np.random.normal(91, 15, int(n * 0.7)),
        np.random.normal(200, 30, int(n * 0.3))
    ])[:n]
    glucose = np.clip(glucose, 55, 300)
    bmi = np.random.normal(28.5, 7, n)
    bmi = np.where(np.random.random(n) < 0.04, np.nan, np.clip(bmi, 10, 55))
    smoke = np.random.choice(
        ["never smoked", "formerly smoked", "smokes", "Unknown"],
        n, p=[0.37, 0.17, 0.15, 0.31]
    )
    # Stroke: imbalanced ~4.9%
    stroke_prob = 0.02 + 0.003 * (age > 60) + 0.04 * hypertension + \
                  0.03 * heart_disease + 0.001 * (glucose > 150)
    stroke_prob = np.clip(stroke_prob, 0, 0.25)
    stroke = np.random.binomial(1, stroke_prob)

    df = pd.DataFrame({
        "id": range(1, n + 1),
        "gender": gender,
        "age": age.round(1),
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": residence,
        "avg_glucose_level": glucose.round(2),
        "bmi": bmi.round(1),
        "smoking_status": smoke,
        "stroke": stroke,
    })
    return df


def prepare_data(df_raw):
    """Replicate notebook preprocessing pipeline."""
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import SMOTE

    df = df_raw.copy()
    df = df.drop(columns=["id"], errors="ignore")
    df = df[df["gender"] != "Other"].reset_index(drop=True)
    bmi_median = df["bmi"].median()
    df["bmi"] = df["bmi"].fillna(bmi_median)

    cat_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le

    feature_names = [c for c in df.columns if c != "stroke"]
    X = df[feature_names]
    y = df["stroke"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_sc, y_train)

    return X_train_sm, X_test_sc, y_train_sm, y_test, scaler, le_dict, feature_names


def train_models(X_train_sm, y_train_sm):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.cluster import KMeans

    rf = RandomForestClassifier(
        n_estimators=100, max_depth=10,
        min_samples_split=5, min_samples_leaf=2,
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train_sm, y_train_sm)

    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    km.fit(X_train_sm)

    return rf, km


def get_risk_level(prob):
    if prob >= 0.6:
        return "TINGGI", "#e74c3c", "🚨", "result-high"
    elif prob >= 0.3:
        return "MENENGAH", "#f39c12", "⚠️", "result-medium"
    else:
        return "RENDAH", "#2ecc71", "✅", "result-low"


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Home":
    st.markdown("""
    <div class='hero'>
        <h1>🧠 Deteksi Dini Risiko Stroke</h1>
        <h3>Analisis Pola dan Prediksi Menggunakan Data Mining</h3>
    </div>
    """, unsafe_allow_html=True)

    # Stats row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""<div class='metric-card'><h2>15M</h2><p>Kasus Stroke / Tahun (WHO)</p></div>""",
                    unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class='metric-card'><h2>#2</h2><p>Penyebab Kematian Global</p></div>""",
                    unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class='metric-card'><h2>80%</h2><p>Kasus Dapat Dicegah</p></div>""",
                    unsafe_allow_html=True)
    with c4:
        st.markdown("""<div class='metric-card'><h2>5110</h2><p>Records Dataset</p></div>""",
                    unsafe_allow_html=True)

    st.markdown("---")

    # Project description
    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.markdown("### 📋 Deskripsi Proyek")
        st.markdown("""
        Proyek ini menerapkan teknik **Data Mining** untuk mendeteksi dini risiko stroke
        pada pasien berdasarkan profil kesehatan mereka. Stroke adalah kondisi medis darurat
        yang terjadi ketika aliran darah ke otak terhenti — mayoritas kasus **dapat dicegah**
        jika faktor risiko terdeteksi lebih awal.

        **Framework:** CRISP-DM *(Cross-Industry Standard Process for Data Mining)*

        **Dataset:** Stroke Prediction Dataset — Kaggle (fedesoriano)
        """)

        st.markdown("### 🔬 Metode yang Digunakan")
        methods = [
            ("🔵", "K-Means Clustering", "Segmentasi pasien menjadi kelompok risiko rendah / menengah / tinggi"),
            ("🟢", "Random Forest", "Prediksi probabilitas risiko stroke (model utama)"),
            ("🟡", "Decision Tree", "Model pembanding untuk justifikasi ensemble"),
            ("🔴", "SHAP Explainable AI", "Penjelasan mengapa model membuat prediksi tertentu"),
        ]
        for icon, name, desc in methods:
            st.markdown(f"**{icon} {name}** — {desc}")

    with col_r:
        st.markdown("### ❓ Pertanyaan Riset")
        questions = [
            "Faktor kesehatan apa yang **paling berpengaruh** terhadap risiko stroke?",
            "Bagaimana **pola segmentasi** pasien berdasarkan profil kesehatan?",
            "Seberapa akurat model dalam **mendeteksi** kasus stroke (recall)?",
            "Apa perbedaan performa **Random Forest vs Decision Tree**?",
        ]
        for i, q in enumerate(questions, 1):
            st.markdown(f"**{i}.** {q}")

        st.markdown("### 🎯 Target Output")
        outputs = [
            "Kelompok cluster pasien berdasarkan profil kesehatan",
            "Probabilitas risiko stroke (0–100%)",
            "Fitur-fitur paling berpengaruh terhadap stroke",
            "Perbandingan performa dua model ML",
        ]
        for o in outputs:
            st.markdown(f"✔️ {o}")

    st.markdown("---")
    st.markdown("### 👥 Identitas Tim")

    members = [
        ("👨‍💻", "Nama Anggota 1", "NIM: 22XXXXXXX", "Ketua"),
        ("👩‍💻", "Nama Anggota 2", "NIM: 22XXXXXXX", "Anggota"),
        ("👨‍💻", "Nama Anggota 3", "NIM: 22XXXXXXX", "Anggota"),
        ("👩‍💻", "Nama Anggota 4", "NIM: 22XXXXXXX", "Anggota"),
    ]
    cols = st.columns(4)
    for col, (avatar, name, nim, role) in zip(cols, members):
        with col:
            st.markdown(f"""
            <div class='team-card'>
                <div class='avatar'>{avatar}</div>
                <div class='name'>{name}</div>
                <div class='nim'>{nim}</div>
                <div style='margin-top:6px;'>
                    <span class='badge'>{role}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align:center; color:#888; margin-top:24px; font-size:13px;'>
        Mata Kuliah: Data Mining &nbsp;|&nbsp; Semester: Genap 2024/2025 &nbsp;|&nbsp;
        Framework: CRISP-DM
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DATASET OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Dataset Overview":
    st.title("📊 Dataset Overview")
    st.markdown("Upload file CSV dataset Anda, atau gunakan **sample data** untuk demo.")

    # Upload
    uploaded = st.file_uploader(
        "Upload `healthcare-dataset-stroke-data.csv`",
        type=["csv"],
        help="Dataset dari Kaggle: fedesoriano/stroke-prediction-dataset"
    )

    col_btn1, col_btn2, _ = st.columns([1, 1, 4])
    with col_btn1:
        use_sample = st.button("🎲 Gunakan Sample Data", type="secondary")
    with col_btn2:
        if st.session_state.df is not None:
            clear = st.button("🗑️ Reset", type="secondary")
            if clear:
                st.session_state.df = None
                st.session_state.model_ready = False
                st.rerun()

    if uploaded:
        st.session_state.df = pd.read_csv(uploaded)
        st.success(f"✅ Dataset berhasil diupload: {len(st.session_state.df):,} baris")
    elif use_sample:
        st.session_state.df = load_sample_data()
        st.info("🎲 Menggunakan sample data sintetis (struktur identik dengan dataset asli Kaggle)")

    if st.session_state.df is not None:
        df = st.session_state.df

        # ── Basic Info ─────────────────────────────────────────────────────
        st.markdown("<div class='section-header'>📋 Informasi Dataset</div>", unsafe_allow_html=True)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Records", f"{len(df):,}")
        c2.metric("Total Fitur", f"{df.shape[1]}")
        c3.metric("Kasus Stroke", f"{df['stroke'].sum():,}" if 'stroke' in df.columns else "N/A")
        c4.metric("Missing Values", f"{df.isnull().sum().sum():,}")
        c5.metric("Duplikat", f"{df.duplicated().sum():,}")

        # ── Data Dictionary ────────────────────────────────────────────────
        with st.expander("📖 Kamus Data (Data Dictionary)", expanded=False):
            dict_data = {
                "Kolom": ["id", "gender", "age", "hypertension", "heart_disease",
                          "ever_married", "work_type", "Residence_type",
                          "avg_glucose_level", "bmi", "smoking_status", "stroke"],
                "Tipe": ["int", "kategorik", "numerik", "biner", "biner",
                         "kategorik", "kategorik", "kategorik",
                         "numerik", "numerik", "kategorik", "biner"],
                "Keterangan": [
                    "ID unik pasien",
                    "Male / Female / Other",
                    "Usia pasien (tahun)",
                    "1 = ada riwayat hipertensi",
                    "1 = ada riwayat penyakit jantung",
                    "Yes / No",
                    "Jenis pekerjaan",
                    "Urban / Rural",
                    "Rata-rata kadar gula darah (mg/dL)",
                    "Body Mass Index",
                    "Status merokok",
                    "TARGET — 1 = Stroke",
                ],
                "Status": ["❌ Dihapus", "⚠️ Other dihapus", "✅ Penting", "✅ Faktor Risiko",
                           "✅ Faktor Risiko", "✅", "✅", "✅",
                           "✅ Penting", "⚠️ 201 missing", "✅", "🎯 Target"],
            }
            st.dataframe(pd.DataFrame(dict_data), use_container_width=True, hide_index=True)

        # ── Sample Data ────────────────────────────────────────────────────
        st.markdown("<div class='section-header'>📄 5 Baris Pertama Dataset</div>", unsafe_allow_html=True)
        st.dataframe(df.head(), use_container_width=True)

        # ── Descriptive Stats ──────────────────────────────────────────────
        st.markdown("<div class='section-header'>📐 Statistik Deskriptif</div>", unsafe_allow_html=True)
        st.dataframe(df.describe().round(3), use_container_width=True)

        # ── Missing Values ──────────────────────────────────────────────────
        st.markdown("<div class='section-header'>🔍 Analisis Missing Values & Distribusi</div>",
                    unsafe_allow_html=True)

        col_mv, col_target = st.columns(2)

        with col_mv:
            mv = df.isnull().sum()
            mv = mv[mv > 0]
            if len(mv) > 0:
                fig_mv = px.bar(
                    x=mv.index, y=mv.values,
                    labels={"x": "Kolom", "y": "Jumlah Missing"},
                    title="Missing Values per Kolom",
                    color=mv.values,
                    color_continuous_scale="Reds",
                )
                fig_mv.update_layout(showlegend=False, height=300,
                                     margin=dict(t=40, b=20, l=10, r=10))
                st.plotly_chart(fig_mv, use_container_width=True)
            else:
                st.success("✅ Tidak ada missing values pada dataset ini.")

        with col_target:
            if "stroke" in df.columns:
                vc = df["stroke"].value_counts()
                fig_pie = px.pie(
                    values=vc.values,
                    names=["Tidak Stroke", "Stroke"],
                    title="Distribusi Target Variable (Stroke)",
                    color_discrete_sequence=["#3498db", "#e74c3c"],
                    hole=0.4,
                )
                fig_pie.update_traces(textinfo="percent+label")
                fig_pie.update_layout(height=300, margin=dict(t=40, b=10, l=10, r=10))
                st.plotly_chart(fig_pie, use_container_width=True)

        # ── Correlation Heatmap ────────────────────────────────────────────
        st.markdown("<div class='section-header'>🌡️ Heatmap Korelasi</div>", unsafe_allow_html=True)

        df_num = df.copy()
        from sklearn.preprocessing import LabelEncoder
        for c in df_num.select_dtypes("object").columns:
            df_num[c] = LabelEncoder().fit_transform(df_num[c].astype(str))
        df_num = df_num.drop(columns=["id"], errors="ignore")
        corr = df_num.corr().round(2)

        fig_heat = px.imshow(
            corr, text_auto=True, aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            title="Korelasi Antar Fitur"
        )
        fig_heat.update_layout(height=500, margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig_heat, use_container_width=True)

        # ── Train Model Button ─────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### ⚙️ Latih Model")
        st.markdown("Klik tombol di bawah untuk melatih model Random Forest & K-Means "
                    "menggunakan dataset yang sudah diupload.")

        if st.button("🚀 Latih Model Sekarang", type="primary", use_container_width=True):
            with st.spinner("Memproses data dan melatih model... (30-60 detik)"):
                try:
                    X_tr, X_te, y_tr, y_te, scaler, le_dict, feat_names = prepare_data(df)
                    rf_model, km_model = train_models(X_tr, y_tr)
                    st.session_state.rf_model = rf_model
                    st.session_state.kmeans_model = km_model
                    st.session_state.scaler = scaler
                    st.session_state.le_dict = le_dict
                    st.session_state.feature_names = feat_names
                    st.session_state.X_test = X_te
                    st.session_state.y_test = y_te
                    st.session_state.model_ready = True
                    st.success("✅ Model berhasil dilatih! Silakan gunakan menu Prediction & Visualization.")
                except Exception as e:
                    st.error(f"❌ Error saat melatih model: {str(e)}")
    else:
        st.markdown("""
        <div class='warn-card'>
            <b>⚠️ Belum ada dataset.</b><br>
            Upload file CSV atau klik <b>Gunakan Sample Data</b> untuk memulai.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PREDICTION / ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮  Prediction / Analysis":
    st.title("🔮 Prediction / Analysis")

    if not st.session_state.model_ready:
        st.warning("⚠️ Model belum dilatih. Silakan pergi ke **Dataset Overview** dan latih model terlebih dahulu.")
        st.stop()

    st.markdown("Masukkan data profil kesehatan pasien untuk mendapatkan prediksi risiko stroke.")
    st.markdown("---")

    # ── Input Form ────────────────────────────────────────────────────────
    with st.form("prediction_form"):
        st.markdown("### 📝 Form Input Data Pasien")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**📊 Informasi Demografis**")
            gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
            age = st.slider("Usia (tahun)", 1, 100, 45)
            ever_married = st.selectbox("Status Pernikahan", ["Yes", "No"])
            residence = st.selectbox("Tipe Tempat Tinggal", ["Urban", "Rural"])

        with col2:
            st.markdown("**🏥 Riwayat Medis**")
            hypertension = st.selectbox("Riwayat Hipertensi", [0, 1],
                                        format_func=lambda x: "Ya (1)" if x == 1 else "Tidak (0)")
            heart_disease = st.selectbox("Riwayat Penyakit Jantung", [0, 1],
                                         format_func=lambda x: "Ya (1)" if x == 1 else "Tidak (0)")
            avg_glucose = st.number_input("Rata-rata Kadar Gula Darah (mg/dL)",
                                          min_value=55.0, max_value=300.0, value=100.0, step=0.1)
            bmi_val = st.number_input("BMI (Body Mass Index)",
                                      min_value=10.0, max_value=60.0, value=25.0, step=0.1)

        with col3:
            st.markdown("**💼 Gaya Hidup**")
            work_type = st.selectbox("Jenis Pekerjaan",
                                     ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
            smoking = st.selectbox("Status Merokok",
                                   ["never smoked", "formerly smoked", "smokes", "Unknown"])

            st.markdown("**ℹ️ Referensi Nilai Normal**")
            st.info("🍬 Gula Normal: 70–130 mg/dL\n\n⚖️ BMI Normal: 18.5–24.9")

        submitted = st.form_submit_button("🔍 Analisis Risiko Stroke", type="primary",
                                          use_container_width=True)

    # ── Prediction Logic ──────────────────────────────────────────────────
    if submitted:
        from sklearn.preprocessing import LabelEncoder

        rf_model = st.session_state.rf_model
        km_model = st.session_state.kmeans_model
        scaler   = st.session_state.scaler
        le_dict  = st.session_state.le_dict
        feature_names = st.session_state.feature_names

        # Build input
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
                try:
                    inp[col] = le.transform(inp[col].astype(str))
                except ValueError:
                    inp[col] = 0

        inp = inp[feature_names]
        inp_sc = scaler.transform(inp)

        pred_cls  = rf_model.predict(inp_sc)[0]
        pred_prob = rf_model.predict_proba(inp_sc)[0]
        cluster_p = km_model.predict(inp_sc)[0]
        prob_stroke = pred_prob[1]

        risk_level, risk_color, risk_icon, risk_class = get_risk_level(prob_stroke)

        st.markdown("---")
        st.markdown("### 🎯 Hasil Prediksi")

        col_res, col_detail = st.columns([2, 3])

        with col_res:
            st.markdown(f"""
            <div class='{risk_class}'>
                <div style='font-size:3rem;'>{risk_icon}</div>
                <h2 style='margin:8px 0 4px 0; color:white;'>RISIKO {risk_level}</h2>
                <h3 style='color:rgba(255,255,255,0.85); margin:0;'>
                    Probabilitas Stroke: {prob_stroke:.1%}
                </h3>
                <p style='margin:8px 0 0 0; color:rgba(255,255,255,0.7);'>
                    Cluster: Kelompok {cluster_p}
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col_detail:
            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob_stroke * 100,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Probabilitas Risiko Stroke (%)", "font": {"size": 14}},
                number={"suffix": "%", "font": {"size": 28}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar": {"color": risk_color},
                    "steps": [
                        {"range": [0, 30], "color": "#e8f5e9"},
                        {"range": [30, 60], "color": "#fff3e0"},
                        {"range": [60, 100], "color": "#ffebee"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 3},
                        "thickness": 0.75,
                        "value": prob_stroke * 100,
                    },
                },
            ))
            fig_gauge.update_layout(height=250, margin=dict(t=30, b=10, l=20, r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        # ── Feature Contribution Bar ───────────────────────────────────────
        st.markdown("### 📊 Kontribusi Faktor Risiko (Feature Importance)")

        fi = dict(zip(feature_names, rf_model.feature_importances_))
        fi_sorted = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:8]
        feat_labels = [f[0] for f in fi_sorted]
        feat_vals   = [f[1] for f in fi_sorted]

        # Highlight user's risky features
        risky = []
        if age > 60:         risky.append("age")
        if hypertension == 1: risky.append("hypertension")
        if heart_disease == 1: risky.append("heart_disease")
        if avg_glucose > 150: risky.append("avg_glucose_level")
        if bmi_val > 30:      risky.append("bmi")

        bar_colors = ["#e74c3c" if f in risky else "#3498db" for f in feat_labels]

        fig_fi = go.Figure(go.Bar(
            x=feat_vals, y=feat_labels, orientation="h",
            marker_color=bar_colors,
        ))
        fig_fi.update_layout(
            title="Feature Importance (Merah = Faktor Risiko Pasien Ini)",
            xaxis_title="Importance Score",
            height=320,
            margin=dict(t=40, b=20, l=10, r=20),
        )
        st.plotly_chart(fig_fi, use_container_width=True)

        # ── Recommendations ─────────────────────────────────────────────────
        st.markdown("### 💡 Rekomendasi")
        recs = []
        if age > 60:
            recs.append("🔵 **Usia > 60 tahun** — lakukan pemeriksaan kesehatan rutin minimal setahun sekali.")
        if hypertension == 1:
            recs.append("🔴 **Hipertensi** — pantau tekanan darah secara rutin dan patuhi pengobatan.")
        if heart_disease == 1:
            recs.append("🔴 **Penyakit Jantung** — konsultasikan dengan dokter spesialis jantung.")
        if avg_glucose > 150:
            recs.append("🟠 **Kadar Gula Tinggi** — kurangi konsumsi gula, evaluasi kemungkinan diabetes.")
        if bmi_val > 30:
            recs.append("🟡 **BMI Obesitas** — terapkan pola makan sehat dan olahraga rutin.")
        if smoking in ["smokes", "formerly smoked"]:
            recs.append("🟡 **Status Merokok** — berhenti merokok dapat menurunkan risiko stroke secara signifikan.")

        if not recs:
            st.success("✅ Profil kesehatan Anda tergolong baik. Pertahankan gaya hidup sehat!")
        else:
            for r in recs:
                st.markdown(r)

        st.markdown("""
        <div class='warn-card'>
            <b>⚠️ Disclaimer:</b> Prediksi ini hanya bersifat informatif dan tidak menggantikan
            diagnosis medis profesional. Konsultasikan dengan dokter untuk evaluasi lebih lanjut.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈  Visualization":
    st.title("📈 Visualization")

    if st.session_state.df is None:
        st.warning("⚠️ Belum ada dataset. Pergi ke **Dataset Overview** untuk upload atau gunakan sample data.")
        st.stop()

    df = st.session_state.df

    tab1, tab2, tab3, tab4 = st.tabs(["📊 EDA", "🔵 Clustering", "🌲 Klasifikasi", "📉 Evaluasi Model"])

    # ── TAB 1: EDA ─────────────────────────────────────────────────────────
    with tab1:
        st.markdown("#### Distribusi Fitur Numerik berdasarkan Status Stroke")

        num_col = st.selectbox("Pilih fitur numerik:", ["age", "avg_glucose_level", "bmi"])
        if num_col in df.columns and "stroke" in df.columns:
            fig_dist = px.histogram(
                df, x=num_col, color="stroke",
                barmode="overlay", nbins=40,
                color_discrete_map={0: "#3498db", 1: "#e74c3c"},
                labels={"stroke": "Status Stroke", num_col: num_col},
                title=f"Distribusi {num_col} berdasarkan Status Stroke",
            )
            fig_dist.update_layout(height=350, margin=dict(t=40, b=20))
            st.plotly_chart(fig_dist, use_container_width=True)

        st.markdown("#### Distribusi Fitur Kategorik vs Stroke Rate")
        cat_col = st.selectbox("Pilih fitur kategorik:",
                               ["gender", "hypertension", "heart_disease",
                                "ever_married", "work_type", "smoking_status"])
        if cat_col in df.columns and "stroke" in df.columns:
            ct = df.groupby(cat_col)["stroke"].agg(["sum", "count"]).reset_index()
            ct.columns = [cat_col, "stroke_count", "total"]
            ct["stroke_rate"] = (ct["stroke_count"] / ct["total"] * 100).round(2)
            fig_cat = px.bar(
                ct, x=cat_col, y="stroke_rate",
                color="stroke_rate",
                color_continuous_scale="Reds",
                title=f"Stroke Rate (%) per Kategori {cat_col}",
                labels={"stroke_rate": "Stroke Rate (%)"},
                text="stroke_rate",
            )
            fig_cat.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_cat.update_layout(height=350, margin=dict(t=40, b=20), showlegend=False)
            st.plotly_chart(fig_cat, use_container_width=True)

        st.markdown("#### Scatter Plot: Usia vs Glukosa")
        if all(c in df.columns for c in ["age", "avg_glucose_level", "stroke"]):
            sample = df.sample(min(2000, len(df)), random_state=42)
            fig_sc = px.scatter(
                sample, x="age", y="avg_glucose_level",
                color=sample["stroke"].map({0: "Tidak Stroke", 1: "Stroke"}),
                color_discrete_map={"Tidak Stroke": "#3498db", "Stroke": "#e74c3c"},
                opacity=0.6, title="Usia vs Rata-rata Kadar Gula (Stroke vs Tidak)",
                labels={"avg_glucose_level": "Rata-rata Gula (mg/dL)", "age": "Usia"},
            )
            fig_sc.update_layout(height=400, margin=dict(t=40, b=20))
            st.plotly_chart(fig_sc, use_container_width=True)

    # ── TAB 2: Clustering ─────────────────────────────────────────────────
    with tab2:
        if not st.session_state.model_ready:
            st.info("💡 Latih model terlebih dahulu di halaman Dataset Overview.")
        else:
            df2 = df.copy()
            from sklearn.preprocessing import LabelEncoder, StandardScaler
            for c in df2.select_dtypes("object").columns:
                df2[c] = LabelEncoder().fit_transform(df2[c].astype(str))
            df2 = df2.drop(columns=["id", "stroke"], errors="ignore")
            df2 = df2.fillna(df2.median())

            sc2 = st.session_state.scaler
            km2 = st.session_state.kmeans_model
            fn2 = st.session_state.feature_names

            df2 = df2[fn2] if all(c in df2.columns for c in fn2) else df2.iloc[:, :len(fn2)]
            X_sc2 = sc2.transform(df2)
            labels = km2.predict(X_sc2)

            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, random_state=42)
            pcs = pca.fit_transform(X_sc2)

            pca_df = pd.DataFrame({"PC1": pcs[:, 0], "PC2": pcs[:, 1],
                                   "Cluster": labels.astype(str)})

            st.markdown("#### Visualisasi Cluster (PCA 2D)")
            fig_pca = px.scatter(
                pca_df, x="PC1", y="PC2", color="Cluster",
                color_discrete_sequence=["#3498db", "#e74c3c", "#2ecc71"],
                opacity=0.6,
                title=f"K-Means Clustering (k=3) — PCA 2D "
                      f"({pca.explained_variance_ratio_.sum()*100:.1f}% variance)",
            )
            fig_pca.update_layout(height=400, margin=dict(t=40, b=20))
            st.plotly_chart(fig_pca, use_container_width=True)

            # Cluster stats
            st.markdown("#### Statistik per Cluster")
            orig_df = df.drop(columns=["id"], errors="ignore").copy()
            for c in orig_df.select_dtypes("object").columns:
                orig_df[c] = LabelEncoder().fit_transform(orig_df[c].astype(str))
            orig_df = orig_df.fillna(orig_df.median())
            orig_df["Cluster"] = labels

            num_feats = [c for c in ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]
                         if c in orig_df.columns]
            cluster_stats = orig_df.groupby("Cluster")[num_feats].mean().round(2)

            if "stroke" in df.columns:
                cluster_stats["Stroke Rate (%)"] = (
                    orig_df.groupby("Cluster")["stroke"].mean() * 100
                ).round(2)

            st.dataframe(cluster_stats, use_container_width=True)

            # Radar
            st.markdown("#### Radar Chart Profil Cluster")
            radar_feats = [c for c in ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]
                           if c in orig_df.columns]
            c_avg = orig_df.groupby("Cluster")[radar_feats].mean().copy()
            for col in radar_feats:
                rng = c_avg[col].max() - c_avg[col].min() + 1e-10
                c_avg[col] = (c_avg[col] - c_avg[col].min()) / rng

            angles = radar_feats + [radar_feats[0]]
            fig_radar = go.Figure()
            colors_r = ["#3498db", "#e74c3c", "#2ecc71"]
            for i, row in c_avg.iterrows():
                vals = row.values.tolist() + [row.values[0]]
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals, theta=angles, fill="toself",
                    name=f"Cluster {i}", line_color=colors_r[i],
                    opacity=0.7,
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title="Radar Chart — Profil Cluster (Ternormalisasi 0-1)",
                height=420, margin=dict(t=60, b=20),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

    # ── TAB 3: Klasifikasi ────────────────────────────────────────────────
    with tab3:
        if not st.session_state.model_ready:
            st.info("💡 Latih model terlebih dahulu di halaman Dataset Overview.")
        else:
            rf_model = st.session_state.rf_model
            feature_names = st.session_state.feature_names

            st.markdown("#### Feature Importance — Random Forest")
            fi = dict(zip(feature_names, rf_model.feature_importances_))
            fi_df = pd.DataFrame(fi.items(), columns=["Fitur", "Importance"]).sort_values(
                "Importance", ascending=True
            )
            fig_fi = px.bar(fi_df, x="Importance", y="Fitur", orientation="h",
                            color="Importance", color_continuous_scale="Blues",
                            title="Feature Importance — Random Forest",
                            labels={"Importance": "Importance Score"})
            fig_fi.update_layout(height=400, showlegend=False, margin=dict(t=40, b=20, l=10, r=20))
            st.plotly_chart(fig_fi, use_container_width=True)

            # Age distribution vs feature importance context
            st.markdown("#### Perbandingan Importance: Random Forest vs Feature Korelasi Target")
            if "stroke" in df.columns:
                df_corr2 = df.copy()
                for c in df_corr2.select_dtypes("object").columns:
                    df_corr2[c] = LabelEncoder().fit_transform(df_corr2[c].astype(str))
                df_corr2 = df_corr2.drop(columns=["id"], errors="ignore").fillna(df_corr2.median())
                corr_stroke = df_corr2.corr()["stroke"].drop("stroke").abs().sort_values(ascending=False)

                comp_df = pd.DataFrame({
                    "Fitur": corr_stroke.index,
                    "Korelasi (|r|)": corr_stroke.values,
                    "RF Importance": [fi.get(f, 0) for f in corr_stroke.index],
                })
                fig_comp = go.Figure()
                fig_comp.add_bar(name="Korelasi (|r|)", x=comp_df["Fitur"],
                                 y=comp_df["Korelasi (|r|)"], marker_color="#3498db")
                fig_comp.add_bar(name="RF Importance", x=comp_df["Fitur"],
                                 y=comp_df["RF Importance"], marker_color="#e74c3c")
                fig_comp.update_layout(barmode="group", height=380,
                                       title="Korelasi vs RF Importance per Fitur",
                                       margin=dict(t=40, b=20))
                st.plotly_chart(fig_comp, use_container_width=True)

    # ── TAB 4: Evaluasi ───────────────────────────────────────────────────
    with tab4:
        if not st.session_state.model_ready:
            st.info("💡 Latih model terlebih dahulu di halaman Dataset Overview.")
        else:
            from sklearn.metrics import (confusion_matrix, accuracy_score,
                                         precision_score, recall_score,
                                         f1_score, roc_auc_score, roc_curve)
            from sklearn.tree import DecisionTreeClassifier

            rf_model = st.session_state.rf_model
            X_test   = st.session_state.X_test
            y_test   = st.session_state.y_test
            X_tr_sm  = None

            # Re-train DT quickly for comparison
            with st.spinner("Melatih Decision Tree untuk perbandingan..."):
                try:
                    df_prep = df.copy()
                    df_prep = df_prep.drop(columns=["id"], errors="ignore")
                    df_prep = df_prep[df_prep.get("gender", pd.Series(["Male"])) != "Other"]
                    for c in df_prep.select_dtypes("object").columns:
                        df_prep[c] = LabelEncoder().fit_transform(df_prep[c].astype(str))
                    df_prep = df_prep.fillna(df_prep.median())
                    feat_names = st.session_state.feature_names
                    X_all = df_prep[feat_names]
                    y_all = df_prep["stroke"]
                    from sklearn.model_selection import train_test_split
                    from imblearn.over_sampling import SMOTE
                    X_tr2, _, y_tr2, _ = train_test_split(X_all, y_all, test_size=0.2,
                                                          random_state=42, stratify=y_all)
                    sc2 = st.session_state.scaler
                    X_tr2_sc = sc2.transform(X_tr2)
                    sm2 = SMOTE(random_state=42)
                    X_tr2_sm, y_tr2_sm = sm2.fit_resample(X_tr2_sc, y_tr2)

                    dt_model = DecisionTreeClassifier(max_depth=8, min_samples_split=10,
                                                      min_samples_leaf=5, random_state=42)
                    dt_model.fit(X_tr2_sm, y_tr2_sm)
                except Exception:
                    dt_model = None

            y_pred_rf = rf_model.predict(X_test)
            y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

            st.markdown("#### Confusion Matrix — Random Forest")
            cm_rf = confusion_matrix(y_test, y_pred_rf)
            fig_cm = px.imshow(
                cm_rf, text_auto=True, aspect="auto",
                color_continuous_scale="Blues",
                labels={"x": "Prediksi", "y": "Aktual"},
                x=["Tidak Stroke", "Stroke"], y=["Tidak Stroke", "Stroke"],
                title="Confusion Matrix — Random Forest",
            )
            fig_cm.update_layout(height=350, margin=dict(t=40, b=20))
            st.plotly_chart(fig_cm, use_container_width=True)

            # Metrics
            st.markdown("#### Metrik Evaluasi")
            metrics = {
                "Accuracy":  accuracy_score(y_test, y_pred_rf),
                "Precision": precision_score(y_test, y_pred_rf, zero_division=0),
                "Recall":    recall_score(y_test, y_pred_rf),
                "F1-Score":  f1_score(y_test, y_pred_rf, zero_division=0),
                "ROC-AUC":   roc_auc_score(y_test, y_prob_rf),
            }
            m_cols = st.columns(5)
            for (k, v), col in zip(metrics.items(), m_cols):
                col.metric(k, f"{v:.4f}", delta=f"{'★ Prioritas' if k=='Recall' else ''}")

            # ROC Curve
            st.markdown("#### ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, y_prob_rf)
            auc_rf = roc_auc_score(y_test, y_prob_rf)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                         name=f"Random Forest (AUC={auc_rf:.4f})",
                                         line=dict(color="#3498db", width=3)))
            if dt_model:
                y_prob_dt = dt_model.predict_proba(X_test)[:, 1]
                fpr_dt, tpr_dt, _ = roc_curve(y_test, y_prob_dt)
                auc_dt = roc_auc_score(y_test, y_prob_dt)
                fig_roc.add_trace(go.Scatter(x=fpr_dt, y=tpr_dt, mode="lines",
                                             name=f"Decision Tree (AUC={auc_dt:.4f})",
                                             line=dict(color="#e74c3c", width=3, dash="dash")))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                         name="Random (AUC=0.50)",
                                         line=dict(color="gray", width=2, dash="dot")))
            fig_roc.update_layout(
                title="ROC Curve — Perbandingan Model",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate (Recall)",
                height=400, margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig_roc, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️  About":
    st.title("ℹ️ About")

    tab_method, tab_dataset, tab_project = st.tabs(["🔬 Metode", "📁 Dataset", "📌 Informasi Proyek"])

    with tab_method:
        st.markdown("### 🔬 Penjelasan Metode Data Mining")

        with st.expander("🔵 K-Means Clustering (Unsupervised Learning)", expanded=True):
            st.markdown("""
            **K-Means Clustering** adalah algoritma unsupervised yang mengelompokkan data ke dalam
            *k* cluster berdasarkan kesamaan fitur, meminimalkan jarak intra-cluster (Within-Cluster
            Sum of Squares / WCSS).

            **Cara Kerja:**
            1. Inisialisasi *k* centroid secara acak
            2. Setiap data point ditetapkan ke centroid terdekat (jarak Euclidean)
            3. Hitung ulang centroid berdasarkan rata-rata anggota cluster
            4. Ulangi langkah 2–3 hingga konvergen

            **Pemilihan K Optimal** menggunakan:
            - 📐 **Elbow Method** — cari titik "siku" pada grafik WCSS vs K
            - 📊 **Silhouette Score** — ukuran kekompakan dan separasi cluster (0–1, semakin tinggi semakin baik)
            - 📉 **Davies-Bouldin Score** — semakin rendah semakin baik

            **Output di Proyek Ini:** 3 cluster pasien berdasarkan profil kesehatan (risiko rendah, menengah, tinggi)
            """)

        with st.expander("🌲 Random Forest (Model Utama)", expanded=True):
            st.markdown("""
            **Random Forest** adalah algoritma ensemble yang membangun banyak Decision Tree secara
            paralel (bagging) dan menggabungkan hasil prediksi melalui voting mayoritas.

            **Keunggulan vs Decision Tree Tunggal:**
            - ✅ Lebih robust terhadap overfitting (averaging banyak pohon)
            - ✅ Feature importance yang lebih stabil
            - ✅ Tidak sensitif terhadap outlier

            **Hyperparameter yang Digunakan:**
            | Parameter | Nilai | Alasan |
            |-----------|-------|--------|
            | n_estimators | 100 | 100 pohon, keseimbangan kecepatan-akurasi |
            | max_depth | 10 | Mencegah overfitting |
            | min_samples_split | 5 | Minimum sampel sebelum split |
            | min_samples_leaf | 2 | Minimum sampel di daun |
            """)

        with st.expander("🟡 Decision Tree (Model Pembanding)"):
            st.markdown("""
            **Decision Tree** membangun model berupa pohon aturan IF-THEN yang mudah diinterpretasi
            secara visual. Digunakan sebagai model pembanding untuk membuktikan superioritas ensemble.

            **Perbedaan dengan Random Forest:**
            | Aspek | Decision Tree | Random Forest |
            |-------|--------------|---------------|
            | Tipe | Single tree | Ensemble (100 trees) |
            | Interpretasi | Mudah (whitebox) | Sulit (blackbox) |
            | Overfitting | Rentan | Lebih robust |
            | Akurasi | Lebih rendah | Lebih tinggi |
            """)

        with st.expander("📐 Penanganan Imbalanced Data — SMOTE"):
            st.markdown("""
            Dataset stroke sangat **tidak seimbang** (~95% tidak stroke, ~5% stroke).
            Tanpa penanganan, model akan bias memprediksi semua data sebagai "tidak stroke".

            **SMOTE (Synthetic Minority Oversampling Technique):**
            - Membuat data sintetis untuk kelas minoritas (stroke)
            - Interpolasi antara sampel minoritas yang ada
            - Diterapkan **hanya pada training data** (mencegah data leakage)

            **Pipeline yang Benar:**
            ```
            Train-Test Split → StandardScaler (fit train only) → SMOTE (train only)
            ```
            """)

        with st.expander("🔍 SHAP Explainable AI (Bonus)"):
            st.markdown("""
            **SHAP (SHapley Additive exPlanations)** menggunakan teori game theory untuk
            mengukur kontribusi setiap fitur terhadap prediksi individual.

            **Jenis Plot SHAP:**
            - **Summary Plot (Beeswarm)** — distribusi pengaruh semua fitur
            - **Bar Plot** — rata-rata kontribusi absolut tiap fitur
            - **Waterfall Plot** — penjelasan prediksi untuk 1 pasien spesifik

            **Cara Membaca:**
            - SHAP positif → mendorong prediksi ke arah **STROKE**
            - SHAP negatif → mendorong prediksi ke arah **TIDAK STROKE**
            """)

        with st.expander("📏 Metrik Evaluasi"):
            st.markdown("""
            | Metrik | Penjelasan | Prioritas |
            |--------|-----------|-----------|
            | **Accuracy** | Proporsi prediksi benar dari total | ⚠️ Misleading pada data imbalanced |
            | **Precision** | Dari yang diprediksi stroke, berapa yang benar? | Sedang |
            | **Recall** | Dari kasus stroke nyata, berapa yang terdeteksi? | ⭐ **TINGGI** |
            | **F1-Score** | Harmonic mean Precision & Recall | Tinggi |
            | **ROC-AUC** | Kemampuan membedakan dua kelas (0-1) | Tinggi |

            > **Recall diprioritaskan** karena dalam konteks medis, *False Negative*
            (stroke tidak terdeteksi) jauh lebih berbahaya dari *False Positive*.
            """)

    with tab_dataset:
        st.markdown("### 📁 Informasi Dataset")
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("""
            **Nama Dataset:** Stroke Prediction Dataset

            **Sumber:** Kaggle — [fedesoriano](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

            **Jumlah Records:** 5.110 baris

            **Jumlah Fitur:** 12 kolom (11 fitur + 1 target)

            **Target Variable:** `stroke` (biner: 0 = tidak stroke, 1 = stroke)

            **Distribusi Target:**
            - Tidak Stroke: ~4.861 (95.1%)
            - Stroke: ~249 (4.9%)
            - Rasio ketidakseimbangan: ~20:1
            """)
        with col_r:
            st.markdown("""
            **Missing Values:**
            - Kolom `bmi`: 201 nilai kosong (3.9%)
            - Penanganan: diisi dengan **median** (lebih robust dari mean terhadap outlier)

            **Preprocessing Steps:**
            1. Hapus kolom `id` (tidak relevan)
            2. Hapus baris `gender='Other'` (hanya 1 baris, noise)
            3. Imputasi `bmi` dengan median
            4. Label Encoding untuk fitur kategorik
            5. Train-Test Split (80:20, stratified)
            6. StandardScaler (fit hanya pada train)
            7. SMOTE oversampling (hanya pada train)

            **Lisensi:** Open Database License (ODbL)
            """)

    with tab_project:
        st.markdown("### 📌 Informasi Proyek")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **📚 Mata Kuliah:** Data Mining

            **📅 Semester:** Genap 2024/2025

            **🏛️ Program Studi:** Informatika / Ilmu Komputer

            **📋 Framework:** CRISP-DM

            **🛠️ Tools & Libraries:**
            - Python 3.x
            - Streamlit (Web App)
            - Scikit-learn (ML)
            - Imbalanced-learn (SMOTE)
            - SHAP (Explainability)
            - Plotly (Visualisasi Interaktif)
            """)
        with col2:
            st.markdown("""
            **✅ Checklist CRISP-DM:**

            | Fase | Status |
            |------|--------|
            | Business Understanding | ✅ |
            | Data Understanding | ✅ |
            | Data Preparation | ✅ |
            | Modeling — Clustering | ✅ |
            | Modeling — Classification | ✅ |
            | Evaluation | ✅ |
            | Deployment (Web App) | ✅ |
            """)

        st.markdown("---")
        st.markdown("### 👥 Tim Proyek")
        members = [
            ("👨‍💻", "Nama Anggota 1", "NIM: 22XXXXXXX", "Ketua — Modeling & Evaluasi"),
            ("👩‍💻", "Nama Anggota 2", "NIM: 22XXXXXXX", "EDA & Visualisasi"),
            ("👨‍💻", "Nama Anggota 3", "NIM: 22XXXXXXX", "Preprocessing & SMOTE"),
            ("👩‍💻", "Nama Anggota 4", "NIM: 22XXXXXXX", "Web App & Deployment"),
        ]
        cols = st.columns(4)
        for col, (avatar, name, nim, role) in zip(cols, members):
            with col:
                st.markdown(f"""
                <div class='team-card'>
                    <div class='avatar'>{avatar}</div>
                    <div class='name'>{name}</div>
                    <div class='nim'>{nim}</div>
                    <div class='nim' style='margin-top:4px; color:#a8dadc;'>{role}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div style='text-align:center; color:#888; font-size:13px; padding:16px;'>
            🧠 Deteksi Dini Risiko Stroke — Proyek Data Mining 2024/2025<br>
            Dibuat menggunakan <b>Streamlit</b> + <b>Scikit-learn</b> + <b>Plotly</b>
        </div>
        """, unsafe_allow_html=True)
