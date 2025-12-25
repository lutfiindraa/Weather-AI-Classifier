import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
import colorsys
import gdown # TAMBAHAN: Import library gdown

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Weather AI Classifier",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- TAMBAHAN: KONFIGURASI ID GOOGLE DRIVE ---
# Masukkan ID File (bukan ID Folder) dari link Google Drive masing-masing model
MODEL_DRIVE_IDS = {
    'model_base_cnn.keras': 'https://drive.google.com/file/d/1XS-me_zBmTh2897EAnDuE83aVCEjMKg4/view?usp=drive_link',       # Contoh: '1A2b3C...'
    'model_mobilenet.keras': 'https://drive.google.com/file/d/1eqO0NkDxHafFaA057rLP1-z2St-4P1XM/view?usp=drive_link',
    'model_resnet.keras': 'https://drive.google.com/file/d/1jIc9gCh0r-6jwS7oJGsrcH2Y8b2EWDS9/view?usp=drive_link'
}

# --- FUNGSI EKSTRAKSI WARNA DOMINAN ---
def get_dominant_colors(image, n_colors=3):
    """Ekstrak warna dominan dari gambar untuk tema dinamis"""
    img_small = image.resize((50, 50))
    pixels = np.array(img_small).reshape(-1, 3)
    
    # Simple clustering manual (ambil sample)
    unique_colors = np.unique(pixels, axis=0)
    if len(unique_colors) > n_colors:
        indices = np.random.choice(len(unique_colors), n_colors, replace=False)
        colors = unique_colors[indices]
    else:
        colors = unique_colors
    
    # Convert ke hex
    hex_colors = ['#%02x%02x%02x' % tuple(color) for color in colors]
    return hex_colors


def adjust_color_brightness(hex_color, factor=0.8):
    """Sesuaikan brightness warna untuk dark mode"""
    rgb = tuple(int(hex_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    h, l, s = colorsys.rgb_to_hls(rgb[0]/255, rgb[1]/255, rgb[2]/255)
    
    # Adjust lightness
    l = max(0, min(1, l * factor))
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    
    return '#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255))


# --- CSS CUSTOM DENGAN DARK MODE SUPPORT ---
st.markdown("""
    <style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Dark Mode Variables */
    :root {
        --bg-primary: #ffffff;
        --bg-secondary: #f8f9fa;
        --text-primary: #1a1a1a;
        --text-secondary: #6c757d;
        --border-color: #e9ecef;
        --card-shadow: rgba(0,0,0,0.08);
    }
    
    @media (prefers-color-scheme: dark) {
        :root {
            --bg-primary: #0e1117;
            --bg-secondary: #1a1d24;
            --text-primary: #fafafa;
            --text-secondary: #a0a0a0;
            --border-color: #2d3139;
            --card-shadow: rgba(0,0,0,0.3);
        }
    }
    
    .main {
        background: var(--bg-primary);
        color: var(--text-primary);
    }
    
    /* Header dengan Gradient */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px var(--card-shadow);
    }
    
    /* Card Modern */
    .metric-card {
        background: var(--bg-secondary);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid var(--border-color);
        box-shadow: 0 4px 15px var(--card-shadow);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin-bottom: 1rem;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px var(--card-shadow);
    }
    
    /* Dynamic Weather Card */
    .weather-result-card {
        background: var(--bg-secondary);
        border-radius: 20px;
        padding: 2rem;
        border-left: 5px solid;
        box-shadow: 0 10px 40px var(--card-shadow);
        backdrop-filter: blur(10px);
    }
    
    /* Button Custom */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Stats Container */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .stat-box {
        background: var(--bg-secondary);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
    }
    
    .stat-box:hover {
        transform: translateY(-3px);
        box-shadow: 0 5px 15px var(--card-shadow);
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .stat-label {
        font-size: 0.85rem;
        color: var(--text-secondary);
        margin-top: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Info Card dengan Icon */
    .info-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border-left: 4px solid #667eea;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    /* Progress Bar Custom */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: var(--bg-secondary);
        border-right: 1px solid var(--border-color);
    }
    
    /* Image Container */
    .image-container {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 10px 30px var(--card-shadow);
        border: 2px solid var(--border-color);
    }
    
    /* Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-in {
        animation: fadeIn 0.6s ease-out;
    }
    
    /* Badge */
    .confidence-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 0.5rem 0;
    }
    
    .badge-high {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
    }
    
    .badge-medium {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    
    .badge-low {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)


# --- DEFINISI KELAS & INFO CUACA LENGKAP ---
CLASS_NAMES = [
    'Dew', 'Fog/Smog', 'Frost', 'Glaze', 'Hail', 
    'Lightning', 'Rain', 'Rainbow', 'Rime', 'Sandstorm', 'Snow'
]


WEATHER_INFO = {
    'Dew': {
        'icon': '💧',
        'title': 'Embun (Dew)',
        'desc': 'Tetesan air yang terbentuk pada permukaan benda ketika uap air mengembun di pagi hari atau malam hari.',
        'tips': ['Kelembaban tinggi', 'Biasanya di pagi hari', 'Suhu permukaan dingin'],
        'color': '#4A90E2'
    },
    'Fog/Smog': {
        'icon': '🌫️',
        'title': 'Kabut/Asap (Fog/Smog)',
        'desc': 'Kondisi atmosfer dengan partikel air/asap yang mengurangi jarak pandang secara signifikan.',
        'tips': ['Berkendara hati-hati', 'Gunakan lampu kabut', 'Hindari aktivitas outdoor'],
        'color': '#95A5A6'
    },
    'Frost': {
        'icon': '❄️',
        'title': 'Embun Beku (Frost)',
        'desc': 'Lapisan es tipis yang terbentuk pada permukaan ketika suhu turun di bawah titik beku.',
        'tips': ['Suhu sangat dingin', 'Lindungi tanaman', 'Hati-hati jalanan licin'],
        'color': '#3498DB'
    },
    'Glaze': {
        'icon': '🧊',
        'title': 'Glaze Ice',
        'desc': 'Lapisan es bening dan halus yang melapisi permukaan benda, sangat licin dan berbahaya.',
        'tips': ['Sangat licin!', 'Hindari berkendara', 'Risiko jatuh tinggi'],
        'color': '#5DADE2'
    },
    'Hail': {
        'icon': '☄️',
        'title': 'Hujan Es (Hail)',
        'desc': 'Butiran es padat yang jatuh dari awan badai petir, dapat merusak properti dan kendaraan.',
        'tips': ['Cari perlindungan', 'Lindungi kendaraan', 'Jangan di luar ruangan'],
        'color': '#85C1E9'
    },
    'Lightning': {
        'icon': '⚡',
        'title': 'Petir (Lightning)',
        'desc': 'Pelepasan listrik alam yang terjadi antara awan dan tanah atau antar awan.',
        'tips': ['BAHAYA! Cari tempat aman', 'Hindari tempat tinggi', 'Jauhi pohon dan air'],
        'color': '#F39C12'
    },
    'Rain': {
        'icon': '🌧️',
        'title': 'Hujan (Rain)',
        'desc': 'Presipitasi air yang jatuh dari awan ke permukaan bumi dalam bentuk tetesan.',
        'tips': ['Bawa payung/jas hujan', 'Berkendara hati-hati', 'Waspadai genangan'],
        'color': '#5499C7'
    },
    'Rainbow': {
        'icon': '🌈',
        'title': 'Pelangi (Rainbow)',
        'desc': 'Fenomena optik dan meteorologi berupa spektrum cahaya yang muncul di langit.',
        'tips': ['Cuaca cerah setelah hujan', 'Moment foto indah', 'Pertanda baik'],
        'color': '#E74C3C'
    },
    'Rime': {
        'icon': '❄️',
        'title': 'Rime Ice',
        'desc': 'Endapan es putih dan kasar yang terbentuk ketika tetesan air super-dingin membeku.',
        'tips': ['Umum di pegunungan', 'Suhu sangat rendah', 'Pemandangan indah'],
        'color': '#AED6F1'
    },
    'Sandstorm': {
        'icon': '🌪️',
        'title': 'Badai Pasir (Sandstorm)',
        'desc': 'Fenomena cuaca dengan angin kencang yang membawa pasir dan debu dalam jumlah besar.',
        'tips': ['Gunakan masker', 'Lindungi mata', 'Tutup jendela rapat'],
        'color': '#D68910'
    },
    'Snow': {
        'icon': '☃️',
        'title': 'Salju (Snow)',
        'desc': 'Presipitasi dalam bentuk kristal es yang jatuh dari awan ketika suhu udara di bawah 0°C.',
        'tips': ['Berpakaian hangat', 'Jalanan licin', 'Aktivitas musim dingin'],
        'color': '#EBF5FB'
    }
}


# --- FUNGSI LOAD MODEL (DIMODIFIKASI UNTUK AUTO-DOWNLOAD) ---
@st.cache_resource
def load_prediction_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, 'model')
    
    # Buat folder model jika belum ada (penting untuk Streamlit Cloud)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Definisi path lokal
    path_base = os.path.join(model_dir, 'model_base_cnn.keras')
    path_mobile = os.path.join(model_dir, 'model_mobilenet.keras')
    path_resnet = os.path.join(model_dir, 'model_resnet.keras')
    
    # Logic Auto-Download dengan gdown
    files_to_check = {
        path_base: MODEL_DRIVE_IDS.get('model_base_cnn.keras'),
        path_mobile: MODEL_DRIVE_IDS.get('model_mobilenet.keras'),
        path_resnet: MODEL_DRIVE_IDS.get('model_resnet.keras')
    }

    try:
        # Cek file satu per satu, download jika tidak ada
        for file_path, drive_id in files_to_check.items():
            if not os.path.exists(file_path):
                if drive_id and drive_id != 'ISI_ID_FILE_DISINI': # Pastikan ID sudah diisi
                    file_name = os.path.basename(file_path)
                    st.info(f"📥 Mengunduh model {file_name} dari Google Drive... (Hanya sekali)")
                    url = f'https://drive.google.com/uc?id={drive_id}'
                    gdown.download(url, file_path, quiet=False)
                else:
                    st.warning(f"⚠️ Model {os.path.basename(file_path)} tidak ditemukan lokal dan ID Drive belum dikonfigurasi.")
        
        # Load Model
        models = {}
        with st.spinner('🔄 Memuat Model AI...'):
            if os.path.exists(path_base):
                models['CNN Base'] = tf.keras.models.load_model(path_base)
            if os.path.exists(path_mobile):
                models['MobileNetV2'] = tf.keras.models.load_model(path_mobile)
            if os.path.exists(path_resnet):
                models['ResNet50V2'] = tf.keras.models.load_model(path_resnet)
            
    except Exception as e:
        st.error(f"❌ Gagal memuat model. Error: {e}")
        return None
    
    return models


# --- FUNGSI PREDIKSI ---
def predict_image(model, image):
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    img_batch = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_batch)
    return predictions[0]


# --- SIDEBAR NAVIGASI MODERN ---
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 1rem 0;'>
            <h1 style='margin: 0; font-size: 2rem;'>🌦️</h1>
            <h2 style='margin: 0.5rem 0; color: #667eea;'>Weather AI</h2>
            <p style='color: var(--text-secondary); font-size: 0.85rem;'>Intelligent Weather Classification</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    page = st.radio(
        "📍 Navigasi",
        ["🔍 Single Prediction", "⚖️ Model Comparison", "📊 Batch Analysis", "ℹ️ About"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Info Box
    st.markdown("""
        <div class="info-card">
            <strong>💡 Quick Tip</strong><br>
            Upload gambar cuaca dengan resolusi tinggi untuk hasil prediksi terbaik!
        </div>
    """, unsafe_allow_html=True)
    
    # Stats
    st.markdown("### 📈 Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Models", "3", delta="Active")
    with col2:
        st.metric("Classes", "11", delta="Types")


# Load Models
models_dict = load_prediction_models()

if not models_dict:
    st.error("⚠️ Aplikasi berhenti karena model gagal dimuat.")
    st.stop()


# --- HALAMAN 1: SINGLE PREDICTION ---
if page == "🔍 Single Prediction":
    st.markdown("""
        <div class="main-header animate-in">
            <h1 style='margin: 0; font-size: 2.5rem;'>🔍 Single Weather Prediction</h1>
            <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>Upload gambar dan biarkan AI mengidentifikasi kondisi cuaca</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1.5], gap="large")
    
    with col1:
        st.markdown("### ⚙️ Configuration")
        
        selected_model_name = st.selectbox(
            "Pilih Model AI:",
            list(models_dict.keys()),
            index=1 if 'MobileNetV2' in models_dict else 0,
            help="MobileNetV2 direkomendasikan untuk performa terbaik"
        )
        
        uploaded_file = st.file_uploader(
            "Upload Gambar Cuaca",
            type=["jpg", "png", "jpeg"],
            help="Format: JPG, PNG, JPEG (Max 200MB)"
        )
        
        if uploaded_file:
            st.markdown("---")
            show_details = st.checkbox("🔬 Tampilkan Analisis Detail", value=True)
            show_all_probs = st.checkbox("📊 Tampilkan Semua Probabilitas", value=False)
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        
        # Ekstrak warna dominan
        dominant_colors = get_dominant_colors(image)
        
        with col1:
            st.markdown("### 🖼️ Input Image")
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Image Info
            st.caption(f"📏 Size: {image.size[0]}x{image.size[1]} px")
            st.caption(f"📦 Format: {image.format}")
        
        with col2:
            if models_dict:
                model = models_dict[selected_model_name]
                
                with st.spinner('🤖 AI sedang menganalisis...'):
                    probs = predict_image(model, image)
                
                top_idx = np.argmax(probs)
                top_prob = probs[top_idx] * 100
                top_label = CLASS_NAMES[top_idx]
                weather_data = WEATHER_INFO[top_label]
                
                # Confidence Badge
                if top_prob >= 80:
                    badge_class = "badge-high"
                    conf_text = "Confidence Tinggi"
                elif top_prob >= 50:
                    badge_class = "badge-medium"
                    conf_text = "Confidence Sedang"
                else:
                    badge_class = "badge-low"
                    conf_text = "Confidence Rendah"
                
                # Main Result Card dengan warna dinamis
                st.markdown(f"""
                    <div class="weather-result-card animate-in" style="border-left-color: {weather_data['color']};">
                        <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
                            <div style="font-size: 3rem;">{weather_data['icon']}</div>
                            <div>
                                <h2 style="margin: 0; color: {weather_data['color']};">{top_label}</h2>
                                <p style="margin: 0; color: var(--text-secondary);">{weather_data['title']}</p>
                            </div>
                        </div>
                        <div class="confidence-badge {badge_class}">{conf_text}: {top_prob:.2f}%</div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Progress Bar
                st.progress(int(top_prob))
                
                # Description
                st.markdown(f"""
                    <div class="info-card">
                        <strong>📝 Deskripsi:</strong><br>
                        {weather_data['desc']}
                    </div>
                """, unsafe_allow_html=True)
                
                # Tips
                st.markdown("### 💡 Tips & Rekomendasi")
                for tip in weather_data['tips']:
                    st.markdown(f"✓ {tip}")
                
                if show_details:
                    st.markdown("---")
                    st.markdown("### 📊 Analisis Detail")
                    
                    # Top 5 Predictions
                    if show_all_probs:
                        n_show = 11
                        title_text = "Semua Probabilitas Kelas"
                    else:
                        n_show = 5
                        title_text = "Top 5 Prediksi Teratas"
                    
                    top_indices = probs.argsort()[-n_show:][::-1]
                    top_probs = probs[top_indices] * 100
                    top_labels = [CLASS_NAMES[i] for i in top_indices]
                    
                    # Color gradient untuk bar chart
                    colors = px.colors.sequential.Viridis[::-1][:len(top_labels)]
                    
                    fig = go.Figure(go.Bar(
                        y=top_labels,
                        x=top_probs,
                        orientation='h',
                        marker=dict(
                            color=top_probs,
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="Probability %")
                        ),
                        text=[f'{p:.1f}%' for p in top_probs],
                        textposition='outside',
                        hovertemplate='<b>%{y}</b><br>Probability: %{x:.2f}%<extra></extra>'
                    ))
                    
                    fig.update_layout(
                        title=title_text,
                        xaxis_title="Probabilitas (%)",
                        yaxis_title="",
                        yaxis_autorange="reversed",
                        height=300 if not show_all_probs else 500,
                        template="plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white",
                        margin=dict(l=0, r=80, t=40, b=40),
                        font=dict(size=12)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Stats Grid
                    st.markdown("### 📈 Statistik Prediksi")
                    stat_col1, stat_col2, stat_col3 = st.columns(3)
                    
                    with stat_col1:
                        st.markdown(f"""
                            <div class="stat-box">
                                <div class="stat-value">{top_prob:.1f}%</div>
                                <div class="stat-label">Confidence</div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with stat_col2:
                        entropy = -np.sum(probs * np.log(probs + 1e-10))
                        st.markdown(f"""
                            <div class="stat-box">
                                <div class="stat-value">{entropy:.2f}</div>
                                <div class="stat-label">Entropy</div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with stat_col3:
                        second_prob = probs[probs.argsort()[-2]] * 100
                        margin = top_prob - second_prob
                        st.markdown(f"""
                            <div class="stat-box">
                                <div class="stat-value">{margin:.1f}%</div>
                                <div class="stat-label">Margin</div>
                            </div>
                        """, unsafe_allow_html=True)


# --- HALAMAN 2: MODEL COMPARISON ---
elif page == "⚖️ Model Comparison":
    st.markdown("""
        <div class="main-header animate-in">
            <h1 style='margin: 0; font-size: 2.5rem;'>⚖️ Model Comparison Arena</h1>
            <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>Bandingkan performa 3 model AI secara langsung</p>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("📤 Upload Gambar untuk Dibandingkan", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        
        # Centered image preview
        col_img = st.columns([1, 2, 1])
        with col_img[1]:
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(image, use_container_width=True, caption="Input Image")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("## 🏆 Model Performance Comparison")
        
        if models_dict:
            # Hitung prediksi untuk semua model
            results = {}
            with st.spinner('🔄 Memproses dengan 3 model...'):
                for name, model in models_dict.items():
                    probs = predict_image(model, image)
                    top_idx = np.argmax(probs)
                    results[name] = {
                        'probs': probs,
                        'top_idx': top_idx,
                        'confidence': probs[top_idx] * 100,
                        'label': CLASS_NAMES[top_idx]
                    }
            
            # Model Cards
            col1, col2, col3 = st.columns(3, gap="large")
            cols = [col1, col2, col3]
            model_names = ['CNN Base', 'MobileNetV2', 'ResNet50V2']
            model_icons = ['🔷', '📱', '🏗️']
            
            for col, name, icon in zip(cols, model_names, model_icons):
                with col:
                    # Cek apakah model tersedia dalam dictionary
                    if name in results:
                        result = results[name]
                        weather_data = WEATHER_INFO[result['label']]
                        
                        # Status color
                        if result['confidence'] > 80:
                            status_color = "#11998e"
                        elif result['confidence'] > 50:
                            status_color = "#f093fb"
                        else:
                            status_color = "#fa709a"
                        
                        st.markdown(f"""
                            <div class="metric-card" style="border-top: 4px solid {status_color};">
                                <div style="text-align: center;">
                                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
                                    <h3 style="margin: 0.5rem 0; color: {status_color};">{name}</h3>
                                    <div style="font-size: 2.5rem; margin: 1rem 0;">{weather_data['icon']}</div>
                                    <h2 style="margin: 0.5rem 0; color: var(--text-primary);">{result['label']}</h2>
                                    <div style="font-size: 2rem; font-weight: bold; color: {status_color}; margin: 1rem 0;">
                                        {result['confidence']:.1f}%
                                    </div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Mini Bar Chart
                        top_3_indices = result['probs'].argsort()[-3:][::-1]
                        top_3_probs = result['probs'][top_3_indices] * 100
                        top_3_labels = [CLASS_NAMES[i] for i in top_3_indices]
                        
                        fig = go.Figure(go.Bar(
                            x=top_3_probs,
                            y=top_3_labels,
                            orientation='h',
                            marker_color=status_color,
                            text=[f'{p:.1f}%' for p in top_3_probs],
                            textposition='outside'
                        ))
                        
                        fig.update_layout(
                            height=200,
                            showlegend=False,
                            template="plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white",
                            margin=dict(l=0, r=60, t=10, b=10),
                            xaxis_showticklabels=False,
                            yaxis=dict(autorange="reversed")
                        )
                        
                        st.plotly_chart(fig, use_container_width=True, key=f"mini_chart_{name}")
            
            # Comparison Analysis
            st.markdown("---")
            st.markdown("## 📊 Analisis Perbandingan Lengkap")
            
            # Agreement Analysis
            valid_models = [name for name in model_names if name in results]
            predictions = [results[name]['label'] for name in valid_models]
            
            if predictions:
                if len(set(predictions)) == 1:
                    agreement_status = "✅ Semua model sepakat!"
                    agreement_color = "#11998e"
                elif len(set(predictions)) == 2:
                    agreement_status = "⚠️ Prediksi model berbeda"
                    agreement_color = "#f5576c"
                else:
                    agreement_status = "❌ Semua model berbeda"
                    agreement_color = "#fa709a"
                
                st.markdown(f"""
                    <div class="info-card" style="border-left-color: {agreement_color};">
                        <strong>{agreement_status}</strong><br>
                        Konsistensi prediksi antar model penting untuk validasi hasil.
                    </div>
                """, unsafe_allow_html=True)
            
            # Detailed Comparison Table
            comparison_data = []
            for name in valid_models:
                result = results[name]
                comparison_data.append({
                    'Model': name,
                    'Prediksi': result['label'],
                    'Confidence (%)': f"{result['confidence']:.2f}",
                    'Top-2 (%)': f"{result['probs'][result['probs'].argsort()[-2]] * 100:.2f}",
                    'Margin (%)': f"{result['confidence'] - result['probs'][result['probs'].argsort()[-2]] * 100:.2f}"
                })
            
            st.dataframe(comparison_data, use_container_width=True, hide_index=True)
            
            # Radar Chart Comparison
            st.markdown("### 🎯 Radar Comparison - Top 5 Classes")
            
            # Ambil top 5 kelas berdasarkan rata-rata probabilitas
            avg_probs = np.mean([results[name]['probs'] for name in valid_models], axis=0)
            top_5_global = avg_probs.argsort()[-5:][::-1]
            categories = [CLASS_NAMES[i] for i in top_5_global]
            
            fig_radar = go.Figure()
            
            colors_radar = ['#667eea', '#f093fb', '#11998e']
            for i, name in enumerate(valid_models):
                probs_selected = results[name]['probs'][top_5_global] * 100
                fig_radar.add_trace(go.Scatterpolar(
                    r=probs_selected,
                    theta=categories,
                    fill='toself',
                    name=name,
                    line_color=colors_radar[i % len(colors_radar)]
                ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                ),
                showlegend=True,
                height=400,
                template="plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white"
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
            
            # Model Performance Stats
            st.markdown("### 📈 Performance Metrics")
            
            perf_col1, perf_col2, perf_col3 = st.columns(3)
            
            for col, name in zip([perf_col1, perf_col2, perf_col3], valid_models):
                with col:
                    result = results[name]
                    entropy = -np.sum(result['probs'] * np.log(result['probs'] + 1e-10))
                    
                    st.markdown(f"#### {name}")
                    st.metric("Max Confidence", f"{result['confidence']:.1f}%")
                    st.metric("Entropy", f"{entropy:.3f}")
                    st.metric("Certainty Score", f"{(100 - entropy * 10):.1f}%")


# --- HALAMAN 3: BATCH ANALYSIS ---
elif page == "📊 Batch Analysis":
    st.markdown("""
        <div class="main-header animate-in">
            <h1 style='margin: 0; font-size: 2.5rem;'>📊 Batch Weather Analysis</h1>
            <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>Upload multiple images untuk analisis batch</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.info("💡 **Tips**: Upload beberapa gambar sekaligus untuk analisis komparatif!")
    
    selected_model_batch = st.selectbox(
        "Pilih Model untuk Batch Analysis:",
        list(models_dict.keys()),
        index=1 if 'MobileNetV2' in models_dict else 0
    )
    
    uploaded_files = st.file_uploader(
        "Upload Multiple Images",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True
    )
    
    if uploaded_files and len(uploaded_files) > 0:
        st.markdown(f"### 📁 {len(uploaded_files)} gambar diupload")
        
        if st.button("🚀 Mulai Batch Processing", use_container_width=True):
            model = models_dict[selected_model_batch]
            
            results_batch = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {idx + 1}/{len(uploaded_files)}: {uploaded_file.name}")
                
                image = Image.open(uploaded_file).convert("RGB")
                probs = predict_image(model, image)
                top_idx = np.argmax(probs)
                
                results_batch.append({
                    'filename': uploaded_file.name,
                    'prediction': CLASS_NAMES[top_idx],
                    'confidence': float(probs[top_idx] * 100),  # Convert to native Python float
                    'icon': WEATHER_INFO[CLASS_NAMES[top_idx]]['icon']
                })
                
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            status_text.text("✅ Processing complete!")
            
            st.markdown("---")
            st.markdown("## 📋 Batch Results")
            
            # Results Grid
            cols_result = st.columns(min(3, len(uploaded_files)))
            
            for idx, result in enumerate(results_batch):
                with cols_result[idx % 3]:
                    weather_data = WEATHER_INFO[result['prediction']]
                    st.markdown(f"""
                        <div class="metric-card">
                            <div style="text-align: center;">
                                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">{result['icon']}</div>
                                <h4 style="margin: 0.5rem 0; color: {weather_data['color']};">{result['prediction']}</h4>
                                <p style="font-size: 0.8rem; color: var(--text-secondary); margin: 0.3rem 0;">{result['filename']}</p>
                                <div style="font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0;">
                                    {result['confidence']:.1f}%
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
            
            # Summary Statistics
            st.markdown("---")
            st.markdown("## 📊 Batch Summary")
            
            # Count by weather type
            from collections import Counter
            weather_counts = Counter([r['prediction'] for r in results_batch])
            
            col_sum1, col_sum2 = st.columns([2, 1])
            
            with col_sum1:
                # Distribution Chart
                labels = list(weather_counts.keys())
                values = list(weather_counts.values())
                colors_pie = [WEATHER_INFO[label]['color'] for label in labels]
                
                fig_pie = go.Figure(data=[go.Pie(
                    labels=labels,
                    values=values,
                    marker=dict(colors=colors_pie),
                    hole=0.4,
                    textinfo='label+percent',
                    hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
                )])
                
                fig_pie.update_layout(
                    title="Distribusi Tipe Cuaca",
                    height=400,
                    template="plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white"
                )
                
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col_sum2:
                st.markdown("### 📈 Quick Stats")
                
                avg_confidence = np.mean([r['confidence'] for r in results_batch])
                max_confidence = max([r['confidence'] for r in results_batch])
                min_confidence = min([r['confidence'] for r in results_batch])
                
                st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
                st.metric("Max Confidence", f"{max_confidence:.1f}%")
                st.metric("Min Confidence", f"{min_confidence:.1f}%")
                st.metric("Total Images", len(uploaded_files))
                
                # Most common prediction
                most_common = weather_counts.most_common(1)[0]
                st.markdown(f"""
                    <div class="info-card">
                        <strong>🏆 Most Common:</strong><br>
                        {WEATHER_INFO[most_common[0]]['icon']} {most_common[0]} ({most_common[1]}x)
                    </div>
                """, unsafe_allow_html=True)
            
            # Detailed Table
            st.markdown("### 📑 Detailed Results Table")
            st.dataframe(results_batch, use_container_width=True, hide_index=True)
            
            # Download Results
            import json
            results_json = json.dumps(results_batch, indent=2)
            st.download_button(
                label="⬇️ Download Results (JSON)",
                data=results_json,
                file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )


# --- HALAMAN 4: ABOUT ---
elif page == "ℹ️ About":
    st.markdown("""
        <div class="main-header animate-in">
            <h1 style='margin: 0; font-size: 2.5rem;'>ℹ️ Tentang Proyek</h1>
            <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>Weather AI Classifier - Deep Learning Project</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Project Overview
    st.markdown("""
        <div class="metric-card animate-in">
            <h2>🎯 Project Overview</h2>
            <p>Aplikasi ini merupakan implementasi sistem klasifikasi cuaca berbasis Deep Learning 
            yang menggunakan tiga arsitektur neural network berbeda untuk membandingkan performa 
            dan akurasi prediksi kondisi cuaca dari gambar.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Model Architecture
    col_about1, col_about2, col_about3 = st.columns(3)
    
    with col_about1:
        st.markdown("""
            <div class="metric-card">
                <div style="text-align: center;">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">🔷</div>
                    <h3>CNN Base</h3>
                    <p style="color: var(--text-secondary); font-size: 0.9rem;">
                    Custom architecture dibangun dari scratch menggunakan layer Conv2D, MaxPooling, 
                    dan Dense. Cocok untuk memahami fundamental deep learning.
                    </p>
                    <hr style="margin: 1rem 0;">
                    <strong>Key Features:</strong>
                    <ul style="text-align: left; font-size: 0.85rem;">
                        <li>3-4 Conv layers</li>
                        <li>MaxPooling layers</li>
                        <li>Dropout regularization</li>
                        <li>Dense layers untuk klasifikasi</li>
                    </ul>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col_about2:
        st.markdown("""
            <div class="metric-card">
                <div style="text-align: center;">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">📱</div>
                    <h3>MobileNetV2</h3>
                    <p style="color: var(--text-secondary); font-size: 0.9rem;">
                    Arsitektur lightweight yang dioptimalkan untuk perangkat mobile. 
                    Menggunakan transfer learning dari ImageNet weights.
                    </p>
                    <hr style="margin: 1rem 0;">
                    <strong>Key Features:</strong>
                    <ul style="text-align: left; font-size: 0.85rem;">
                        <li>Inverted residual blocks</li>
                        <li>Depthwise separable convolutions</li>
                        <li>Efficient parameter usage</li>
                        <li>Fast inference time</li>
                    </ul>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col_about3:
        st.markdown("""
            <div class="metric-card">
                <div style="text-align: center;">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">🏗️</div>
                    <h3>ResNet50V2</h3>
                    <p style="color: var(--text-secondary); font-size: 0.9rem;">
                    Deep residual network dengan skip connections. Sangat powerful 
                    untuk ekstraksi fitur kompleks dari gambar.
                    </p>
                    <hr style="margin: 1rem 0;">
                    <strong>Key Features:</strong>
                    <ul style="text-align: left; font-size: 0.85rem;">
                        <li>50 layer depth</li>
                        <li>Residual connections</li>
                        <li>Batch normalization</li>
                        <li>High accuracy potential</li>
                    </ul>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # Dataset Information
    st.markdown("---")
    st.markdown("## 📚 Dataset Information")
    
    col_data1, col_data2 = st.columns([2, 1])
    
    with col_data1:
        st.markdown("""
            <div class="metric-card">
                <h3>🗂️ Weather Dataset</h3>
                <p>Dataset diambil dari <strong>Kaggle Weather Dataset</strong> yang berisi ribuan gambar cuaca 
                dalam 11 kategori berbeda. Setiap kategori merepresentasikan kondisi cuaca spesifik 
                yang umum ditemui di berbagai belahan dunia.</p>
                
                <h4 style="margin-top: 1.5rem;">📊 Dataset Statistics:</h4>
                <ul style="text-align: left; padding-left: 1.5rem;">
                    <li style="margin-bottom: 0.5rem;"><strong>Total Classes:</strong> 11 weather types</li>
                    <li style="margin-bottom: 0.5rem;"><strong>Image Resolution:</strong> 224x224 pixels (resized)</li>
                    <li style="margin-bottom: 0.5rem;"><strong>Color Space:</strong> RGB (3 channels)</li>
                    <li style="margin-bottom: 0.5rem;"><strong>Data Split:</strong> Train / Validation / Test</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col_data2:
        st.markdown("""
            <div class="metric-card">
                <h3>🌦️ Weather Classes</h3>
        """, unsafe_allow_html=True)
        
        for class_name in CLASS_NAMES:
            weather_data = WEATHER_INFO[class_name]
            st.markdown(f"{weather_data['icon']} **{class_name}**")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Technical Details
    st.markdown("---")
    st.markdown("## 🔧 Technical Implementation")
    
    tech_col1, tech_col2 = st.columns(2)
    
    with tech_col1:
        st.markdown("""
            <div class="metric-card">
                <h3>⚙️ Technologies Used</h3>
                <ul>
                    <li><strong>Framework:</strong> TensorFlow / Keras</li>
                    <li><strong>UI:</strong> Streamlit</li>
                    <li><strong>Visualization:</strong> Plotly</li>
                    <li><strong>Image Processing:</strong> PIL, NumPy</li>
                    <li><strong>Model Format:</strong> .keras (TF 2.x)</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with tech_col2:
        st.markdown("""
            <div class="metric-card">
                <h3>🎨 Features</h3>
                <ul>
                    <li>Single image prediction</li>
                    <li>Multi-model comparison</li>
                    <li>Batch processing</li>
                    <li>Interactive visualizations</li>
                    <li>Dark mode support</li>
                    <li>Responsive design</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    # Developer Info
    st.markdown("---")
    st.markdown("""
        <div class="metric-card" style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);">
            <h2 style="text-align: center; margin-bottom: 1rem;">👨‍💻 Developer Information</h2>
            <div style="text-align: center;">
                <p><strong>Nama:</strong> Lutfi Indra Nur Pradiya</p>
                <p><strong>NIM:</strong> 202210370311482</p>
                <p><strong>Project:</strong> Machine Learning - Weather Classification</p>
                <p style="margin-top: 1rem; color: var(--text-secondary);">
                    <em>"Leveraging Deep Learning for Accurate Weather Pattern Recognition"</em>
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; padding: 2rem 0; color: var(--text-secondary);">
            <p>Made with ❤️ using Streamlit & TensorFlow</p>
            <p style="font-size: 0.85rem;">© 2024 Weather AI Classifier. All rights reserved.</p>
        </div>
    """, unsafe_allow_html=True)