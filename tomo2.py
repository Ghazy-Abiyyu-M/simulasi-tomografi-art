import streamlit as st
import numpy as np
import time
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import rotate
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, iradon_sart, resize
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from typing import Tuple, List, Dict, Optional


def init_app():
    st.set_page_config(page_title="Tomografi ART - Simulasi", layout="wide", page_icon="🔬", initial_sidebar_state="expanded")
    
    st.markdown("""
        <style>
        .main { background-color: #0E1117; color: #FAFAFA; font-family: 'Inter', sans-serif; }
        
        .metric-container { display: flex; justify-content: space-between; gap: 15px; margin-bottom: 20px;}
        .stMetric { 
            flex: 1;
            background: linear-gradient(145deg, #1A1C23, #252830); 
            padding: 20px; 
            border-radius: 12px; 
            box-shadow: 4px 4px 15px rgba(0,0,0,0.4);
            border-top: 4px solid #00F0FF;
            transition: transform 0.2s ease-in-out;
        }
        .stMetric:hover { transform: translateY(-3px); }
        
        .theory-box { background: #1A202C; border: 1px solid #4A5568; padding: 20px; border-radius: 8px; margin-bottom: 20px; font-size: 0.95rem; line-height: 1.6;}
        
        .app-header { display: flex; flex-direction: column; padding-bottom: 15px; border-bottom: 1px solid #333; margin-bottom: 25px;}
        .app-header h1 { margin:0; padding:0; color: #00F0FF; font-weight: 800; font-size: 2.2rem;}
        .app-header h3 { margin:5px 0 0 0; color: #A0AEC0; font-weight: 400; font-size: 1.2rem;}
        
        /* Footer custom CSS */
        .app-footer { margin-top: 50px; padding-top: 25px; border-top: 1px solid #333; text-align: center; color: #A0AEC0; padding-bottom: 20px;}
        .app-footer p { margin: 0; font-size: 0.95rem; }
        .app-footer a { color: #00F0FF; text-decoration: none; font-weight: bold; padding: 5px 10px; border-radius: 5px; background: rgba(0, 240, 255, 0.1); transition: 0.3s;}
        .app-footer a:hover { background: rgba(0, 240, 255, 0.2); }
        </style>
    """, unsafe_allow_html=True)

    if 'anim_micro' not in st.session_state:
        st.session_state.anim_micro = False
    if 'anim_macro' not in st.session_state:
        st.session_state.anim_macro = False
    if 'macro_frame' not in st.session_state:
        st.session_state.macro_frame = 1


@st.cache_data
def build_micro_system(N: int = 16, num_angles: int = 4, noise_level: float = 0.0):
    x_true_2d = np.zeros((N, N))
    x_true_2d[N//4:N//2+2, N//4:N//2+2] = 1.0 
    x_true_2d[N//2+1:N-3, N//2+1:N-3] = 0.5   
    x_true = x_true_2d.flatten()

    angles = np.linspace(0, 180, num_angles, endpoint=False)
    x_coords = np.arange(-N//2, N//2) + 0.5
    X, Y = np.meshgrid(x_coords, x_coords)
    X_flat, Y_flat = X.flatten(), Y.flatten()

    A_rows = []
    ray_info = []
    
    for theta in angles:
        th = np.radians(theta)
        r_max = int(np.ceil(np.sqrt(2) * N / 2))
        for r in np.arange(-r_max, r_max + 1, 0.8):
            d = np.abs(X_flat * np.cos(th) + Y_flat * np.sin(th) - r)
            weights = np.zeros_like(d)
            mask = d < 0.8
            weights[mask] = 1.0 - d[mask] 
            
            if np.sum(weights) > 0.1:
                A_rows.append(weights)
                ray_info.append({"angle": theta, "offset": r})
                
    A = np.array(A_rows)
    b = A @ x_true
    if noise_level > 0:
        b += np.random.normal(0, noise_level * np.max(b), b.shape)
        
    return A, b, x_true_2d, ray_info

@st.cache_data
def simulate_micro_art_steps(A, b, N, relaxation=1.0):
    M, N_sq = A.shape
    x = np.zeros(N_sq)
    history = [{'x': x.copy().reshape(N, N), 'ray_idx': -1, 'error': 0, 'delta': np.zeros((N,N))}]
    
    for i in range(M):
        Ai = A[i]
        bi = b[i]
        proj = np.dot(Ai, x)
        error = bi - proj
        norm_sq = np.dot(Ai, Ai)
        
        if norm_sq > 1e-6:
            delta_x = relaxation * (error / norm_sq) * Ai
            x = x + delta_x
            history.append({
                'x': x.copy().reshape(N, N),
                'ray_idx': i,
                'error': error,
                'delta': delta_x.reshape(N, N),
                'Ai': Ai.reshape(N, N),
                'bi': bi,
                'proj': proj
            })
    return history

def plot_micro_grid(img, title, colorscale='gray', zmin=0, zmax=1.0, show_grid=True):
    fig = px.imshow(img, color_continuous_scale=colorscale, zmin=zmin, zmax=zmax, title=title)
    if show_grid:
        fig.update_traces(xgap=2, ygap=2) 
    fig.update_layout(coloraxis_showscale=False, margin=dict(l=10, r=10, t=40, b=10), height=320, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
    return fig

@st.cache_data
def load_phantom(custom_image_array: Optional[np.ndarray] = None) -> np.ndarray:
    target_size = (256, 256)
    if custom_image_array is not None:
        img = resize(custom_image_array, target_size, anti_aliasing=True)
    else:
        img = resize(shepp_logan_phantom(), target_size, anti_aliasing=True)
    max_val = np.max(img)
    return img / max_val if max_val > 0 else img

@st.cache_data
def generate_realistic_sinogram(phantom: np.ndarray, angles: int, noise_type: str, noise_level: float, I0: float) -> Tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0., 180., angles, endpoint=False)
    ideal_proj = radon(phantom, theta=theta)
    
    if noise_type == "Poisson (Tingkat Gangguan Tinggi)":
        transmitted_intensity = I0 * np.exp(-ideal_proj)
        noisy_intensity = np.random.poisson(transmitted_intensity)
        noisy_intensity[noisy_intensity == 0] = 1 
        sinogram = -np.log(noisy_intensity / I0)
    else: 
        sinogram = ideal_proj + np.random.normal(0, noise_level, ideal_proj.shape)
        
    return np.clip(sinogram, 0, None), theta

@st.cache_data(show_spinner=False)
def compute_reconstructions(sinogram: np.ndarray, theta: np.ndarray, max_iters: int) -> Tuple[np.ndarray, List[np.ndarray]]:
    img_fbp = np.clip(iradon(sinogram, theta=theta, filter_name='ramp'), 0, 1)
    sart_history = [np.clip(iradon_sart(sinogram, theta=theta), 0, 1)]
    for _ in range(1, max_iters):
        sart_history.append(np.clip(iradon_sart(sinogram, theta=theta, image=sart_history[-1]), 0, 1))
    return img_fbp, sart_history

def compute_advanced_metrics(true_img: np.ndarray, recon_img: np.ndarray, roi: Tuple[int,int,int,int]) -> Dict[str, float]:
    y1, y2, x1, x2 = roi
    t_img, r_img = true_img[y1:y2, x1:x2], recon_img[y1:y2, x1:x2]
    if t_img.size == 0: return {}
    mse = mean_squared_error(t_img, r_img)
    return {
        "RMSE": np.sqrt(mse),
        "PSNR": peak_signal_noise_ratio(t_img, r_img, data_range=1.0),
        "SSIM": structural_similarity(t_img, r_img, data_range=1.0, win_size=min(7, t_img.shape[0])) if min(7, t_img.shape[0]) >= 3 else 0
    }

def main():
    init_app()
    
    # --- HEADER ---
    st.markdown("""
        <div class='app-header'>
            <h1>Simulasi Rekonstruksi Tomografi: FBP vs ART</h1>
            <h3>Tugas Pengganti Minggu ke-7 | Ghazy Abiyyu Maulana (1104220120)</h3>
        </div>
    """, unsafe_allow_html=True)
    
    with st.expander("📚 Konsep Dasar Tomografi (Wajib Baca untuk Pemula)", expanded=False):
        st.markdown("""
        <div class='theory-box'>
        <b>1. Masalah Dasar Tomografi:</b> <br>
        Tomografi adalah proses merekonstruksi struktur penampang melintang suatu objek (citra 2D) berdasarkan kumpulan data proyeksi (sinyal 1D) yang ditangkap oleh sensor dari berbagai sudut. Bayangkan mencoba menebak isi ruangan hanya dengan melihat bayangan yang jatuh ke dinding dari berbagai arah senter.
        <br><br>
        <b>2. Filtered Back Projection (FBP) - "Cara Lama":</b> <br>
        FBP adalah metode analitik konvensional. Cara kerjanya adalah dengan memproyeksikan balik (<i>back-projecting</i>) sinyal dari sensor secara langsung ke ruang matriks citra. FBP sangat cepat secara komputasi dan sangat akurat <b>hanya jika</b> jumlah sudut proyeksinya lengkap (360 derajat). Jika sudutnya terbatas (<i>few-view</i>), data menjadi tidak lengkap dan FBP akan menghasilkan artefak garis (<i>streak artifacts</i>) yang parah.
        <br><br>
        <b>3. Algebraic Reconstruction Technique (ART) - "Metode Baru":</b> <br>
        ART adalah metode iteratif. Alih-alih menggambar balik secara langsung, ART memodelkan proses ini sebagai sistem persamaan linier matematika: <b>Ax = b</b>.
        <ul>
            <li><b>Matriks A:</b> Merepresentasikan geometri lintasan sinar.</li>
            <li><b>Vektor x:</b> Merepresentasikan citra objek yang ingin kita cari/rekonstruksi.</li>
            <li><b>Vektor b:</b> Merepresentasikan data redaman yang diukur oleh sensor.</li>
        </ul>
        Algoritma ART menebak nilai citra (<i>x</i>), membandingkannya dengan data sensor asli (<i>b</i>), lalu menyebarkan selisihnya (nilai <i>error</i>) untuk mengoreksi tebakan citra. Proses ini diulang (iterasi) hingga error menjadi sekecil mungkin, sehingga mampu membersihkan artefak garis pada kondisi data terbatas.
        <br><br>
        <i>Referensi Jurnal: "Iterative Image Reconstruction Algorithm with Parameter Estimation by Neural Network for Computed Tomography" (Takeshi Kojima and Tetsuya Yoshinaga, MDPI Algorithms, 2023).</i>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"f_{1}(z;y,\lambda) := \frac{\sum_{i=1}^{I}A_{ij}\left(\frac{y_{i}}{(A_{i}z)^{k}}\right)^{\gamma}}{\sum_{i=1}^{I}A_{ij}\left(\frac{A_{i}z}{(A_{i}z)^{k}}\right)^{\gamma}}")

    st.sidebar.markdown("### 🎛️ Navigasi Modul")
    app_mode = st.sidebar.radio("Pilih Mode Simulasi:", [
        "1. Konsep Matriks Linier (Micro-ART)", 
        "2. Simulasi Fisis Akuisisi & Rekonstruksi"
    ])
    st.sidebar.markdown("---")

    if app_mode == "1. Konsep Matriks Linier (Micro-ART)":
        st.sidebar.subheader("⚙️ Parameter Resolusi Dasar")
        grid_N = st.sidebar.slider("Resolusi Matriks Grid (N x N)", 8, 32, 16, step=4, help="Menentukan ukuran gambar. Makin besar, makin banyak persamaan matriks yang harus diselesaikan.")
        n_angles = st.sidebar.slider("Jumlah Sudut Proyeksi", 2, 18, 4, help="Berapa banyak arah sinar yang menembus objek.")

        st.info("💡 **Penjelasan Modul 1:** Modul ini menyederhanakan cara kerja ART. Alih-alih gambar organ tubuh yang rumit, kita menggunakan kotak-kotak piksel sederhana. Anda bisa melihat secara *real-time* bagaimana komputer menggunakan matriks lintasan sinar ($A$) dan data sensor ($b$) untuk menebak dan memperbaiki nilai setiap piksel ($x$) satu per satu.")

        A, b, x_true, ray_info = build_micro_system(grid_N, n_angles, 0.0)
        M_rays = A.shape[0]
        history = simulate_micro_art_steps(A, b, grid_N)

        st.subheader(f"Visualisasi Proses Pembaruan Iteratif (ART)")
        
        col_anim1, col_anim2 = st.columns([1, 4])
        with col_anim1:
            if st.button("▶️ Eksekusi Animasi Iterasi"):
                st.session_state.anim_micro = True
            if st.button("⏹️ Hentikan Animasi"):
                st.session_state.anim_micro = False

        step_slider = st.empty()
        step = step_slider.slider("Indeks Sinar ke-", 1, M_rays, 1, key="slider_manual_micro")

        viz_container = st.empty()

        def render_micro_art_step(current_step):
            state = history[current_step]
            with viz_container.container():
                col_L, col_R = st.columns([1.2, 1])
                
                with col_L:
                    st.markdown(r"#### 1. Dinamika Pembaruan Matriks Citra")
                    st.caption("Lihat bagaimana nilai piksel diubah. Komputer memperbaiki 'Tebakan Sebelum' menjadi 'Tebakan Sesudah' berdasarkan selisih error.")
                    c1, c2 = st.columns(2)
                    c1.plotly_chart(plot_micro_grid(history[current_step-1]['x'], r"Kondisi Matriks Sebelum Pembaruan", zmax=1), use_container_width=True, key=f"c1_{current_step}")
                    c2.plotly_chart(plot_micro_grid(state['x'], r"Kondisi Matriks Sesudah Pembaruan", zmax=1), use_container_width=True, key=f"c2_{current_step}")
                    
                    st.markdown("#### 2. Komparasi dengan Target Asli")
                    st.caption("Tujuan akhir komputer adalah membuat tebakan sesudah sama persis dengan 'Citra Target Asli' di bawah ini.")
                    pseudo_fbp = np.linalg.pinv(A) @ b
                    c3, c4 = st.columns(2)
                    c3.plotly_chart(plot_micro_grid(x_true, "Citra Target Asli (Ground Truth)", zmax=1), use_container_width=True, key=f"c3_{current_step}")
                    c4.plotly_chart(plot_micro_grid(pseudo_fbp.reshape(grid_N, grid_N), "Hasil Pseudo-Invers (Cara Analitik)", zmax=1), use_container_width=True, key=f"c4_{current_step}")

                with col_R:
                    st.markdown(r"#### 3. Geometri Pembobot Sinar ($A_{ij}$)")
                    st.caption("Garis merah transparan menunjukkan area/piksel mana saja yang dilewati sinar pada putaran iterasi ini.")
                    img_bg = history[current_step-1]['x'].copy()
                    ray_mask = state['Ai']
                    
                    fig_ray = go.Figure()
                    fig_ray.add_trace(go.Heatmap(z=img_bg, colorscale='gray', zmin=0, zmax=1, showscale=False))
                    overlay = np.where(ray_mask > 0.05, ray_mask, np.nan)
                    fig_ray.add_trace(go.Heatmap(z=overlay, colorscale='Reds', zmin=0, zmax=1, showscale=False, opacity=0.7))
                    fig_ray.update_layout(title=f"Sudut Proyeksi Sinar Aktif: {ray_info[current_step-1]['angle']}°", margin=dict(l=10, r=10, t=40, b=10), height=320, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    fig_ray.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
                    st.plotly_chart(fig_ray, use_container_width=True, key=f"ray_{current_step}")

                    st.markdown("#### 4. Kalkulasi Error Matematis")
                    st.caption("Jika Error Proporsional bukan 0, komputer akan membagi error tersebut ke piksel-piksel yang dilewati garis merah di atas.")
                    st.success(
                        f"**Data Sensor Asli ($b_i$):** `{state['bi']:.3f}` \n\n"
                        f"**Estimasi Komputer ($A_i x$):** `{state['proj']:.3f}` \n\n"
                        f"**Selisih Error ($e_i$):** `{state['error']:.3f}`"
                    )

        if st.session_state.anim_micro:
            for s in range(1, M_rays + 1):
                if not st.session_state.anim_micro: break
                render_micro_art_step(s)
                time.sleep(0.15)
            st.session_state.anim_micro = False
        else:
            render_micro_art_step(step)

    else:
        with st.sidebar:
            st.subheader("⚙️ Parameter Skenario Alat CT")
            mode = st.selectbox("Pilihan Skenario Proyeksi:", [
                "1. Full Projection (180 Sudut, Ideal)", 
                "2. Few-View Projection (30 Sudut, Sudut Terbatas)"
            ], help="Ubah ke 'Few-View' untuk melihat perbedaan FBP dan ART.")
            
            if mode == "1. Full Projection (180 Sudut, Ideal)": 
                angles, noise_type, iters = 180, "Gaussian", 10; noise_lvl = 0.0
            else: 
                angles, noise_type, iters = 30, "Poisson (Tingkat Gangguan Tinggi)", 20; noise_lvl = 0.5
                
            uploaded_file = st.file_uploader("📥 Input Citra Phantom Tambahan (Opsional)", type=['png', 'jpg'])

        st.info("💡 **Mengapa Gambar FBP dan ART Terlihat Sama di Skenario 1?** Pada kondisi ideal (Full Projection/180 Sudut), alat CT mendeteksi objek dengan sempurna dari segala arah. Di sini, metode FBP sudah lebih dari cukup untuk menghasilkan gambar sempurna, sehingga ART tidak menunjukkan perbedaan visual. **Ubah pilihan di panel kiri ke Skenario 2 (Few-View)** untuk melihat garis-garis artefak FBP muncul, dan perhatikan bagaimana ART mampu membersihkannya!")

        custom_img = np.array(Image.open(uploaded_file).convert('L')) if uploaded_file else None
        phantom = load_phantom(custom_img)
        I0 = 50000 if noise_type == "Poisson (Tingkat Gangguan Tinggi)" else 1
        
        with st.spinner("🚀 Mengeksekusi Transformasi Radon & Modul Rekonstruksi..."):
            sinogram, theta = generate_realistic_sinogram(phantom, angles, noise_type, noise_lvl, I0)
            img_fbp, sart_history = compute_reconstructions(sinogram, theta, iters)
            current_sart = sart_history[-1]

        tab1, tab2, tab3 = st.tabs(["📊 Perbandingan FBP vs ART (Live)", "📡 Model Akuisisi (Sinogram)", "📈 Metrik Kinerja Kuantitatif"])

        with tab1:
            st.markdown(f"### Visualisasi Rekonstruksi - Skenario: {mode.split('(')[0]}")
            st.write("Panel ini membandingkan hasil cara lama (FBP) dengan cara baru (ART). Gunakan slider iterasi untuk melihat bagaimana ART membersihkan *noise* secara bertahap.")
            
            col_btn1, col_btn2, col_btn3 = st.columns([1.5, 1.5, 5])
            with col_btn1:
                if st.button("▶️ Putar Animasi ART", key="btn_start_macro", use_container_width=True):
                    st.session_state.anim_macro = True
                    st.session_state.macro_frame = 1
                    st.rerun()
            with col_btn2:
                if st.button("⏹️ Hentikan Animasi", key="btn_stop_macro", use_container_width=True):
                    st.session_state.anim_macro = False
                    st.rerun()

            if st.session_state.get('anim_macro', False):
                macro_step = st.session_state.get('macro_frame', 1)
                st.progress(macro_step / iters, text=f"⏳ Menjalankan iterasi konvergensi ke-{macro_step} dari {iters} iterasi maksimum...")
                st.slider("Kontrol Manual Iterasi:", 1, iters, macro_step, key="slider_macro_disabled", disabled=True)
            else:
                macro_step = st.slider("Geser slider untuk melihat proses pembersihan iteratif ART:", 1, iters, iters, key="slider_macro")

            disp_sart = sart_history[macro_step-1]
            current_mse = np.sqrt(mean_squared_error(phantom, disp_sart))
            current_ssim = structural_similarity(phantom, disp_sart, data_range=1.0, win_size=7)

            m1, m2, m3 = st.columns(3)
            m1.metric("Tingkat Sebaran Error (RMSE)", f"{current_mse:.4f}", delta=f"{current_mse - np.sqrt(mean_squared_error(phantom, sart_history[max(0, macro_step-2)])):.4f}" if macro_step > 1 else None, delta_color="inverse")
            m2.metric("Integritas Bentuk (SSIM)", f"{current_ssim:.4f}", delta=f"{current_ssim - structural_similarity(phantom, sart_history[max(0, macro_step-2)], data_range=1.0, win_size=7):.4f}" if macro_step > 1 else None)
            m3.metric("Tahap Iterasi ART Saat Ini", f"Ke-{macro_step} dari total {iters}")

            fig_ws = make_subplots(rows=2, cols=2, 
                                   subplot_titles=("1. Citra Target Asli (Ground Truth)", 
                                                   "2. Hasil FBP (Cara Analitik Lama)", 
                                                   f"3. Hasil ART (Metode Iteratif ke-{macro_step})", 
                                                   "4. Peta Deteksi Error (Area Merah/Kuning = Salah)"),
                                   vertical_spacing=0.1)
            
            fig_ws.add_trace(go.Heatmap(z=phantom, colorscale='gray', coloraxis="coloraxis1"), row=1, col=1)
            fig_ws.add_trace(go.Heatmap(z=img_fbp, colorscale='gray', coloraxis="coloraxis1"), row=1, col=2)
            fig_ws.add_trace(go.Heatmap(z=disp_sart, colorscale='gray', coloraxis="coloraxis1"), row=2, col=1)
            
            error_map = np.abs(phantom - disp_sart)
            cmax_error = max(0.05, np.max(error_map) * 0.8) 
            
            fig_ws.add_trace(go.Heatmap(z=error_map, colorscale='inferno', coloraxis="coloraxis2"), row=2, col=2)
            
            fig_ws.update_layout(height=750, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False,
                                 coloraxis1=dict(colorscale='gray', cmin=0, cmax=1, showscale=False),
                                 coloraxis2=dict(colorscale='inferno', cmin=0, cmax=cmax_error, showscale=True, colorbar=dict(title="Magnitude Cacat", x=1.05)))
            fig_ws.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False, autorange='reversed')
            
            st.plotly_chart(fig_ws, use_container_width=True, key=f"macro_fig_render_{macro_step}")

            if st.session_state.get('anim_macro', False):
                if st.session_state.macro_frame < iters:
                    time.sleep(0.3) 
                    st.session_state.macro_frame += 1
                    st.rerun() 
                else:
                    st.session_state.anim_macro = False
                    st.rerun() 

        with tab2:
            st.markdown("### 📡 Transformasi Radon (Cara Kerja Alat Fisik)")
            st.info("💡 **Penjelasan:** Di alam nyata, komputer tidak langsung mendapatkan gambar kepala/organ. Alat CT Scanner memancarkan sinar dan menangkap profil berupa **gelombang redaman (grafik tengah)**. Kumpulan grafik dari puluhan/ratusan sudut ini kemudian ditumpuk menjadi satu gambar abstrak yang disebut **Sinogram (gambar kanan)**. Rekonstruksi adalah proses mengubah Sinogram kembali menjadi gambar organ.")
            
            selected_idx = st.slider("Geser Slider untuk Memutar Alat Sensor CT", 0, angles-1, 0, key="gantry_slider")
            current_angle = theta[selected_idx]
            
            rotated_phantom = rotate(phantom, -current_angle, reshape=False, mode='nearest')
            proj_1d = sinogram[:, selected_idx]

            col_p1, col_p2, col_p3 = st.columns([1, 1, 1.2])
            
            with col_p1:
                st.markdown(f"**A. Posisi Objek di Sensor ({current_angle:.1f}°)**")
                fig_rot = px.imshow(rotated_phantom, color_continuous_scale='gray')
                fig_rot.add_hline(y=128, line_dash="solid", line_color="red", opacity=0.5) 
                fig_rot.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=10, b=0), height=300, paper_bgcolor='rgba(0,0,0,0)')
                fig_rot.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
                st.plotly_chart(fig_rot, use_container_width=True)

            with col_p2:
                st.markdown(f"**B. Sinyal yang Ditangkap Sensor (1D)**")
                fig_1d = go.Figure(data=go.Scatter(y=proj_1d, mode='lines', line=dict(color='#00F0FF', width=3), fill='tozeroy', fillcolor='rgba(0, 240, 255, 0.1)'))
                fig_1d.update_layout(
                    xaxis_title="Deretan Kamera Detektor", yaxis_title="Intensitas Bayangan",
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300, margin=dict(l=0, r=0, t=10, b=0)
                )
                fig_1d.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#333')
                fig_1d.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#333', range=[0, np.max(sinogram)*1.1])
                st.plotly_chart(fig_1d, use_container_width=True)

            with col_p3:
                st.markdown("**C. Matriks Akumulasi (Sinogram)**")
                fig_sino = px.imshow(sinogram, color_continuous_scale='gray', aspect='auto')
                fig_sino.add_hline(y=selected_idx, line_dash="solid", line_color="#00F0FF", line_width=2)
                fig_sino.update_layout(xaxis_title="Kamera Detektor", yaxis_title="Deretan Sudut", margin=dict(l=0, r=0, t=10, b=0), height=300)
                st.plotly_chart(fig_sino, use_container_width=True)

        with tab3:
            st.markdown("### 🎯 Evaluasi Akurasi (Region of Interest)")
            st.info("💡 **Penjelasan:** Kualitas gambar medis tidak bisa hanya dilihat dengan mata kasar, harus dibuktikan dengan angka. **SSIM** mengukur kemiripan bentuk dengan gambar asli (Makin dekat ke nilai 1.0, artinya sangat mirip). **RMSE** mengukur jumlah titik yang cacat/error (Makin dekat ke 0.0, artinya makin sedikit errornya). Evaluasi ini fokus di area *bounding box* (kotak) yang Anda atur.")
            
            with st.expander("⚙️ Atur Ukuran Kotak Area Evaluasi (Bounding Box)", expanded=True):
                col_roi1, col_roi2 = st.columns(2)
                with col_roi1:
                    roi_x = st.slider("Batas Kiri - Kanan (Sumbu X)", 0, 256, (50, 200))
                with col_roi2:
                    roi_y = st.slider("Batas Atas - Bawah (Sumbu Y)", 0, 256, (50, 200))
                
            col_m1, col_m2 = st.columns([1, 1.5])
            with col_m1:
                dynamic_roi = (roi_y[0], roi_y[1], roi_x[0], roi_x[1])
                m_sart = compute_advanced_metrics(phantom, current_sart, dynamic_roi)
                m_fbp = compute_advanced_metrics(phantom, img_fbp, dynamic_roi)
                
                ssim_diff = m_sart.get('SSIM',0) - m_fbp.get('SSIM',0)
                rmse_diff = m_sart.get('RMSE',0) - m_fbp.get('RMSE',0)

                st.markdown("#### Hasil Penilaian (Hanya di area kotak)")
                st.metric(label="Integritas Struktur Gambar (SSIM)", 
                          value=f"{m_sart.get('SSIM',0):.4f}", 
                          delta=f"Selisih {ssim_diff:.4f} vs FBP (+ berarti ART lebih bagus)",
                          help="Nilai optimal = 1.0 (Identik secara absolut).")
                
                st.metric(label="Total Error Tersebar (RMSE)", 
                          value=f"{m_sart.get('RMSE',0):.4f}", 
                          delta=f"Selisih {rmse_diff:.4f} vs FBP (- berarti ART lebih bagus)",
                          delta_color="inverse",
                          help="Nilai optimal = 0.0. Delta negatif menandakan metode iteratif berhasil mengurangi nilai error.")
            
            with col_m2:
                st.markdown("#### Peta 3D Gunung Error (Ketinggian = Cacat)")
                st.caption("Jika area 3D ini datar, berarti tebakan komputer sempurna. Semakin tinggi tonjolan (gunung), berarti titik piksel tersebut masih salah letak atau salah warna.")
                
                error_map_final = np.abs(phantom - current_sart)
                error_roi = error_map_final[roi_y[0]:roi_y[1], roi_x[0]:roi_x[1]]
                
                if error_roi.size > 0:
                    fig_3d = go.Figure(data=[go.Surface(z=error_roi, colorscale='inferno', showscale=False)])
                    fig_3d.update_layout(
                        height=450, margin=dict(l=0, r=0, t=0, b=0),
                        scene=dict(
                            xaxis_title='Sumbu X', 
                            yaxis_title='Sumbu Y', 
                            zaxis_title='Besaran Cacat Error',
                            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)) 
                        )
                    )
                    st.plotly_chart(fig_3d, use_container_width=True)
                else:
                    st.warning("⚠️ Area kotak evaluasi terlalu kecil atau pengaturan limit terbalik.")

    st.markdown("""
        <div class='app-footer'>
            <p><b>Tugas Pengganti Kuliah Minggu ke-7: Teknik Tomografi</b></p>
            <p style='margin-bottom: 15px; font-size: 0.9rem;'>Dibuat oleh <b>Ghazy Abiyyu Maulana</b> (1104220120) | Teknik Fisika, Telkom University</p>
            <a href='https://github.com/Ghazy-Abiyyu-M/simulasi-tomografi-art' target='_blank'>
                📦 Kunjungi Repository GitHub
            </a>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
