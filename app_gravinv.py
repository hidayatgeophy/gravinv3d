import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import griddata
from scipy.signal import fftconvolve
import plotly.graph_objects as go
from scipy.signal import fftconvolve
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator

# ==========================================
# MODUL 1: GRIDDING, KERNEL & WEIGHTING
# ==========================================
def setup_3d_grid(x, y, anomaly, nx, ny, nz, z_top, z_bottom):
    xi = np.linspace(x.min(), x.max(), nx)
    yi = np.linspace(y.min(), y.max(), ny)
    XI, YI = np.meshgrid(xi, yi)
    dx = xi[1] - xi[0]
    dy = yi[1] - yi[0]
    
    g_grid = griddata((x, y), anomaly, (XI, YI), method='cubic')
    g_grid = np.nan_to_num(g_grid, nan=np.nanmean(g_grid))
    
    z_bounds = np.linspace(z_top, z_bottom, nz + 1)
    return XI, YI, g_grid, dx, dy, z_bounds

def calculate_kernel_layer(XI, YI, dx, dy, z_layer_top, z_layer_bottom, rho_layer, beta_weight):
    GAMMA = 6.67430e-3 
    X_k = XI - np.mean(XI)
    Y_k = YI - np.mean(YI)
    R2 = X_k**2 + Y_k**2
    
    # Kernel Dasar
    K = rho_layer * ( (-1 / np.sqrt(R2 + z_layer_bottom**2)) - (-1 / np.sqrt(R2 + z_layer_top**2)) )
    
    # Penerapan Depth Weighting
    z_center = (z_layer_top + z_layer_bottom) / 2.0
    epsilon = dx / 2.0 
    depth_weight = 1.0 / (z_center + epsilon)**beta_weight
    
    K_weighted = K * depth_weight
    
    S = GAMMA * K_weighted.sum() * dx * dy 
    S_matrix = np.full_like(XI, S)
    return GAMMA * K_weighted, S_matrix

# ==========================================
# MODUL 2: FILTER SEPARASI LAYER
# ==========================================
def upward_continuation(field, dz, dx, dy):
    ny, nx = field.shape
    kx = np.fft.fftfreq(nx, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(ny, d=dy) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    K_wave = np.sqrt(KX**2 + KY**2)
    return np.real(np.fft.ifft2(np.fft.fft2(field) * np.exp(-dz * K_wave)))

def downward_continuation_lavrentiev(field, dz, kappa, dx, dy):
    ny, nx = field.shape
    kx = np.fft.fftfreq(nx, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(ny, d=dy) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    K_wave = np.sqrt(KX**2 + KY**2)
    down_filter_reg = 1.0 / (np.exp(-dz * K_wave) + kappa)
    return np.real(np.fft.ifft2(np.fft.fft2(field) * down_filter_reg))

def extract_layer_anomaly(g_obs, z_depth, kappa, dx, dy):
    if z_depth <= 0: return g_obs
    g_up = upward_continuation(g_obs, z_depth, dx, dy)
    g_down = downward_continuation_lavrentiev(g_up, 2 * z_depth, kappa, dx, dy)
    return upward_continuation(g_down, z_depth, dx, dy)


# ==========================================
# VISUALISASI PRE-INVERSI
# ==========================================
def plot_3d_grid_wireframe(nx, ny, nz, z_top, z_bottom, x_max, y_max):
    fig = go.Figure()
    x_lines = [0, x_max, x_max, 0, 0, 0, x_max, x_max, 0, 0, x_max, x_max, x_max, x_max, 0, 0]
    y_lines = [0, 0, y_max, y_max, 0, 0, 0, y_max, y_max, 0, 0, 0, y_max, y_max, y_max, y_max]
    z_lines = [z_top, z_top, z_top, z_top, z_top, z_bottom, z_bottom, z_bottom, z_bottom, z_bottom, z_bottom, z_top, z_top, z_bottom, z_bottom, z_top]
    
    fig.add_trace(go.Scatter3d(x=x_lines, y=y_lines, z=z_lines, mode='lines', line=dict(color='blue', width=4), name='Bounding Box'))
    z_layers = np.linspace(z_top, z_bottom, nz+1)
    for z in z_layers[1:-1]:
        fig.add_trace(go.Scatter3d(
            x=[0, x_max, x_max, 0, 0], y=[0, 0, y_max, y_max, 0], z=[z, z, z, z, z],
            mode='lines', line=dict(color='gray', width=1, dash='dot'), showlegend=False
        ))
    fig.update_layout(
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Depth (Z)', zaxis=dict(autorange="reversed")),
        margin=dict(l=0, r=0, b=0, t=0), height=400
    )
    return fig

# ==========================================
# MODUL 3: INVERSI KOREKSI LOKAL ITERATIF
# ==========================================
from scipy.ndimage import gaussian_filter

def gravity_inversion_3d(XI, YI, g_obs, dx, dy, z_bounds, nx, ny, nz, rho_1d_model, beta_weight, max_iter, smooth_sigma, max_dev):
    # 1. Sabuk Pengaman DC Shift (Menyelaraskan kurva)
    g_mean = np.nanmean(g_obs)
    g_obs_shifted = g_obs - g_mean
    
    Density_Model_3D = np.zeros((ny, nx, nz))
    g_calc_total = np.zeros((ny, nx))
    
    # Regularisasi Lavrentiev yang lebih kuat agar tidak meledak di bawah
    kappas = np.logspace(-2, 0.5, nz + 1)
    
    g_bases = [extract_layer_anomaly(g_obs_shifted, z, k, dx, dy) for z, k in zip(z_bounds, kappas)]
        
    progress_bar = st.progress(0)
    for k in range(nz):
        z_top = z_bounds[k]
        z_bottom = z_bounds[k+1]
        z_center = (z_top + z_bottom) / 2
        
        # 2. Interpolasi Gradien Densitas 1D yang Mulus
        rho_layer = np.interp(z_center, rho_1d_model['Depth (m)'].values, rho_1d_model['Density (g/cc)'].values)
        
        g_layer = g_bases[k] - g_bases[k+1]
        g_layer_initial = np.copy(g_layer)
        
        K, S = calculate_kernel_layer(XI, YI, dx, dy, z_top, z_bottom, rho_layer, beta_weight)
        Phi = np.zeros_like(g_layer)
        
        for theta in range(1, int(max_iter) + 1): 
            delta_U = fftconvolve(g_layer, K, mode='same') * dx * dy
            S_S, S_dU = np.sum(S*S)*dx*dy, np.sum(S*delta_U)*dx*dy
            dU_dU, dg_dU = np.sum(delta_U*delta_U)*dx*dy, np.sum(g_layer*delta_U)*dx*dy
            dg_S = np.sum(g_layer*S)*dx*dy
            
            Q = (dU_dU * S_S) - (S_dU**2)
            if Q == 0: break
                
            alpha = (S_S * dg_dU - S_dU * dg_S) / Q
            beta_corr = (dU_dU * dg_S - S_dU * dg_dU) / Q
            
            Phi = Phi + (alpha * g_layer) + beta_corr
            
            # 3. Filter Horizontal (Menghapus tiang vertikal)
            if smooth_sigma > 0:
                Phi = gaussian_filter(Phi, sigma=smooth_sigma)
                
            # 4. Batas Toleransi Geologi (Mencegah Density Inversion)
            Phi = np.clip(Phi, -max_dev/rho_layer, max_dev/rho_layer)
            
            g_layer = g_layer - (alpha * delta_U) - (beta_corr * S)
            if np.sqrt(np.sum(g_layer**2)*dx*dy) < 1e-3: break
                
        Density_Model_3D[:, :, k] = rho_layer + (rho_layer * Phi)
        g_calc_total += (g_layer_initial - g_layer)
        progress_bar.progress((k + 1) / nz)
        
    # 5. POST-PROCESSING: 3D Geological Smoothing
    # Memadukan antar kedalaman agar bentuknya menyambung seperti horizon (garis merah)
    if smooth_sigma > 0:
        Density_Model_3D = gaussian_filter(Density_Model_3D, sigma=(smooth_sigma, smooth_sigma, 1.0))
        
    g_calc_total = g_calc_total + g_mean
    return Density_Model_3D, g_calc_total

# ==========================================
# STREAMLIT USER INTERFACE
# ==========================================
st.set_page_config(page_title="G-Invert Pro", layout="wide")
st.title("G-Invert Pro: Interactive 3D Gravity Inversion")
st.markdown("*Dibuat oleh Hidayat untuk kemajuan geofisika Indonesia 🇮🇩*")

if 'inversion_done' not in st.session_state: st.session_state.inversion_done = False

# --- UI SIDEBAR: Parameter ---
st.sidebar.header("1. Model Geometry")
nx = st.sidebar.number_input("Nx (Blok X)", value=50, step=10)
ny = st.sidebar.number_input("Ny (Blok Y)", value=50, step=10)
nz = st.sidebar.number_input("Nz (Layer Z)", value=10, step=2)
z_top = st.sidebar.number_input("Z Top (m)", value=0.0)
z_bottom = st.sidebar.number_input("Z Bottom (m)", value=5000.0)

st.sidebar.header("2. Inversion Parameters")
beta_weight = st.sidebar.slider("Depth Weighting (Beta)", 0.0, 3.0, 1.5, 0.1)

# --- TAMBAHAN KONTROL GEOLOGI ---
smooth_sigma = st.sidebar.slider("Horizontal Smoothness", 0.0, 5.0, 2.0, 0.1, help="Memaksa anomali menyambung mulus secara lateral (menghapus tiang zebra).")
max_dev = st.sidebar.slider("Max Density Deviation (+/- g/cc)", 0.05, 0.50, 0.20, 0.05, help="Batas maksimal algoritma boleh mengubah densitas awal.")
max_iter = st.sidebar.number_input("Max Iterasi", value=15)

# --- MAIN AREA ---
st.header("Step 1: Upload Data & Initial 1D Model")
uploaded_file = st.file_uploader("Upload File (CSV/TXT: X, Y, Elev, Anomali mGal)", type=['csv', 'txt'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep=None, engine='python')
    cols = df.columns
    x_data, y_data, anomaly_data = df[cols[0]].values, df[cols[1]].values, df[cols[3]].values
    
    XI, YI, g_grid, dx, dy, z_bounds = setup_3d_grid(x_data, y_data, anomaly_data, nx, ny, nz, z_top, z_bottom)
    
    st.markdown("### Data Preview & Mesh Geometry")
    col_map, col_mesh = st.columns(2)
    with col_map:
        st.write("2D Anomaly Map (Interpolated)")
        fig2d, ax2d = plt.subplots(figsize=(6, 5))
        c2d = ax2d.contourf(XI, YI, g_grid, levels=50, cmap='jet')
        plt.colorbar(c2d, ax=ax2d, label='Anomali (mGal)')
        ax2d.set_xlabel("X (m)"); ax2d.set_ylabel("Y (m)")
        st.pyplot(fig2d)
        
    with col_mesh:
        st.write(f"3D Grid Design: {nx}x{ny}x{nz} Blocks")
        fig3d = plot_3d_grid_wireframe(nx, ny, nz, z_top, z_bottom, XI.max()-XI.min(), YI.max()-YI.min())
        st.plotly_chart(fig3d, use_container_width=True)

    st.markdown("### Step 2: Define 1D Initial Density Model $\\rho_0(z)$")
    default_depths = np.linspace(z_top, z_bottom, 5)
    default_rho = pd.DataFrame({"Depth (m)": default_depths, "Density (g/cc)": [2.2, 2.4, 2.6, 2.7, 2.8]})
    rho_1d_model = st.data_editor(default_rho, num_rows="dynamic", use_container_width=True)

    st.markdown("---")
    if st.button("🚀 Run 3D Inversion", use_container_width=True):
        with st.spinner("Executing Local Corrections & Matrix Inversion..."):
            Density_3D, g_calc = gravity_inversion_3d(
                XI, YI, g_grid, dx, dy, z_bounds, nx, ny, nz, rho_1d_model, beta_weight, max_iter, smooth_sigma, max_dev
            )
            
            # --- PERBAIKAN 1: SIMPAN SEMUA VARIABEL PENTING KE BRANKAS ---
            st.session_state['Density_3D'] = Density_3D
            st.session_state['XI'] = XI
            st.session_state['YI'] = YI
            st.session_state['z_bounds'] = z_bounds
            st.session_state['g_obs'] = g_grid  # Disimpan agar bisa diplot di QC
            st.session_state['g_calc'] = g_calc
            st.session_state.inversion_done = True # Buka kunci pintu untuk Step 3-6
            # -------------------------------------------------------------
        # ==========================================
        # 🎉 EFEK SELEBRASI & MUSIK DISCO 🎉
        # ==========================================
        # 1. Animasi balon terbang dari bawah layar
        st.balloons() 

        # 2. Efek Musik Autoplay (Hanya main detik ke-10 sampai ke-15)
        try:
            # Tambahkan start_time dan end_time di sini
            st.audio("give_it_up.mp3", format="audio/mp3", start_time=32, end_time=39, autoplay=True)
            st.success("🎵 Inversi Selesai!🕺")
        except FileNotFoundError:
            st.warning("⚠️ Inversi Selesai, tapi file 'give_it_up.mp3' belum ada")
        # ==========================================
# ==========================================
# VISUALISASI QC & SEMUA MODUL EXPORT
# ==========================================
# --- PERBAIKAN 2: BUNGKUS SEMUA MODUL KE DALAM SATU RUMAH UTAMA ---
if st.session_state.inversion_done:
    st.markdown("---")
    st.header("Step 3: Quality Control & Analysis")
    
    # Tarik semua data dari brankas memori ke dunia nyata
    XI, YI, z_bounds = st.session_state.XI, st.session_state.YI, st.session_state.z_bounds
    Density_3D = st.session_state.Density_3D
    g_obs, g_calc = st.session_state.g_obs, st.session_state.g_calc
    
    ny_shape, nx_shape = g_obs.shape
    x_min, x_max = float(XI.min()), float(XI.max())
    y_min, y_max = float(YI.min()), float(YI.max())
    
    st.markdown("**Colorbar Settings:**")
    col_v1, col_v2, _ = st.columns([1, 1, 2])
    with col_v1: vmin_plot = st.number_input("Warna Min", value=2.0, step=0.1)
    with col_v2: vmax_plot = st.number_input("Warna Max", value=3.0, step=0.1)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        # --- PERBAIKAN: TAMBAHAN OPSI ARBITRARY ---
        profil_pilihan = st.radio("Orientasi Slicing:", ("Profil XZ (B-T)", "Profil YZ (S-U)", "Arbitrary (Garis Bebas)"))
        
        if profil_pilihan == "Profil XZ (B-T)":
            idx = st.slider("Geser Indeks Y:", 0, ny_shape - 1, ny_shape // 2)
            jarak_x, o_curve, c_curve, slice_den = XI[idx, :], g_obs[idx, :], g_calc[idx, :], Density_3D[idx, :, :]
            xlabel, title = "Jarak X (m)", f"Profil XZ di Y={YI[idx, 0]:.1f}"
            
        elif profil_pilihan == "Profil YZ (S-U)":
            idx = st.slider("Geser Indeks X:", 0, nx_shape - 1, nx_shape // 2)
            jarak_x, o_curve, c_curve, slice_den = YI[:, idx], g_obs[:, idx], g_calc[:, idx], Density_3D[:, idx, :]
            xlabel, title = "Jarak Y (m)", f"Profil YZ di X={XI[0, idx]:.1f}"
            
        else:
            # --- MODUL POTONG ARBITRARY ---
            st.markdown("**Koordinat Garis Sayatan:**")
            c_a1, c_a2 = st.columns(2)
            with c_a1: xa = st.slider("X Awal (A)", min_value=x_min, max_value=x_max, value=x_min, format="%.0f")
            with c_a2: ya = st.slider("Y Awal (A)", min_value=y_min, max_value=y_max, value=y_min, format="%.0f")
            
            c_b1, c_b2 = st.columns(2)
            with c_b1: xb = st.slider("X Akhir (B)", min_value=x_min, max_value=x_max, value=x_max, format="%.0f")
            with c_b2: yb = st.slider("Y Akhir (B)", min_value=y_min, max_value=y_max, value=y_max, format="%.0f")
            
            # Buat titik-titik sepanjang garis A-B
            num_points = 100
            line_x = np.linspace(xa, xb, num_points)
            line_y = np.linspace(ya, yb, num_points)
            jarak_x = np.sqrt((line_x - xa)**2 + (line_y - ya)**2)
            
            # Mesin Interpolasi 2D untuk kurva Calculated dan Observed
            x_1d, y_1d = XI[0, :], YI[:, 0]
            interp_obs = RegularGridInterpolator((y_1d, x_1d), g_obs, bounds_error=False, fill_value=np.nan)
            interp_calc = RegularGridInterpolator((y_1d, x_1d), g_calc, bounds_error=False, fill_value=np.nan)
            points_2d = np.array([line_y, line_x]).T
            o_curve = interp_obs(points_2d)
            c_curve = interp_calc(points_2d)
            
            # Mesin Interpolasi 3D untuk mengiris Kubus Densitas
            z_centers = (z_bounds[:-1] + z_bounds[1:]) / 2
            interp_3d = RegularGridInterpolator((y_1d, x_1d, z_centers), Density_3D, bounds_error=False, fill_value=np.nan)
            
            slice_den = np.zeros((num_points, len(z_centers)))
            for i in range(len(z_centers)):
                points_3d = np.array([line_y, line_x, np.full(num_points, z_centers[i])]).T
                slice_den[:, i] = interp_3d(points_3d)
                
            xlabel, title = "Jarak dari Titik A (m)", "Sayatan Arbitrary (A-B)"

        # --- UPDATE PETA INSET ---
        st.markdown("<br>**📍 Posisi Lintasan Slicing:**", unsafe_allow_html=True)
        fig_inset, ax_inset = plt.subplots(figsize=(4, 4))
        ax_inset.contourf(XI, YI, g_obs, levels=30, cmap='jet', alpha=0.5)
        
        if profil_pilihan == "Profil XZ (B-T)":
            y_line = YI[idx, 0]
            ax_inset.axhline(y=y_line, color='red', linewidth=3, linestyle='--')
            ax_inset.text(XI.min(), y_line, ' A', color='red', weight='bold', va='bottom')
            ax_inset.text(XI.max(), y_line, 'B ', color='red', weight='bold', va='bottom', ha='right')
        elif profil_pilihan == "Profil YZ (S-U)":
            x_line = XI[0, idx]
            ax_inset.axvline(x=x_line, color='red', linewidth=3, linestyle='--')
            ax_inset.text(x_line, YI.min(), ' A', color='red', weight='bold', ha='left')
            ax_inset.text(x_line, YI.max(), 'B ', color='red', weight='bold', ha='left', va='top')
        else:
            # Peta Inset untuk Arbitrary
            ax_inset.plot([xa, xb], [ya, yb], color='red', linewidth=3, linestyle='--')
            ax_inset.plot(xa, ya, 'ro') # Titik A
            ax_inset.plot(xb, yb, 'ro') # Titik B
            ax_inset.text(xa, ya, ' A', color='red', weight='bold', va='bottom')
            ax_inset.text(xb, yb, ' B', color='red', weight='bold', va='bottom')

        ax_inset.set_xticks([]); ax_inset.set_yticks([])
        ax_inset.set_title("Top-Down View", fontsize=10)
        st.pyplot(fig_inset)

    with col2:
        fig = plt.figure(figsize=(10, 6))
        gs = GridSpec(2, 1, height_ratios=[1.5, 3], hspace=0.15)
        
        ax0 = fig.add_subplot(gs[0])
        ax0.plot(jarak_x, o_curve, 'ko', markersize=4, label='Observed')
        ax0.plot(jarak_x, c_curve, 'r-', linewidth=2, label='Calculated')
        ax0.set_title(title, fontweight='bold'); ax0.legend(); ax0.grid(True, linestyle='--')
        
        ax1 = fig.add_subplot(gs[1], sharex=ax0)
        z_centers = (z_bounds[:-1] + z_bounds[1:]) / 2
        levels = np.linspace(vmin_plot, vmax_plot, 100)
        
        slice_den_visual = gaussian_filter(slice_den, sigma=1.0) 
        c = ax1.contourf(jarak_x, z_centers, slice_den_visual.T, levels=levels, cmap='jet', extend='both')
        ax1.invert_yaxis()
        ax1.set_xlabel(xlabel); ax1.set_ylabel("Depth (m)")
        plt.colorbar(c, ax=ax1, orientation='horizontal', pad=0.15, label='Density (g/cc)')
        st.pyplot(fig)

    # ==========================================
    # MODUL 4: EXPORT TO TXT (FULL 3D CUBE)
    # ==========================================
    st.markdown("---")
    st.header("Step 4: Export Full 3D Cube Model")
    st.write("Simpan seluruh volume matriks 3D (Point Cloud) ke format TXT untuk software eksternal.")

    ny_s, nx_s, nz_s = Density_3D.shape
    x_3d = np.repeat(XI[:, :, np.newaxis], nz_s, axis=2)
    y_3d = np.repeat(YI[:, :, np.newaxis], nz_s, axis=2)
    z_1d = -1.0 * z_centers
    z_3d = np.tile(z_1d, (ny_s, nx_s, 1))
    
    df_export = pd.DataFrame({
        'X': x_3d.flatten(), 'Y': y_3d.flatten(), 'Depth': z_3d.flatten(), 'Density': Density_3D.flatten()
    })
    txt_data = df_export.to_csv(index=False, sep='\t')
    st.success(f"✅ Matriks 3D Cube berhasil dirangkum! Total titik: **{len(df_export):,} titik**.")
    
    st.download_button(
        label="💾 Download Full 3D Cube (.txt)",
        data=txt_data,
        file_name="Full_3D_Density_Cube.txt",
        mime="text/plain",
        width="stretch" # Perbaikan sintaks Streamlit
    )

    # ==========================================
    # MODUL 5: PREDICTED SEISMIC GENERATOR
    # ==========================================
    st.markdown("---")
    st.header("Step 5: Predicted Seismic Section (Synthetic Seismogram)")
    st.write("Mengonversi penampang densitas menjadi seismik sintetik ala GeoModeller (Gardner & Ricker Wavelet).")
    
    col_s1, col_s2 = st.columns([1, 3])
    with col_s1:
        st.markdown("**Parameter Seismik**")
        freq = st.slider("Frekuensi Wavelet (Hz)", 10, 80, 30, step=5)
        st.info("💡 Penampang otomatis sinkron dengan irisan yang dipilih di Step 3.")
        
    with col_s2:
        dz = abs(z_bounds[1] - z_bounds[0]) 
        nx_slice, nz_slice = slice_den.shape
        
        slice_den_safe = np.clip(slice_den, 1.0, 4.0)
        Vp = (slice_den_safe / 0.31)**4
        AI = slice_den_safe * Vp
        
        RC = np.zeros_like(AI)
        for i in range(nz_slice - 1):
            RC[:, i] = (AI[:, i+1] - AI[:, i]) / (AI[:, i+1] + AI[:, i] + 1e-10)
            
        mean_vp = np.mean(Vp)
        wavelength = mean_vp / freq
        panjang_gelombang = min(wavelength, 10000.0) 
        f_spasial = 1.0 / panjang_gelombang
        
        z_wave = np.arange(-panjang_gelombang, panjang_gelombang, dz)
        wavelet = (1.0 - 2.0 * (np.pi**2) * (f_spasial**2) * (z_wave**2)) * np.exp(- (np.pi**2) * (f_spasial**2) * (z_wave**2))
        
        seismic = np.zeros_like(RC)
        for i in range(nx_slice):
            seismic[i, :] = np.convolve(RC[i, :], wavelet, mode='same')
            
        fig_seis = plt.figure(figsize=(10, 8))
        gs_seis = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.2)
        
        ax_seis = fig_seis.add_subplot(gs_seis[0])
        vm = np.max(np.abs(seismic)) * 0.3 
        c_seis = ax_seis.pcolormesh(jarak_x, z_centers, seismic.T, cmap='RdBu_r', vmin=-vm, vmax=vm, shading='nearest')
        ax_seis.invert_yaxis()
        ax_seis.set_title("Predicted Seismic Section", fontweight='bold')
        ax_seis.set_ylabel("Depth (m)")
        ax_seis.grid(True, linestyle=':', alpha=0.5)
        
        ax_den = fig_seis.add_subplot(gs_seis[1], sharex=ax_seis)
        levels_den = np.linspace(np.min(slice_den), np.max(slice_den), 100)
        c_den = ax_den.contourf(jarak_x, z_centers, slice_den.T, levels=levels_den, cmap='jet', extend='both')
        ax_den.invert_yaxis()
        ax_den.set_xlabel("Jarak (m)")
        ax_den.set_ylabel("Depth (m)")
        st.pyplot(fig_seis)

    # ==========================================
    # MODUL 6: EXPORT TO SEGY
    # ==========================================
    st.markdown("---")
    st.header("Step 6: Export to SEG-Y")
    
    export_col1, export_col2 = st.columns([1, 2])
    with export_col1:
        try:
            from obspy.core import Trace, Stream
            from obspy.io.segy.core import _write_segy
            import io
            
            with st.spinner("Membungkus data ke format SEG-Y..."):
                seis_stream = Stream()
                for i in range(nx_slice):
                    data_trace = np.float32(seismic[i, :])
                    tr = Trace(data=data_trace)
                    tr.stats.delta = 0.001
                    
                    tr.stats.segy = {}
                    tr.stats.segy.trace_header = {}
                    tr.stats.segy.trace_header.trace_sequence_number_within_line = i + 1
                    tr.stats.segy.trace_header.source_coordinate_x = int(jarak_x[i])
                    seis_stream.append(tr)
                    
                segy_buffer = io.BytesIO()
                _write_segy(seis_stream, segy_buffer, data_encoding=5) 
                segy_bytes = segy_buffer.getvalue()
                
                if profil_pilihan == "Profil XZ (B-T)":
                    nama_file_segy = f"Predicted_Seismic_XZ_Y{int(YI[idx, 0])}.sgy"
                elif profil_pilihan == "Profil YZ (S-U)":
                    nama_file_segy = f"Predicted_Seismic_YZ_X{int(XI[0, idx])}.sgy"
                else:
                    nama_file_segy = "Predicted_Seismic_Arbitrary.sgy"
            # ---------------------------------
            
            st.download_button(
                label="💾 Download File SEG-Y (*.sgy)",
                data=segy_bytes,
                file_name=nama_file_segy, # Panggil variabel nama file yang sudah aman
                mime="application/octet-stream",
                width="stretch" 
            )
            
        except ImportError:
            st.error("⚠️ Library 'obspy' belum terinstal. Buka terminal dan ketik: `pip install obspy`")
            
    with export_col2:
        st.info("💡 **Catatan Teknis:** File SEG-Y yang diekspor berada dalam **Domain Kedalaman (Depth Domain)**.")
        
        # Tambahan peringatan WAJIB untuk nilai Z-Step
        st.warning(f"⚠️ **PENTING SAAT IMPORT KE PETREL / OPENDTECT:**\n\nKarena keterbatasan bawaan format SEG-Y standar, jarak antar sampel di dalam file dikunci ke nilai dummy. \n\nSaat melakukan *Load/Import* file ini ke *software* eksternal, Bapak **WAJIB** melakukan *override* Z-Step/Sample Interval menjadi **{dz:.2f} meter**.")
