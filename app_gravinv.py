import streamlit as st
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
import harmonica as hm
import tempfile
import os
import plotly.graph_objects as go

# --- KONFIGURASI HALAMAN STREAMLIT ---
st.set_page_config(page_title="Modul Koreksi Gravity V2", layout="wide")
st.title("Grav-inv3D: FAA to CBA & Residual Converter V2")

# --- FUNGSI VISUALISASI ALA SURFER ---
def plot_smooth_grid(x, y, z_matrix, title, colorscale='Jet'):
    """Fungsi untuk membuat peta kontur mulus ala Surfer menggunakan Plotly"""
    fig = go.Figure(data=go.Contour(
        z=z_matrix,
        x=x, 
        y=y, 
        colorscale=colorscale,
        contours=dict(showlines=False), 
        colorbar=dict(title="mGal")
    ))
    
    fig.update_layout(
        title=title,
        autosize=False,
        width=700,
        height=600,
        margin=dict(l=50, r=50, b=50, t=50)
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig

# --- FUNGSI POLINOMIAL REGIONAL-RESIDUAL ---
def fit_polynomial_surface(X, Y, Z, order=2):
    """Menghitung Regional dan Residual menggunakan Least Squares Polynomial dengan Normalisasi UTM"""
    x_flat = X.ravel()
    y_flat = Y.ravel()
    z_flat = Z.ravel()
    
    # Filter nilai NaN (jaga-jaga pada batas grid)
    valid = ~np.isnan(z_flat)
    x_v = x_flat[valid]
    y_v = y_flat[valid]
    z_v = z_flat[valid]
    
    # Normalisasi koordinat UTM (Mencegah matriks singular/error komputasi)
    x_mean, x_std = np.mean(x_v), np.std(x_v)
    y_mean, y_std = np.mean(y_v), np.std(y_v)
    
    xn = (x_v - x_mean) / x_std
    yn = (y_v - y_mean) / y_std
    
    # Membangun Design Matrix
    if order == 1:
        A = np.c_[np.ones(xn.shape[0]), xn, yn]
    elif order == 2:
        A = np.c_[np.ones(xn.shape[0]), xn, yn, xn**2, yn**2, xn*yn]
    elif order == 3:
        A = np.c_[np.ones(xn.shape[0]), xn, yn, xn**2, yn**2, xn*yn, xn**3, yn**3, xn**2*yn, xn*yn**2]
        
    # Least Squares Regression untuk mencari koefisien tren regional
    C, _, _, _ = np.linalg.lstsq(A, z_v, rcond=None)
    
    # Terapkan koefisien ke seluruh permukaan grid
    Xn_full = (X - x_mean) / x_std
    Yn_full = (Y - y_mean) / y_std
    
    if order == 1:
        regional = C[0] + C[1]*Xn_full + C[2]*Yn_full
    elif order == 2:
        regional = C[0] + C[1]*Xn_full + C[2]*Yn_full + C[3]*Xn_full**2 + C[4]*Yn_full**2 + C[5]*Xn_full*Yn_full
    elif order == 3:
        regional = C[0] + C[1]*Xn_full + C[2]*Yn_full + C[3]*Xn_full**2 + C[4]*Yn_full**2 + C[5]*Xn_full*Yn_full + C[6]*Xn_full**3 + C[7]*Yn_full**3 + C[8]*Xn_full**2*Yn_full + C[9]*Xn_full*Yn_full**2
        
    residual = Z - regional
    return regional, residual

# --- FUNGSI PEMROSESAN UTAMA ---
def process_faa_to_cba_prism(faa_file_path, elev_file_path, do_separation=False, poly_order=2):
    # 1. Membaca data grid
    faa_ds = rioxarray.open_rasterio(faa_file_path, parse_coordinates=True)
    elev_ds = rioxarray.open_rasterio(elev_file_path, parse_coordinates=True)
    
    if faa_ds.rio.crs is None: faa_ds.rio.write_crs("EPSG:32748", inplace=True) 
    if elev_ds.rio.crs is None: elev_ds.rio.write_crs("EPSG:32748", inplace=True)

    # 2. Resampling & Ekstraksi Matriks
    elev_matched = elev_ds.rio.reproject_match(faa_ds)
    faa_ds = faa_ds.sortby(['x', 'y'])
    elev_matched = elev_matched.sortby(['x', 'y'])
    
    faa_val = faa_ds.values[0]
    z_val = elev_matched.values[0]
    x_coords = faa_ds.x.values
    y_coords = faa_ds.y.values
    X, Y = np.meshgrid(x_coords, y_coords)

    # 3. Model Prisma 3D
    rho_darat = 2670.0  
    rho_laut = 1030.0   
    density_grid = np.where(z_val >= 0, rho_darat, (rho_darat - rho_laut))
    
    prisms = hm.prism_layer(
        coordinates=(x_coords, y_coords), surface=z_val, reference=0, properties={"density": density_grid}
    )

    # 4. Hitung Efek Topografi Total & CBA
    Z_obs = np.full_like(X, 1.0) 
    topo_effect = prisms.prism_layer.gravity(coordinates=(X, Y, Z_obs), field="g_z")
    cba_val = faa_val - topo_effect

    # 5. Hitung Regional - Residual (Jika diaktifkan)
    regional_val = np.full_like(cba_val, np.nan)
    residual_val = np.full_like(cba_val, np.nan)
    
    if do_separation:
        regional_val, residual_val = fit_polynomial_surface(X, Y, cba_val, order=poly_order)

    # 6. Susun DataFrame untuk Export CSV
    df_dict = {
        "X_UTM": X.ravel(),
        "Y_UTM": Y.ravel(),
        "Elevasi_m": z_val.ravel(),
        "FAA_mGal": faa_val.ravel(),
        "CBA_mGal": cba_val.ravel()
    }
    
    # Tambahkan kolom baru jika pemisahan diaktifkan
    if do_separation:
        df_dict["Regional_mGal"] = regional_val.ravel()
        df_dict["Residual_mGal"] = residual_val.ravel()
        
    df = pd.DataFrame(df_dict).dropna()
    
    faa_ds.close()
    elev_ds.close()
    elev_matched.close()
    
    return df, x_coords, y_coords, faa_val, cba_val, regional_val, residual_val

# --- ANTARMUKA UI STREAMLIT ---
st.sidebar.header("📂 Input Data (.grd UTM)")
faa_file = st.sidebar.file_uploader("Upload Grid FAA", type=['grd'])
elev_file = st.sidebar.file_uploader("Upload Grid Elevasi", type=['grd'])

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Parameter Filtering")
do_sep = st.sidebar.checkbox("Lakukan Pemisahan Regional-Residual", value=True)

poly_order = 2
if do_sep:
    # Memberikan opsi Orde 1, 2, atau 3 (Default: Orde 2)
    poly_order = st.sidebar.selectbox("Pilih Orde Polinomial", options=[1, 2, 3], index=1)

st.sidebar.markdown("---")
if st.sidebar.button("🚀 Mulai Proses Koreksi"):
    if faa_file and elev_file:
        with st.spinner("Sedang memproses... Prisma 3D & Interpolasi Polinomial mungkin memakan waktu sesaat."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".grd") as tmp_faa:
                    tmp_faa.write(faa_file.getvalue())
                    faa_path = tmp_faa.name
                    
                with tempfile.NamedTemporaryFile(delete=False, suffix=".grd") as tmp_elev:
                    tmp_elev.write(elev_file.getvalue())
                    elev_path = tmp_elev.name

                # Eksekusi Pemrosesan
                hasil_df, x_arr, y_arr, faa_matrix, cba_matrix, reg_matrix, res_matrix = process_faa_to_cba_prism(
                    faa_path, elev_path, do_separation=do_sep, poly_order=poly_order
                )
                
                st.success("✅ Koreksi dan Pemisahan Anomali berhasil diselesaikan!")
                
                # --- VISUALISASI FAA & CBA ---
                st.subheader("📊 Visualisasi Anomali Utama")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Free Air Anomaly (Input)**")
                    st.plotly_chart(plot_smooth_grid(x_arr, y_arr, faa_matrix, "FAA (mGal)", colorscale='Jet'), use_container_width=True)
                with col2:
                    st.markdown("**Complete Bouguer Anomaly (Hasil Koreksi)**")
                    st.plotly_chart(plot_smooth_grid(x_arr, y_arr, cba_matrix, "CBA (mGal)", colorscale='Jet'), use_container_width=True)
                
                # --- VISUALISASI REGIONAL & RESIDUAL ---
                if do_sep:
                    st.markdown("---")
                    st.subheader(f"🔍 Hasil Pemisahan Anomali (Polinomial Orde {poly_order})")
                    col3, col4 = st.columns(2)
                    with col3:
                        st.markdown("**Regional Anomaly (Trend Geologi Dalam)**")
                        st.plotly_chart(plot_smooth_grid(x_arr, y_arr, reg_matrix, "Regional (mGal)", colorscale='Jet'), use_container_width=True)
                    with col4:
                        st.markdown("**Residual Anomaly (Target Struktur Dangkal)**")
                        st.plotly_chart(plot_smooth_grid(x_arr, y_arr, res_matrix, "Residual (mGal)", colorscale='Jet'), use_container_width=True)
                
                # --- TOMBOL DOWNLOAD & PREVIEW DATA ---
                st.markdown("---")
                st.subheader("💾 Export Data Hasil Pengolahan Lengkap")
                st.dataframe(hasil_df.head(10))
                
                csv = hasil_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Data Format CSV",
                    data=csv,
                    file_name='Data_Gravitasi_Lengkap_V2.csv',
                    mime='text/csv',
                )
                
            except Exception as e:
                st.error(f"Terjadi kesalahan komputasi: {e}")
    else:
        st.warning("⚠️ Mohon upload kedua file grid terlebih dahulu di sidebar sebelah kiri.")
else:
    st.info("Silakan unggah file grid FAA dan Elevasi, pilih parameter, lalu klik tombol 'Mulai Proses Koreksi'.")
