import numpy as np
import streamlit as st
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai

# --- Configuración de Gemini AI ---
try:
    genai.configure(api_key=st.secrets["gemini"]["api_key"])
except:
    st.error("❌ Error: Configura tu API Key en .streamlit/secrets.toml")
    st.stop()

# Configuración del modelo Gemini
model = genai.GenerativeModel('gemini-2.0-flash')

# --- Funciones de cálculo Black-Scholes ---
def black_scholes_call(S, K, T, r, sigma):
    """Calcula el precio de una opción CALL usando Black-Scholes"""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def black_scholes_put(S, K, T, r, sigma):
    """Calcula el precio de una opción PUT usando Black-Scholes"""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

def calculate_greeks(S, K, T, r, sigma):
    """Calcula las griegas para opciones CALL y PUT"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    delta_call = norm.cdf(d1)
    delta_put = delta_call - 1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) * 0.01  # por 1% cambio en vol
    theta_call = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    theta_put = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    rho_call = K * T * np.exp(-r * T) * norm.cdf(d2) * 0.01  # por 1% cambio en tasa
    rho_put = -K * T * np.exp(-r * T) * norm.cdf(-d2) * 0.01
    
    return {
        'Delta Call': delta_call,
        'Delta Put': delta_put,
        'Gamma': gamma,
        'Vega': vega,
        'Theta Call': theta_call,
        'Theta Put': theta_put,
        'Rho Call': rho_call,
        'Rho Put': rho_put
    }

def enhanced_heatmap(S, K, T, r, min_spot, max_spot, min_vol, max_vol):
    """Genera un heatmap mejorado con contornos"""
    spot_prices = np.linspace(min_spot, max_spot, 100)
    volatilities = np.linspace(min_vol, max_vol, 100)
    spot_grid, vol_grid = np.meshgrid(spot_prices, volatilities)
    
    # Vectorizar el cálculo de precios
    vectorized_bs = np.vectorize(black_scholes_call)
    price_grid = vectorized_bs(spot_grid, K, T, r, vol_grid)
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Heatmap con contornos
    heatmap = ax.contourf(spot_grid, vol_grid, price_grid, levels=20, cmap='viridis')
    contour = ax.contour(spot_grid, vol_grid, price_grid, levels=10, colors='black', linewidths=0.5)
    ax.clabel(contour, inline=True, fontsize=8)
    
    # Marcar el punto actual
    if min_spot <= S <= max_spot and min_vol <= sigma <= max_vol:
        ax.scatter(S, sigma, color='red', s=100, label='Current Position')
        ax.legend()
    
    ax.set_xlabel('Spot Price')
    ax.set_ylabel('Volatility')
    ax.set_title('Option Price Heatmap with Contours')
    plt.colorbar(heatmap, ax=ax, label='Option Price')
    
    return fig

def get_ai_analysis(S, K, call_price, put_price):
    """Obtiene análisis de rentabilidad de Gemini AI"""
    prompt = f"""
    Como analista financiero senior, proporciona un dictamen conciso (1-2 oraciones) sobre qué opción es más rentable actualmente basado en:
    - Precio actual: {S}
    - Precio ejercicio: {K} 
    - Valor CALL: {call_price:.2f}
    - Valor PUT: {put_price:.2f}
    
    Responde comenzando con "ANÁLISIS:" y destacando solo el factor más decisivo.
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error en análisis: {str(e)}"

# --- Interfaz de Usuario ---
st.set_page_config(layout="wide", page_title="Black-Scholes AI", page_icon="📊")
st.title("📈 Black-Scholes con Análisis de Gemini AI")
st.markdown("By Leo Aguilar")

# Sidebar
with st.sidebar:
    st.header("Parámetros")
    S = st.number_input("Precio Actual (S)", value=100.0, min_value=0.01)
    K = st.number_input("Precio Ejercicio (K)", value=100.0, min_value=0.01)
    T = st.number_input("Tiempo a Vencimiento (años)", value=1.0, min_value=0.01, max_value=50.0)
    sigma = st.number_input("Volatilidad (σ)", value=0.2, min_value=0.01, max_value=5.0)
    r = st.number_input("Tasa Libre de Riesgo", value=0.05, min_value=0.0, max_value=1.0)
    
    st.header("Configuración")
    heatmap_active = st.checkbox("Mostrar Heatmap", True)
    if heatmap_active:
        min_spot = st.number_input("Mínimo Spot", value=90.0, min_value=0.01)
        max_spot = st.number_input("Máximo Spot", value=110.0, min_value=0.01)
        min_vol = st.number_input("Mínima Volatilidad", value=0.1, min_value=0.01)
        max_vol = st.number_input("Máxima Volatilidad", value=0.5, min_value=0.01)

# Pestañas
tab1, tab2 = st.tabs(["Calculadora", "Griegas"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        option_type = st.radio("Tipo de Opción", ["Call / Put"])
    
    with col2:
        if st.button("Calcular"):
            if T > 0 and sigma > 0:
                call_price = black_scholes_call(S, K, T, r, sigma)
                put_price = black_scholes_put(S, K, T, r, sigma)
                
                st.session_state.call_price = call_price
                st.session_state.put_price = put_price
                
                st.metric("Precio CALL", f"${call_price:.2f}")
                st.metric("Precio PUT", f"${put_price:.2f}")
                
                if option_type == "Call / Put":
                    st.divider()
                    with st.spinner("Analizando rentabilidad..."):
                        analysis = get_ai_analysis(S, K, call_price, put_price)
                        st.success(analysis)
            else:
                st.warning("Tiempo y Volatilidad deben ser > 0")

    if heatmap_active and 'call_price' in st.session_state:
        st.divider()
        st.header("Heatmap de Precios")
        heatmap_fig = enhanced_heatmap(S, K, T, r, min_spot, max_spot, min_vol, max_vol)
        st.pyplot(heatmap_fig)

with tab2:
    if T > 0 and sigma > 0:
        greeks = calculate_greeks(S, K, T, r, sigma)
        
        cols = st.columns(4)
        with cols[0]:
            st.metric("Delta CALL", f"{greeks['Delta Call']:.4f}")
            st.metric("Gamma", f"{greeks['Gamma']:.6f}")
        with cols[1]:
            st.metric("Delta PUT", f"{greeks['Delta Put']:.4f}")
            st.metric("Vega", f"{greeks['Vega']:.4f}")
        with cols[2]:
            st.metric("Theta CALL", f"{greeks['Theta Call']:.4f}/día")
            st.metric("Rho CALL", f"{greeks['Rho Call']:.4f}")
        with cols[3]:
            st.metric("Theta PUT", f"{greeks['Theta Put']:.4f}/día")
            st.metric("Rho PUT", f"{greeks['Rho Put']:.4f}")
    else:
        st.warning("Calcule primero los precios")

# Nota legal
st.caption("⚠️ Este análisis es informativo. Consulte con un profesional antes de invertir.")
