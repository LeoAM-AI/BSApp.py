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
def black_scholes_call(S, K, days_to_expiry, r, sigma):
    """Calcula el precio de una opción CALL usando Black-Scholes"""
    T = days_to_expiry / 365.0
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def black_scholes_put(S, K, days_to_expiry, r, sigma):
    """Calcula el precio de una opción PUT usando Black-Scholes"""
    T = days_to_expiry / 365.0
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

def calculate_greeks(S, K, days_to_expiry, r, sigma):
    """Calcula las griegas para opciones CALL y PUT"""
    T = days_to_expiry / 365.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    delta_call = norm.cdf(d1)
    delta_put = delta_call - 1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) * 0.01
    theta_call = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    theta_put = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    rho_call = K * T * np.exp(-r * T) * norm.cdf(d2) * 0.01
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

# --- Funciones para gráficas ---
def plot_option_prices(S_range, call_prices, put_prices, current_S):
    """Grafica los precios de CALL y PUT en función del precio del subyacente"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(S_range, call_prices, label='CALL', color='green')
    ax.plot(S_range, put_prices, label='PUT', color='red')
    ax.axvline(x=current_S, color='blue', linestyle='--', label='Precio Actual')
    ax.set_xlabel('Precio del Subyacente')
    ax.set_ylabel('Precio de la Opción')
    ax.set_title('Precios de Opciones vs. Precio del Subyacente')
    ax.legend()
    ax.grid(True)
    return fig

def plot_greeks(S_range, deltas, thetas, rhos, current_S):
    """Grafica las griegas principales en función del precio del subyacente"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Delta
    ax1.plot(S_range, deltas['call'], label='Delta CALL', color='green')
    ax1.plot(S_range, deltas['put'], label='Delta PUT', color='red')
    ax1.axvline(x=current_S, color='blue', linestyle='--')
    ax1.set_title('Delta de las Opciones')
    ax1.legend()
    ax1.grid(True)
    
    # Theta
    ax2.plot(S_range, thetas['call'], label='Theta CALL', color='green')
    ax2.plot(S_range, thetas['put'], label='Theta PUT', color='red')
    ax2.axvline(x=current_S, color='blue', linestyle='--')
    ax2.set_title('Theta de las Opciones (por día)')
    ax2.legend()
    ax2.grid(True)
    
    # Rho
    ax3.plot(S_range, rhos['call'], label='Rho CALL', color='green')
    ax3.plot(S_range, rhos['put'], label='Rho PUT', color='red')
    ax3.axvline(x=current_S, color='blue', linestyle='--')
    ax3.set_title('Rho de las Opciones')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    return fig

# --- Interfaz de Usuario ---
st.set_page_config(layout="wide", page_title="Black-Scholes AI", page_icon="📊")
st.title("📈 Black-Scholes con Visualizaciones Mejoradas")
st.markdown("By Leo Aguilar")

# Sidebar
with st.sidebar:
    st.header("Parámetros")
    S = st.number_input("Precio Actual (S)", value=100.0, min_value=0.01)
    K = st.number_input("Precio Ejercicio (K)", value=100.0, min_value=0.01)
    days_to_expiry = st.number_input("Días a la Expiración", value=30, min_value=1, max_value=365*5)
    sigma = st.number_input("Volatilidad (σ)", value=0.2, min_value=0.01, max_value=5.0)
    r = st.number_input("Tasa Libre de Riesgo", value=0.05, min_value=0.0, max_value=1.0)
    
    st.header("Rango de Análisis")
    S_min = st.number_input("Precio Mínimo", value=80.0, min_value=0.01)
    S_max = st.number_input("Precio Máximo", value=120.0, min_value=0.01)
    n_points = st.slider("Número de puntos", 20, 200, 50)

# Pestañas
tab1, tab2 = st.tabs(["Precios", "Griegas"])

# Generar rango de precios para análisis
S_range = np.linspace(S_min, S_max, n_points)

with tab1:
    if st.button("Calcular Precios y Mostrar Gráficas"):
        # Calcular precios para el rango
        call_prices = [black_scholes_call(s, K, days_to_expiry, r, sigma) for s in S_range]
        put_prices = [black_scholes_put(s, K, days_to_expiry, r, sigma) for s in S_range]
        
        # Precios actuales
        current_call = black_scholes_call(S, K, days_to_expiry, r, sigma)
        current_put = black_scholes_put(S, K, days_to_expiry, r, sigma)
        
        # Mostrar métricas
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Precio CALL Actual", f"${current_call:.2f}")
        with col2:
            st.metric("Precio PUT Actual", f"${current_put:.2f}")
        
        # Mostrar gráfica de precios
        st.subheader("Evolución de Precios")
        fig_prices = plot_option_prices(S_range, call_prices, put_prices, S)
        st.pyplot(fig_prices)
        
        # Análisis de IA
        with st.spinner("Generando análisis..."):
            analysis = model.generate_content(f"Analiza estos precios de opciones: CALL=${current_call:.2f}, PUT=${current_put:.2f} con S={S}, K={K}, días={days_to_expiry}, σ={sigma}, r={r}. Resumen conciso de 1-2 oraciones.")
            st.success(analysis.text)

with tab2:
    if st.button("Calcular Griegas y Mostrar Gráficas"):
        # Calcular griegas para el rango
        deltas = {'call': [], 'put': []}
        thetas = {'call': [], 'put': []}
        rhos = {'call': [], 'put': []}
        
        for s in S_range:
            greeks = calculate_greeks(s, K, days_to_expiry, r, sigma)
            deltas['call'].append(greeks['Delta Call'])
            deltas['put'].append(greeks['Delta Put'])
            thetas['call'].append(greeks['Theta Call'])
            thetas['put'].append(greeks['Theta Put'])
            rhos['call'].append(greeks['Rho Call'])
            rhos['put'].append(greeks['Rho Put'])
        
        # Griegas actuales
        current_greeks = calculate_greeks(S, K, days_to_expiry, r, sigma)
        
        # Mostrar métricas
        cols = st.columns(4)
        with cols[0]:
            st.metric("Delta CALL", f"{current_greeks['Delta Call']:.4f}")
        with cols[1]:
            st.metric("Theta CALL", f"{current_greeks['Theta Call']:.4f}/día")
        with cols[2]:
            st.metric("Rho CALL", f"{current_greeks['Rho Call']:.4f}")
        with cols[3]:
            st.metric("Delta PUT", f"{current_greeks['Delta Put']:.4f}")
        
        # Mostrar gráficas de griegas
        st.subheader("Evolución de las Griegas")
        fig_greeks = plot_greeks(S_range, deltas, thetas, rhos, S)
        st.pyplot(fig_greeks)

# Nota legal
st.caption("⚠️ Este análisis es informativo. Consulte con un profesional antes de invertir.")
