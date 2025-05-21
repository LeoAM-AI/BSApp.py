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
    """Calcula todas las griegas para opciones CALL y PUT"""
    T = days_to_expiry / 365.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Cálculo de griegas
    delta_call = norm.cdf(d1)
    delta_put = delta_call - 1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))  # Gamma es igual para CALL y PUT
    vega = S * norm.pdf(d1) * np.sqrt(T) * 0.01  # Por 1% cambio en volatilidad
    theta_call = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    theta_put = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    rho_call = K * T * np.exp(-r * T) * norm.cdf(d2) * 0.01  # Por 1% cambio en tasa
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
    """Grafica los precios de CALL y PUT"""
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

def plot_delta_gamma(S_range, deltas, gammas, current_S):
    """Grafica Delta y Gamma"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Delta
    ax1.plot(S_range, deltas['call'], label='Delta CALL', color='green')
    ax1.plot(S_range, deltas['put'], label='Delta PUT', color='red')
    ax1.axvline(x=current_S, color='blue', linestyle='--')
    ax1.set_title('Delta de las Opciones')
    ax1.legend()
    ax1.grid(True)
    
    # Gamma (igual para CALL y PUT)
    ax2.plot(S_range, gammas, label='Gamma', color='purple')
    ax2.axvline(x=current_S, color='blue', linestyle='--')
    ax2.set_title('Gamma (igual para CALL y PUT)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    return fig

def plot_vega_theta_rho(S_range, vegas, thetas, rhos, current_S):
    """Grafica Vega, Theta y Rho"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Vega (igual para CALL y PUT)
    ax1.plot(S_range, vegas, label='Vega', color='orange')
    ax1.axvline(x=current_S, color='blue', linestyle='--')
    ax1.set_title('Vega (sensibilidad a la volatilidad)')
    ax1.legend()
    ax1.grid(True)
    
    # Theta
    ax2.plot(S_range, thetas['call'], label='Theta CALL', color='green')
    ax2.plot(S_range, thetas['put'], label='Theta PUT', color='red')
    ax2.axvline(x=current_S, color='blue', linestyle='--')
    ax2.set_title('Theta (decaimiento temporal por día)')
    ax2.legend()
    ax2.grid(True)
    
    # Rho
    ax3.plot(S_range, rhos['call'], label='Rho CALL', color='green')
    ax3.plot(S_range, rhos['put'], label='Rho PUT', color='red')
    ax3.axvline(x=current_S, color='blue', linestyle='--')
    ax3.set_title('Rho (sensibilidad a tasas de interés)')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    return fig

# --- Interfaz de Usuario ---
st.set_page_config(layout="wide", page_title="Black-Scholes con Griegas", page_icon="📊")
st.title("📈 Análisis Completo de Opciones con Todas las Griegas")
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

# Generar rango de precios para análisis
S_range = np.linspace(S_min, S_max, n_points)

# Pestañas
tab1, tab2, tab3 = st.tabs(["Precios", "Delta & Gamma", "Vega, Theta & Rho"])

with tab1:
    if st.button("Calcular Precios"):
        call_prices = [black_scholes_call(s, K, days_to_expiry, r, sigma) for s in S_range]
        put_prices = [black_scholes_put(s, K, days_to_expiry, r, sigma) for s in S_range]
        
        current_call = black_scholes_call(S, K, days_to_expiry, r, sigma)
        current_put = black_scholes_put(S, K, days_to_expiry, r, sigma)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Precio CALL Actual", f"${current_call:.2f}")
        with col2:
            st.metric("Precio PUT Actual", f"${current_put:.2f}")
        
        st.pyplot(plot_option_prices(S_range, call_prices, put_prices, S))

with tab2:
    if st.button("Calcular Delta y Gamma"):
        deltas = {'call': [], 'put': []}
        gammas = []
        
        for s in S_range:
            greeks = calculate_greeks(s, K, days_to_expiry, r, sigma)
            deltas['call'].append(greeks['Delta Call'])
            deltas['put'].append(greeks['Delta Put'])
            gammas.append(greeks['Gamma'])
        
        current_greeks = calculate_greeks(S, K, days_to_expiry, r, sigma)
        
        cols = st.columns(2)
        with cols[0]:
            st.metric("Delta CALL Actual", f"{current_greeks['Delta Call']:.4f}")
            st.metric("Delta PUT Actual", f"{current_greeks['Delta Put']:.4f}")
        with cols[1]:
            st.metric("Gamma Actual", f"{current_greeks['Gamma']:.6f}")
        
        st.pyplot(plot_delta_gamma(S_range, deltas, gammas, S))

with tab3:
    if st.button("Calcular Vega, Theta y Rho"):
        vegas = []
        thetas = {'call': [], 'put': []}
        rhos = {'call': [], 'put': []}
        
        for s in S_range:
            greeks = calculate_greeks(s, K, days_to_expiry, r, sigma)
            vegas.append(greeks['Vega'])
            thetas['call'].append(greeks['Theta Call'])
            thetas['put'].append(greeks['Theta Put'])
            rhos['call'].append(greeks['Rho Call'])
            rhos['put'].append(greeks['Rho Put'])
        
        current_greeks = calculate_greeks(S, K, days_to_expiry, r, sigma)
        
        cols = st.columns(3)
        with cols[0]:
            st.metric("Vega Actual", f"{current_greeks['Vega']:.4f}")
        with cols[1]:
            st.metric("Theta CALL Actual", f"{current_greeks['Theta Call']:.4f}/día")
            st.metric("Theta PUT Actual", f"{current_greeks['Theta Put']:.4f}/día")
        with cols[2]:
            st.metric("Rho CALL Actual", f"{current_greeks['Rho Call']:.4f}")
            st.metric("Rho PUT Actual", f"{current_greeks['Rho Put']:.4f}")
        
        st.pyplot(plot_vega_theta_rho(S_range, vegas, thetas, rhos, S))

# Nota legal
st.caption("⚠️ Este análisis es informativo. Consulte con un profesional antes de invertir.")
