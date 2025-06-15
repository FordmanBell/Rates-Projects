import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.optimize import curve_fit


def generate_guess(local_strikes, local_sigmas, s0, f, b, B, T, plot=False):
    def eval_cubic_func(Xs, coef0, coef1, coef2):
        return [coef0 + coef1*x + coef2 * x ** 2 for x in Xs]

    # Fit coefficients
    Xs = [np.log((s+b)/(f+b)) for s in local_strikes]
    coefs, _ = curve_fit(eval_cubic_func, Xs, local_sigmas, p0=(1, 1, 1), maxfev=10000)

    if plot:
        curve_xs = np.linspace(min(Xs), max(Xs), 100)
        curve_ys = [coefs[0] + coefs[1]*x + coefs[2]*x**2 for x in curve_xs]
        plt.scatter(Xs, local_sigmas)        
        plt.plot(curve_xs, curve_ys)
        plt.title("ATM-Fitted Polynomial")
        plt.xlabel("Distance from ATM (bps)")
        plt.ylabel("Implied Volatility")

    # Get guesses
    v_0_sq = (3 * coefs[0] * coefs[2]) - (
        1/2 * (1-B)**2 * coefs[0]**2) + (
        3/2*(2*coefs[1] + (1-B) * coefs[0]) ** 2)
    
    p_0 = 1/np.sqrt(v_0_sq) * (2 * coefs[1] + (1-B) * coefs[0])
    
    if v_0_sq < 0:
        p_0 = 2 * coefs[1] + (1-B) * coefs[0]
        v_0 = 1 / p_0 * (2 * coefs[1] + (1-B) * coefs[0])
    else:
        v_0 = np.sqrt(v_0_sq)

    return p_0, v_0


def sigma(f, K, b, a, B, p, v, T):
    # Helpers
    def chi(z, p, v):
        in_log = (np.sqrt(1 - 2*p*z + z**2) - p + z)/(1-p)
        return  1/v * np.log(in_log)
    
    def zed(f, K, b, a, B, v):
        return v / (a * (1-B)) * ((f+b)**(1-B) - (K+b)**(1-B))
    
    def g(f, K, b, a, B):
        part1 = 1/24 * (B-1)**2*(f+b)**(B-1)
        return part1 * (K+b)**(B-1)*a**2

    # Calculation
    if K != f:
        coef1 = 1 / (chi(zed(f, K, b, a, B, v), p, v))
        coef2 = np.log((f+b)/(K+b))
        term1 = g(f, K, b, a, B)
        term21 = 1/4 * p * v * a * B * (f+b) ** ((B-1)/2)
        term2 = term21 * (K + b) ** ((B-1)/2)
        term3 = 1/24 * (2 - 3*p**2) * v**2
        return coef1 * coef2 * (1+(term1 + term2 + term3) * T)
    else:
        coef = a * (f+b) ** (B-1)
        term1 = g(f, K, b, a, B)
        term2 = 1/4 * p * v * a * B * (f+b)**(B-1)
        term3 = 1/24 * (2 - 3 * p**2) * v**2
        main = 1 + (term1 + term2 + term3)*T
        return coef * main


def alpha_from_sigma(s0, f, b, B, p, v, T):
    coef1 = ((1-B)**2 * T)/(24 * (f+b)**(2-2*B))
    coef2 = (p*B*v*T) / (4 * (f+b)**(1-B))
    coef3 = (1 + (2-3*p**2)/24 * v**2 * T)
    coef4 = -s0 * (f+b)**(1-B)
    r = np.roots([coef1, coef2, coef3, coef4])
    return r[(r.imag==0) & (r.real>=0)].real.min() 


def estimate_alpha_rho_nu(strikes, sigmas, s0, f, b, B, T, guess = [0, 1]):
    def eval_sigma(Ks, p, v):
        a_from_sigma = alpha_from_sigma(s0, f, b, B, p, v, T)
        return [sigma(f, K, b, a_from_sigma, B, p, v, T) for K in Ks]
    return curve_fit(eval_sigma, strikes, sigmas, p0=guess, maxfev=int(1e6))


def plot_smile(f, b, a, B, p, v, T, strikes, sigmas=[]):
    curve_xs = np.linspace(strikes.min(), strikes.max(), 100)
    curve_ys = [sigma(f, x, b, a, B, p, v, T) for x in curve_xs]

    plt.figure(figsize=(8, 5))

    # Plot SABR model curve
    plt.plot(curve_xs, curve_ys, label='SABR Curve')

    # Plot market vol points if provided
    if len(sigmas) > 0:
        plt.scatter(strikes, sigmas, color='red', label='Market Vols')

    # Labels and title
    plt.xlabel("Strike")
    plt.ylabel("Implied Volatility")
    plt.title("Implied Volatility vs. Strike\n"
              + fr"$\alpha$={a:.4f}, $\beta$={B:.2f}, $\rho$={p:.4f}, $\nu$={v:.4f}, f={f:.4f}, b={b:.4f}, T={T:.2f}")
    
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
