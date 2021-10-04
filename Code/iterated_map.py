import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint

N = 10000
MAXTIME = 100

def plot(S, I, R, maxtime, S2 = None, I2 = None, R2 = None):
    fig, (ax1, ax2, ax3) = plt.subplots(nrows = 3)
    ax1.plot(range(0, maxtime), S, label = "SIR Model")
    ax1.set_title("Susceptible")
    ax2.plot(range(0, maxtime), I)
    ax2.set_title("Infected")
    ax3.plot(range(0, maxtime), R)
    ax3.set_title("Removed")

    if S2 is not None:
        ax1.plot(range(0, maxtime), S2, label = "p-based Model")
        ax2.plot(range(0, maxtime), I2)
        ax3.plot(range(0, maxtime), R2)

    ax1.legend()
    plt.tight_layout()
    plt.show()

def simulate_sir(alpha = 0.00013, maxtime = MAXTIME, n = 10000):
    # Alpha and n are chosen so that alpha * S[0] is slightly larger than 1
    # Maxtime is chosen by inspection
    S = np.zeros(maxtime)
    I = np.zeros(maxtime)
    R = np.zeros(maxtime)

    S[0] = n - 1
    I[0] = 1
    R[0] = 0

    for t in range(0, maxtime - 1):
        S[t + 1] = S[t] - alpha * I[t] * S[t]
        I[t + 1] = alpha * I[t] * S[t]
        R[t + 1] = R[t] + I[t]

    return (S, I, R)

def simulate_agent(p = 0.00013, maxtime = MAXTIME, n = 10000):
    print("------------------------")
    print(" Simulating with p = ", p)
    print("------------------------")
    S = np.zeros(maxtime)
    I = np.zeros(maxtime)
    R = np.zeros(maxtime)

    S[0] = n - 1
    I[0] = 1
    R[0] = 0

    for t in range(0, maxtime - 1):
        print("TIME STEP", t)
        print("     S = ", S[t])
        print("     I = ", I[t])
        print("     R = ", R[t])
        S[t + 1] = S[t] - (1 - (1 - p)**I[t]) * S[t]
        I[t + 1] = (1 - (1 - p)**I[t]) * S[t]
        R[t + 1] = R[t] + I[t]

    return (S, I, R)

def quality_of_fit(I_1, I_2):
    return np.sum(np.abs(I_1 - I_2))

def evaluate_p(p, maxtime, I_original):
    (S, I, R) = simulate_agent(p, maxtime, n = 10000)
    return quality_of_fit(I_original, I)

def fit_agent(I_original):
    x0 = 0.0004
    result = minimize(fun = evaluate_p, x0 = x0, args = (MAXTIME, I_original), bounds = [(0,1)])
    print("BEST PROBABILITY:", result.x)
    print("BEST QUALITY:", quality_of_fit(I_original, result.x))
    print("INITIAL QUALITY:", quality_of_fit(I_original, x0))
    return result.x


(S, I, R) = simulate_sir(alpha = 0.00011)
p = fit_agent(I)
(S2, I2, R2) = simulate_agent(p = p)
plot(S, I, R, MAXTIME, S2, I2, R2)
print("p = ", p)