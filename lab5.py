import numpy as np
import matplotlib.pyplot as plt
from math import cos, pi, sqrt, erf, log

# -------- ФУНКЦІЯ --------
def f(x):
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12)**2)

# -------- ТОЧНИЙ ІНТЕГРАЛ --------
def exact_integral(a, b):
    part1 = 50 * (b - a)

    part2 = (-240 / pi) * (cos(pi * b / 12) - cos(pi * a / 12))

    alpha = 0.2
    coef = 5 * sqrt(pi / alpha) / 2
    part3 = coef * (erf(sqrt(alpha) * (b - 12)) - erf(sqrt(alpha) * (a - 12)))

    return part1 + part2 + part3

# -------- СІМПСОН --------
def simpson(a, b, N):
    if N % 2 != 0:
        N += 1

    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f(x)

    S = y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2])
    return S * h / 3

# -------- ЕЙТКЕН --------
def aitken(I1, I2, I3):
    return log(abs(I1 - I2) / abs(I2 - I3)) / log(2)

# -------- АДАПТИВНИЙ СІМПСОН --------
def adaptive_simpson(f, a, b, eps):
    def recursive(a, b, fa, fm, fb, S, eps):
        m = (a + b) / 2
        lm = (a + m) / 2
        rm = (m + b) / 2

        fl = f(lm)
        fr = f(rm)

        S_left = (fa + 4 * fl + fm) * (m - a) / 6
        S_right = (fm + 4 * fr + fb) * (b - m) / 6

        if abs(S_left + S_right - S) < 15 * eps:
            return S_left + S_right + (S_left + S_right - S) / 15

        return (recursive(a, m, fa, fl, fm, S_left, eps / 2) +
                recursive(m, b, fm, fr, fb, S_right, eps / 2))

    fa = f(a)
    fb = f(b)
    m = (a + b) / 2
    fm = f(m)

    S = (fa + 4 * fm + fb) * (b - a) / 6

    return recursive(a, b, fa, fm, fb, S, eps)

# -------- MAIN --------
a, b = 0, 24
I0 = exact_integral(a, b)

print("Точний інтеграл:", I0)

# -------- ПОМИЛКА ВІД N --------
N_values = []
eps_values = []

Nopt = None

for N in range(10, 1000):
    if N % 2 != 0:
        continue

    IN = simpson(a, b, N)
    eps = abs(IN - I0)

    N_values.append(N)
    eps_values.append(eps)

    if Nopt is None and eps < 1e-12:
        Nopt = N

print("Nopt =", Nopt)

# -------- N0 --------
N0 = max(8, int((Nopt or 80) / 10))
while N0 % 8 != 0:
    N0 += 1

IN0 = simpson(a, b, N0)
eps0 = abs(IN0 - I0)

print("N0 =", N0)
print("eps0 =", eps0)

# -------- РУНГЕ-РОМБЕРГ --------
IN_half = simpson(a, b, N0 // 2)
IR = IN0 + (IN0 - IN_half) / 15
epsR = abs(IR - I0)

print("Runge:", IR, "error:", epsR)

# -------- ЕЙТКЕН --------
I1 = simpson(a, b, N0 // 4)
I2 = simpson(a, b, N0 // 2)
I3 = simpson(a, b, N0)

p = aitken(I1, I2, I3)
IA = I3 + (I3 - I2) / (2**p - 1)
epsA = abs(IA - I0)

print("Aitken p =", p)
print("Aitken I =", IA, "error:", epsA)

# -------- АДАПТИВНИЙ --------
for eps in [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]:
    I_ad = adaptive_simpson(f, a, b, eps)
    print("Adaptive eps =", eps, "result =", I_ad, "error =", abs(I_ad - I0))

# -------- ГРАФІК --------
import matplotlib.pyplot as plt

plt.plot(N_values, eps_values)
plt.yscale("log")
plt.xlabel("N")
plt.ylabel("error")
plt.title("Залежність похибки від N")
plt.grid()
plt.show()