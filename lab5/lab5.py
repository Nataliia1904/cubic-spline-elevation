import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# =========================================================
# 1. ЗАДАНА ФУНКЦІЯ
# навантаження на сервер 
# =========================================================
def f(x):
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12)**2)

a, b = 0, 24

# =========================================================
# 2. ТОЧНЕ ЗНАЧЕННЯ ІНТЕГРАЛУ
# =========================================================
I0, _ = quad(f, a, b)

print("=== Точне значення інтегралу ===")
print("I0 =", I0)

# =========================================================
# 3. СКЛАДОВА ФОРМУЛА СІМПСОНА

# =========================================================
def simpson(f, a, b, N):
    if N % 2 != 0:
        raise ValueError("Для формули Сімпсона N має бути парним.")

    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f(x)

    odd_sum = np.sum(y[1:N:2])     # f1 + f3 + ... + fN-1
    even_sum = np.sum(y[2:N-1:2])  # f2 + f4 + ... + fN-2

    I = (h / 3) * (y[0] + 4 * odd_sum + 2 * even_sum + y[N])
    return I

# =========================================================
# 4. ДОСЛІДЖЕННЯ похибки eps(N) = |I(N) - I0|

# =========================================================
eps_target = 1e-12
N_values = np.arange(10, 1001, 2)   # парні N
errors = []
integrals = []

for N in N_values:
    I_N = simpson(f, a, b, N)
    integrals.append(I_N)
    errors.append(abs(I_N - I0))

N_opt = None
eps_opt = None

for N, err in zip(N_values, errors):
    if err <= eps_target:
        N_opt = N
        eps_opt = err
        break

print("\n=== Пошук Nopt ===")
if N_opt is not None:
    print("Nopt =", N_opt)
    print("epsopt =", eps_opt)
else:
    print("На відрізку N = 10..1000 точність 1e-12 не досягнута.")

# =========================================================
# 5. ВИБІР N0 ~ Nopt / 10, N0 кратне 8

# =========================================================
if N_opt is not None:
    N0 = max(8, int(round((N_opt / 10) / 8) * 8))
else:
    N0 = 16

# підстраховка
if N0 % 8 != 0:
    N0 += 8 - (N0 % 8)

I_N0 = simpson(f, a, b, N0)
eps0 = abs(I_N0 - I0)

print("\n=== Обчислення при N0 ===")
print("N0 =", N0)
print("I(N0) =", I_N0)
print("eps0 =", eps0)

# =========================================================
# 6. МЕТОД РУНГЕ-РОМБЕРГА

# =========================================================
I_N0_2 = simpson(f, a, b, N0 // 2)
I_R = I_N0 + (I_N0 - I_N0_2) / 15
eps_R = abs(I_R - I0)

print("\n=== Метод Рунге-Ромберга ===")
print("I(N0/2) =", I_N0_2)
print("IR =", I_R)
print("epsR =", eps_R)

# =========================================================
# 7. МЕТОД ЕЙТКЕНА

# =========================================================
I_N0_4 = simpson(f, a, b, N0 // 4)

num = I_N0 - I_N0_2
den = I_N0_2 - I_N0_4

if abs(den) < 1e-15 or abs(num) < 1e-15:
    p_A = None
    I_A = I_N0
else:
    ratio = num / den
    p_A = -np.log2(abs(ratio))
    I_A = I_N0 + (I_N0 - I_N0_2) / (2**p_A - 1)

eps_A = abs(I_A - I0)

print("\n=== Метод Ейткена ===")
print("I(N0/4) =", I_N0_4)
print("I(N0/2) =", I_N0_2)
print("I(N0)   =", I_N0)

if p_A is not None:
    print("pA =", p_A)
    print("IA =", I_A)
else:
    print("pA не вдалося обчислити коректно.")
    print("IA =", I_A)

print("epsA =", eps_A)

# =========================================================
# 8. АДАПТИВНИЙ АЛГОРИТМ СІМПСОНА
# =========================================================
def simpson_local(f, a, b):
    c = (a + b) / 2
    return (b - a) * (f(a) + 4 * f(c) + f(b)) / 6

def adaptive_simpson(f, a, b, eps):
    eval_count = 0

    def recursive(a, b, eps, whole):
        nonlocal eval_count
        c = (a + b) / 2

        left = simpson_local(f, a, c)
        right = simpson_local(f, c, b)
        eval_count += 2  # умовно рахуємо нові локальні обчислення

        if abs(left + right - whole) <= 15 * eps:
            return left + right + (left + right - whole) / 15

        return recursive(a, c, eps / 2, left) + recursive(c, b, eps / 2, right)

    whole = simpson_local(f, a, b)
    eval_count += 3
    I_ad = recursive(a, b, eps, whole)
    return I_ad, eval_count

eps_list = [1e-2, 1e-4, 1e-6, 1e-8]
adaptive_results = []

print("\n=== Адаптивний алгоритм ===")
for eps in eps_list:
    I_ad, calls = adaptive_simpson(f, a, b, eps)
    err = abs(I_ad - I0)
    adaptive_results.append((eps, I_ad, err, calls))
    print(f"eps = {eps:1.0e} | I = {I_ad:.15f} | error = {err:.3e} | calls = {calls}")

# =========================================================
# 9. ПОРІВНЯННЯ
# =========================================================
print("\n=== Порівняння методів ===")
print("I0 (точне)      =", I0)
print("I(N0)           =", I_N0)
print("IR              =", I_R)
print("IA              =", I_A)
print("eps0            =", eps0)
print("epsR            =", eps_R)
print("epsA            =", eps_A)

# =========================================================
# 10. ГРАФІКИ
# =========================================================
x_plot = np.linspace(a, b, 1000)
y_plot = f(x_plot)

plt.figure(figsize=(7, 12))

plt.subplot(3, 1, 1)
plt.plot(x_plot, y_plot)
plt.title("Графік функції навантаження на сервер")
plt.xlabel("Час, x (год)")
plt.ylabel("Навантаження, f(x)")
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(N_values, errors)
plt.title("Залежність похибки eps(N) = |I(N) - I0|")
plt.xlabel("Кількість розбиттів N")
plt.ylabel("Похибка")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.semilogy(N_values, errors)
plt.title("Логарифмічний графік похибки")
plt.xlabel("Кількість розбиттів N")
plt.ylabel("log(eps(N))")
plt.grid(True)

plt.tight_layout()
plt.show()