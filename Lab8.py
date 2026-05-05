import math
import cmath

# --- 1. Табуляція функції ---
def F(x):
    return math.sin(x) - 0.5 * x  # приклад трансцендентної функції

def tabulate(a, b, h, filename="table.txt"):
    with open(filename, "w") as f:
        x = a
        while x <= b:
            y = F(x)
            f.write(f"{x:.4f}\t{y:.6f}\n")
            x += h
    print(f"Табуляція завершена. Дані записано у {filename}")

# --- 2. Методи розв’язку нелінійних рівнянь ---
def simple_iteration(x0, eps=1e-10, max_iter=1000):
    def f(x): return x - 0.5 * (math.sin(x) - 0.5 * x)
    for i in range(max_iter):
        x1 = f(x0)
        if abs(F(x1)) < eps and abs(x1 - x0) < eps:
            return x1, i + 1
        x0 = x1
    return x1, max_iter

def newton(x0, eps=1e-10, max_iter=1000):
    for i in range(max_iter):
        x1 = x0 - F(x0) / (math.cos(x0) - 0.5)
        if abs(F(x1)) < eps and abs(x1 - x0) < eps:
            return x1, i + 1
        x0 = x1
    return x1, max_iter

def chebyshev(x0, eps=1e-10, max_iter=1000):
    for i in range(max_iter):
        f = F(x0)
        f1 = math.cos(x0) - 0.5
        f2 = -math.sin(x0)
        if abs(f1) < 1e-12:  # захист від ділення на нуль
            return x0, i
        correction = (f2 * f**2) / (2 * f1**3)
        # обмеження величини поправки
        if abs(correction) > 1e6:
            correction = math.copysign(1e6, correction)
        x1 = x0 - f / f1 - correction
        if abs(F(x1)) < eps and abs(x1 - x0) < eps:
            return x1, i + 1
        x0 = x1
    return x1, max_iter

def secant(x0, x1, eps=1e-10, max_iter=1000):
    for i in range(max_iter):
        x2 = x1 - F(x1) * (x1 - x0) / (F(x1) - F(x0))
        if abs(F(x2)) < eps and abs(x2 - x1) < eps:
            return x2, i + 1
        x0, x1 = x1, x2
    return x2, max_iter

# --- 3. Алгебраїчне рівняння третього порядку ---
def polynomial(x, coeffs):
    return sum(c * x**i for i, c in enumerate(coeffs))

def horner_method(coeffs, x0, eps=1e-10, max_iter=1000):
    for i in range(max_iter):
        # схема Горнера для F(x)
        b = [coeffs[-1]]
        for a in reversed(coeffs[:-1]):
            b.insert(0, a + x0 * b[0])
        # схема Горнера для похідної F'(x)
        c = [b[-1]]
        for bi in reversed(b[:-1]):
            c.insert(0, bi + x0 * c[0])
        if abs(c[0]) < 1e-12:  # захист від ділення на нуль
            return x0, i
        x1 = x0 - b[0] / c[0]
        if abs(x1 - x0) < eps:
            return x1, i + 1
        x0 = x1
    return x1, max_iter


# --- 4. Метод Ліна для комплексних коренів ---
def lin_method(coeffs, a0, b0, eps=1e-10, max_iter=1000):
    a, b = a0, b0
    for i in range(max_iter):
        p = -2 * a
        q = a**2 + b**2
        m = len(coeffs) - 1
        b_arr = [0] * (m + 1)
        b_arr[m] = coeffs[m]
        for j in range(m - 1, -1, -1):
            b_arr[j] = coeffs[j] + p * b_arr[j + 1] + q * (b_arr[j + 2] if j + 2 <= m else 0)
        a_new = -p / 2
        b_new = math.sqrt(q - a_new**2)
        if abs(a_new - a) < eps and abs(b_new - b) < eps:
            return complex(a_new, b_new), i + 1
        a, b = a_new, b_new
    return complex(a, b), max_iter

# --- 5. Демонстрація ---
if __name__ == "__main__":
    tabulate(-5, 5, 0.1, filename="/Users/nataliiatymoshenko/Desktop/q/2 .2/methods/Lab8/table.txt")
    print("Метод простої ітерації:", simple_iteration(1))
    print("Метод Ньютона:", newton(1))
    print("Метод Чебишева:", chebyshev(2))
    print("Метод хорд:", secant(0, 2))

    coeffs = [2, -3, 0, 1]  # приклад кубічного рівняння
    print("Метод Горнера:", horner_method(coeffs, 1))
    print("Метод Ліна:", lin_method(coeffs, 1, 1))
