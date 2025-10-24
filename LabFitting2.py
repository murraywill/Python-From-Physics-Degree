import sympy as sp
from sympy import symbols, integrate, sin, pi, exp, Heaviside

t_ = symbols('t_')
lamda = symbols('λ', positive = True, real = True)
std = symbols('std', positive = True, real = True)
t = symbols('t')
f = sp.exp(t_ * (-lamda + t_ * (1/(2 * std**2)) - t/std**2)) * sp.Heaviside(t_)

result = integrate(f, t_)
print(result)