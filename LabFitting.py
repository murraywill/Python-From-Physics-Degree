import sympy as sp
from sympy import symbols, integrate, sin, pi, exp, Heaviside
import math
t_ = symbols('t_')
lamda = symbols('λ')
std = symbols('std')
t = symbols('t')
f = sp.exp(t_ * (-lamda - t_ * (1/(2 * std**2)) + t/std**2)) * sp.Heaviside(t_)

result = integrate(f, (t_, -sp.oo, sp.oo))
print(result)
S_i = 1
std = 1
lamda_i = 1
x = 0

T = 13.1 * 10**3
y1 = S_i * sp.exp((std**2 * lamda_i**2)/2 - x * lamda_i) *( (1/sp.exp(lamda_i * T) - 1) + (1 + sp.erf(x / (sp.sqrt(2) * std) - (std * lamda_i)/sp.sqrt(2)))/2)
print(y1)

y2 = -0.824360635350064 - 0.824360635350064*math.erf(math.sqrt(2)/2)
print((y2))