import sympy as sp
from sympy import symbols, integrate, sin, pi, exp, Heaviside

t_ = symbols('t_')
lamda = symbols('λ', positive = True, real = True)
std = symbols('std', positive = True, real = True)
t = symbols('t')
s_0 = symbols('S_0')
#f = sp.exp(t_ * (-lamda - t_ * (1/(2 * std**2)) + t/std**2)) * sp.Heaviside(t_)
f1 = s_0 * sp.exp(-t ** 2 / (2 * std**2)) * (1 / (sp.sqrt(2 * pi * std**2))) * sp.exp(t_ * (-lamda - t_ * (1/(2 * std**2)) + t/std**2)) * sp.Heaviside(t_)
g = s_0 * (sp.exp(-t_ * lamda) * sp.Heaviside(t_) * sp.exp(-(t - t_)**2 / (2 * std**2))) / (sp.sqrt( 2* pi * std**2))

result = integrate(g, (t_, -sp.oo, sp.oo))
print(result)

