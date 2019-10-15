import numpy as np
import pandas as pd
import sys
import minimize as mini

# f(x, y) = 100(y-x²)² + (1-x)²
def f(entry):
    x, y = entry[0], entry[1]
    return 100*(y-x**2)**2 + (1-x)**2
def grad_f(entry):
    x, y = entry[0], entry[1]
    return np.array([2*(200*x**3-200*x*y+x-1), 200*(y-x**2)])
def hess_f(entry):
    x, y = entry[0], entry[1]
    return np.array([[-400*(y-x**2)+800*x**2+2, -400*x],[-400*x, 200]])

mims = []
methods = ["newton", "BFGS", "gradient"]
colors = ["darkblue", "yellow", "purple"]
for i in range(len(methods)):
    mims.append(mini.Minimizer(f, 2, np.array([0, 0])))
    mims[i].f_grad = grad_f
    mims[i].f_hess = hess_f
    mims[i].iterate(method=methods[i], log=True)

vet_f = np.vectorize(lambda in1, in2: f((in1, in2)))
drawer = mini.Drawer()
drawer.draw_f(vet_f, mims[0])
for i in range(len(methods)):
    drawer.draw_path(vet_f, mims[i], mims[i].x, color=colors[i])
drawer.show()