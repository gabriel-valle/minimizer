import numpy as np
import pandas as pd
import sys
import minimize as mini

delta = 10e-4
if len(sys.argv) > 1:
    method = sys.argv[1]
else:
    method = "newton"

print('method:',method) 

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

mim = mini.Minimizer(f, 2, np.array([0, 0]))
mim.f_grad = grad_f
mim.f_hess = hess_f
mim.iterate(method=method, log=True)

vet_f = np.vectorize(lambda in1, in2: f((in1, in2)))
drawer = mini.Drawer()
drawer.draw_f(vet_f, mim)
drawer.draw_path(vet_f, mim, mim.x)
drawer.show()