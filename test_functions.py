import math
import numpy as np

def Rastrigin(n):
    #f(x) = A*n + [sum (x_i² - A*cos(2*pi*x_i)) i = 1 .. n]
    A = 10
    def f(x):
        return A*n + sum(map(lambda x: x**2-A*math.cos(2*math.pi*x), x))
    def grad_f(x):
        grad = []
        for i in range(len(x)):
            grad.append(2*x[i] + 2*math.pi*A*math.sin(2*math.pi*x[i]))
        return np.array(grad)
    def hess_f(x):
        hess = [[2+(A*2*math.pi)**2*math.cos(2*math.pi*x[i]) if i==j else 0 for j in range(n)] for i in range(n)]
        return np.array(hess)
    return {"function": f, "gradient":grad_f, "hessian":hess_f, "search_region": None, "global_minimum":[0 for i in range(n)], "name":"Rastrigin"}

def Rosenbrock(n):
    # f(x, y) = 100(y-x²)² + (1-x)²
    if n != 2:
        return None
    def f(entry):
        x, y = entry[0], entry[1]
        return 100*(y-x**2)**2 + (1-x)**2
    def grad_f(entry):
        #return aprox_grad(f, entry)
        x, y = entry[0], entry[1]
        return np.array([2*(200*x**3-200*x*y+x-1), 200*(y-x**2)])
    def hess_f(entry):
        x, y = entry[0], entry[1]
        return np.array([[-400*(y-x**2)+800*x**2+2, -400*x],[-400*x, 200]])
    return {"function": f, "gradient": grad_f, "hessian": hess_f, "search_region": None, "global_minimum":[1 for i in range(n)], "name":"Rosenbrock"}

def Sphere(n):
    # f(x) = sum x_i² i = 1 .. n
    def f(x):
        return sum(map(lambda it: it**2, x))
    def grad_f(x):
        grad = [2*x[i] for i in range(n)]
        return np.array(grad)
    def hess_f(x):
        hess = [[2 if i==j else 0 for j in range(n)] for i in range(n)]
        return np.array(hess)
    return {"function": f, "gradient":grad_f, "hessian":hess_f, "search_region":None, "global_minimum":[0,0], "name":"Sphere"}

def Goldstein_price(n):
    if n != 2:
        return None
    def f(entry):
        x, y = entry[0], entry[1]
        return (1+(x+y+1)**2*(19-14*x+3*x**2-14*y+6*x*y+3*y**2))*(30+(2*x-3*y)**2*(18-32*x+12*x**2+48*y-36*x*y+27*y**2))
    def grad_f(entry):
        return aprox_grad(f, entry)
    return {"function": f, "gradient":grad_f, "hessian":None, "search_region": [-2,2], "global_minimum":[0,-1], "name":"Goldstein price"}
 
def aprox_deriv(f, x, h=10**(-4)):  # symetric difference quotient
    dy = f(x+h) - f(x-h)
    return dy/(2*h)

def aprox_grad(f, x):
    grad = []
    for i in range(len(x)):
        fs = lambda inp: f([*x[0:i], inp, *x[(i+1):len(x)]])
        grad.append(aprox_deriv(fs, x[i]))
    #fs = [lambda inp: f([*x[0:i], inp, *x[(i+1):len(x)]]) for i in range(len(x))]
    return np.array(grad)

def sph(x):
    return x[0]**2+x[1]**2

fns = [Rastrigin, Rosenbrock, Sphere, Goldstein_price]