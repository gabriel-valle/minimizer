import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import math
import sys

delta = 10e-4
if len(sys.argv) > 1:
    method = sys.argv[1]
else:
    method = "newton"

print('method:',method)
class Minimizer:
    def __init__(self, f, inputs, x_0 = np.array([0,0]), alpha = 0.1):
        self.f = f
        self.inputs = inputs
        self.f_grad = None
        self.f_hess = None
        self.x = [x_0]
        self.steps = []
        self.alpha = alpha
        self.B = []

    def line_search(self, dr):
        t = 1
        x = self.x[-1]
        #Armijo condition
        while self.f(x + t*dr) >= self.f(x) + self.alpha*t*np.inner(self.f_grad(x), dr):
            if t == 0:
                print(dr, f(x), self.alpha*t*np.inner(self.f_grad(x), dr))
                print('line_search failed')
                break
            t /= 2
        return t

    def gradient_step(self):
        x = self.x[-1]
        direction = -self.f_grad(x)
        t = self.line_search(direction)
        print('t =', t)
        self.steps.append(t*direction)
        self.x.append(self.x[-1]+self.steps[-1])
        return t*direction
        #gradient method
    def newton_step(self):
        x = self.x[-1]
        step = np.linalg.solve(self.f_hess(x), -self.f_grad(x))
        self.steps.append(step)
        self.x.append(self.x[-1]+self.steps[-1])
        return step
    def BFGS_step(self):
        x = self.x[-1]
        direction = np.linalg.solve(self.B[-1], -self.f_grad(x))
        t = self.line_search(direction)
        step = t*direction
        self.steps.append(step)
        self.x.append(self.x[-1]+self.steps[-1])
        y = self.f_grad(self.x[-1]) - self.f_grad(self.x[-2])
        B_ = self.B[-1]
        B_0 = np.outer(y,y)/np.inner(y, step)
        B_1 = - np.outer(B_ @ step, B_ @ step)/(np.transpose(step) @ B_ @ step)
        #computes next Hessian approximation
        self.B.append(B_ + B_0 + B_1)
        return step

    def step(self, method = 'default'):
        if method == 'default' and self.f_hess != None:
            if np.array_equal(self.f_hess(self.x[-1]), np.transpose(self.f_hess(self.x[-1]))):
                try:
                    np.linalg.cholesky(self.f_hess(self.x[-1]))
                    method = 'newton'
                except np.linalg.LinAlgError:
                    print('non-positive definite matrix')
        if method == 'BFGS':
            if len(self.B) == 0:
                self.B.append(np.identity(self.inputs))
            self.BFGS_step()
        elif method == 'newton':
            self.newton_step()
        else: # gradient descent is default
            self.gradient_step()
        return (self.steps[-1], method)

class Drawer:
    def __init__(self, res = 0.2, margin = 0.2, min_dist = 10e-2):
        self.fig = None
        self.res = 0.2
        self.margin = 0.2
        self.min_dist = 10e-2/5
        self.view_area = None
    def draw_f(self, vet_f, mim):
        xs = list(map(lambda inp: inp[0], mim.x))
        ys = list(map(lambda inp: inp[1], mim.x))
        if self.view_area == None:
            x_low, x_high = min(xs), max(xs)
            y_low, y_high = min(ys), max(ys)
            rangex, rangey = max(abs(x_low), abs(x_high)), max(abs(y_low), abs(y_high))
            rangex *= 1+self.margin
            rangey *= 1+self.margin
            view_area = [[-rangex, -rangey],[rangex, rangey]]
        X = np.arange(view_area[0][0], view_area[1][0]+self.res, self.res)
        Y = np.arange(view_area[0][1], view_area[1][1]+self.res, self.res)
        X, Y = np.meshgrid(X, Y)
        Z = vet_f(X, Y)
        surface = go.Figure(data=[go.Surface(x = X, y = Y, z = Z, opacity=0.8)])
        if self.fig == None:
            self.fig = surface
        else:
            self.fig.data.append(surface.data[0])
        self.fig.update_layout(title='Function', autosize=True,
                    width=1000, height=1000,
                    margin=dict(l=65, r=50, b=65, t=90))
        return self.fig
    def draw_path(self, vet_f, mim, X_0, X_f, color='darkblue'):
        path = []
        norm = np.linalg.norm(X_f-X_0)
        steps = math.ceil(4*norm.item()/self.res)
        print('steps =', steps, 'norm =', norm.item(), 'steps=', 4*norm.item()/self.res)
        if steps == 0:
            return self.fig
        for t in range(0,steps+1):
            vet = X_0 + (t/steps)*(X_f-X_0)
            path.append(list(vet))
        path = np.array(path)
        X_line = np.array(list(map(lambda entry: entry[0], path)))
        Y_line = np.array(list(map(lambda entry: entry[1], path)))
        Z_line = vet_f(X_line, Y_line)
        self.fig.add_trace(go.Scatter3d(x=X_line, y=Y_line, z=Z_line, mode='lines',
            line=dict(
                #color=(color+np.sqrt(X_line**2+Y_line**2))/10,
                color='darkblue',
                #colorscale='Viridis',
                width=10
            ))
        )
        return self.fig
    def show(self):
        self.fig.show()  


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

mim = Minimizer(f, 2, np.array([0, 0]))
mim.f_grad = grad_f
mim.f_hess = hess_f

for i in range(2000):
    iter = len(mim.x) - 1
    x = mim.x[-1]
    print('x_%d = [%f,%f], f(x_%d) = %f, grad_f(x_%d) =' % (iter, round(x[0].item(),4), round(x[1].item(),4), iter, round(mim.f(x).item(), 4), iter), mim.f_grad(x))
    if np.linalg.norm(mim.f_grad(mim.x[-1])) < delta:
        break
    st = mim.step(method='newton')
    print(st)

vet_f = np.vectorize(lambda in1, in2: f((in1, in2)))
drawer = Drawer()
drawer.draw_f(vet_f, mim)
count = 0
i = 0
while i in range(len(mim.x)-1):
    j = i+1
    while j!=len(mim.x)-1 and np.linalg.norm(mim.x[j] - mim.x[i]) < drawer.min_dist:
        j+=1
    drawer.draw_path(vet_f, mim, mim.x[i], mim.x[j], count)
    count += np.linalg.norm(mim.x[j]-mim.x[i])
    print('path', i, j, 'drawn')
    i = j
drawer.show()
#fig.show()