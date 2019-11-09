import numpy as np
import plotly.graph_objects as go
import math
import pandas as pd
class Minimizer:
    def __init__(self, f, inputs, x_0 = np.array([0,0]), alpha = 0.1):
        self.f = f
        self.inputs = inputs
        self.f_grad = lambda inp: inp
        self.f_hess = lambda inp: np.identity(inputs)
        self.x = [x_0]
        self.steps = []
        self.alpha = alpha
        self.B = []
        self.log = []

    def line_search(self, dr):
        t = 1
        x = self.x[-1]
        #Armijo condition
        while self.f(x + t*dr) >= self.f(x) + self.alpha*t*np.inner(self.f_grad(x), dr):
            if t == 0:
                #print(dr, self.f(x), self.alpha*t*np.inner(self.f_grad(x), dr))
                print('line_search failed:')
                print('x =', *x)
                print('grad_x =', self.f_grad(x))
                print('f(x) =', self.f(x))
                return -1
            t /= 2
        return t

    def gradient_step(self):
        x = self.x[-1]
        direction = -self.f_grad(x)
        t = self.line_search(direction)
        if t == -1:
            return None
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
        if t == -1:
            return None
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
            st = self.BFGS_step()
        elif method == 'newton':
            st = self.newton_step()
        else: # gradient descent is default
            st = self.gradient_step()
        return (st, method)
    def iterate(self, n=5000, delta=10e-4, method='default', log=False):
        for _ in range(n):
            #iter = len(self.x) - 1
            x = self.x[-1]
            f_x = self.f(x)
            grad = self.f_grad(x)
            if log:
                self.log.append([list(map(lambda it: round(it, 3), x)), round(f_x, 3), list(map(lambda it: round(it,3),grad)), method])
                #round_x = map(lambda x: round(x, 4), x)
                #round_grad = map(lambda x: round(x, 4), grad)
                #print('k = ',iter,', x = (', *round_x, '), f(x) = ', round(f_x, 4), ', grad_f(x) = (', *round_grad, ')', sep='')
            #print('x_%d = [%f,%f], f(x_%d) = %f, grad_f(x_%d) =' % (iter, round(x[0].item(),4), round(x[1].item(),4), iter, round(f_x, 4), iter), grad)
            if delta != None and np.linalg.norm(self.f_grad(self.x[-1])) < delta:
                break
            st = self.step(method=method)
            if st[0] is None:
                print('iteration interrupted')
                break
            #print(st)
    def pretty_print(self):
        return pd.DataFrame(self.log, columns=["x", "f(x)", "grad_f(x)", "method"])

class Drawer:
    def __init__(self, res = 0.05, margin = 0.2, min_dist = 10e-2):
        self.fig = None
        self.res = res
        self.margin = margin
        self.min_dist = min_dist
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
            self.view_area = [[-rangex, -rangey],[rangex, rangey]]
        X = np.arange(self.view_area[0][0], self.view_area[1][0]+self.res, self.res)
        Y = np.arange(self.view_area[0][1], self.view_area[1][1]+self.res, self.res)
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
    def draw_curve(self, vet_f, mim, X_0, X_f, color='darkblue'):
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
        X_line = path[:, 0]
        Y_line = path[:, 1]
        Z_line = vet_f(X_line, Y_line)
        self.fig.add_trace(go.Scatter3d(x=X_line, y=Y_line, z=Z_line, mode='lines',
            line=dict(
                color='darkblue',
                width=10
            ))
        )
        return self.fig
    def draw_path(self, vet_f, mim, path, color='darkblue', projection=False, density = 20):
        path = np.array(path)
        #smooth_len = np.norm(path[-1] - path[0])*density
        smooth_path = []
        for i in range(len(path)-1):
            X_0, X_f = path[i], path[i+1]
            steps = math.floor(np.linalg.norm(X_f - X_0)*density)
            for t in range(0,steps):
                vet = X_0 + (t/steps)*(X_f-X_0)
                smooth_path.append(list(vet))
        smooth_path = np.array(smooth_path)
        X_scatter = smooth_path[:, 0]
        Y_scatter = smooth_path[:, 1]
        Z_scatter = vet_f(X_scatter, Y_scatter)

        X_steps = path[:, 0]
        Y_steps = path[:, 1]
        Z_steps = vet_f(X_steps, Y_steps)

        mark_colors = np.array([0.01*i for i in range(len(X_steps))])

        #draw smooth path
        self.fig.add_trace(go.Scatter3d(x=X_scatter, y=Y_scatter, z=Z_scatter, mode='lines',
            line=dict(
                width = 10,
                color = color
            ), opacity=0.7)
        )

        #step markers
        self.fig.add_trace(go.Scatter3d(x=X_steps, y=Y_steps, z=Z_steps, mode='markers',
            marker=dict(
                size=5,
                color=mark_colors,
                colorscale='Viridis',
                opacity=0.8
            ))
        )
        if projection:
            self.fig.add_trace(go.Scatter3d(x=X_scatter, y=Y_scatter, z=np.zeros(X_scatter.shape), mode='lines',
                line=dict(
                    color=color,
                    width=10,
                ), opacity=0.7)
        )
    def draw_marker(self, vet_f, pos, color='black', symbol_type='square'):
        pos_z = vet_f(pos[0], pos[1])
        self.fig.add_trace(go.Scatter3d(x=[pos[0]], y=[pos[1]], z=[pos_z], mode='markers',
            marker=dict(
                size=4,
                color=color,
                opacity=0.8,
                symbol=symbol_type
            ))
        )
    def show(self):
        self.fig.show()