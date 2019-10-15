import numpy as np
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
    def iterate(self, n=2000, delta=10e-4, method='default', log=False):
        for _ in range(2000):
            iter = len(self.x) - 1
            x = self.x[-1]
            print('x_%d = [%f,%f], f(x_%d) = %f, grad_f(x_%d) =' % (iter, round(x[0].item(),4), round(x[1].item(),4), iter, round(self.f(x).item(), 4), iter), self.f_grad(x))
            if np.linalg.norm(self.f_grad(self.x[-1])) < delta:
                break
            st = self.step(method=method)
            print(st)