import numpy as np
import math
import pandas as pd
import sys
import minimize as mini
import test_functions
from test_functions import fns

delta = 10e-4
if len(sys.argv) > 1:
    method = sys.argv[1]
else:
    method = "gradient"

print('method:',method)

for item in fns:
    for n in range(2, 5):
        fn = item(n)
        if fn == None:  # Verifies if n is a valid dimension for the problem
            break
        if 'name' in fn:
            print(n, '-D, ', fn['name'], sep='')
        if fn['search_region'] == None:
            fn['search_region'] = [-500, 500]
        x_0 = np.random.uniform(fn['search_region'][0], fn['search_region'][1], n)
        print('x_0 =', x_0)
        mim = mini.Minimizer(fn['function'], n, x_0)
        mim.f_grad = fn['gradient']
        mim.f_hess = fn['hessian']
        mim.iterate(method=method, log=True, n=10000)
        print(mim.pretty_print())
        print("minimo global:", fn['global_minimum'])
        print()