# import numpy as np
import autograd.numpy as np
from autograd import grad
# from collections import namedtuple
from recordtype import recordtype

InitConds = recordtype('InitConds',['t0','tf','N_t','x0','y0','z0','p0','theta0','phi0'])
Bounds = recordtype('Bounds',['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'])

# a couple of canned examples
# default example
ic = InitConds(t0=0., tf=1e-8, N_t=1001, x0=0., y0=0., z0=0., p0=105., theta0=np.pi/6, phi0=0.)
bounds = Bounds(xmin=-0.95, xmax=0.95, ymin=-0.95, ymax=0.95, zmin=-1., zmax=1.)

# Mu2e examples
ic_Mu2e = InitConds(t0=0., tf=2e-7, N_t=20001,
                    x0=0.054094482, y0=0.03873037, z0=5.988900879,
                    p0=104.96, theta0=np.pi/3, phi0=0.)
ic_Mu2e_bounce = InitConds(t0=0., tf=2e-7, N_t=20001,
                           x0=0.054094482, y0=0.03873037, z0=5.988900879,
                           p0=104.96, theta0=np.pi-np.pi/3, phi0=0.)
bounds_Mu2e = Bounds(xmin=-0.95, xmax=0.95, ymin=-0.95, ymax=0.95, zmin=3.239, zmax=14.139)

# descent functions
def gradient_descent(g, alpha, max_its, w):
    # compute gradient module using autograd
    gradient = grad(g)
    # gradient descent loop
    weight_history = [w] # weight history container
    cost_history = [g(w)] # cost history container
    for k in range(max_its):
        # eval gradient
        grad_eval = gradient(w)
        # descent step
        w = w - alpha*grad_eval
        # record weight and cost
        weight_history.append(w)
        cost_history.append(g(w))

    return np.array(weight_history), np.array(cost_history)

def gradient_descent_dimalpha(g, alpha, max_its, w):
    # compute gradient module using autograd
    gradient = grad(g)
    # gradient descent loop
    weight_history = [w] # weight history container
    cost_history = [g(w)] # cost history container
    for k in range(max_its):
        # eval gradient
        grad_eval = gradient(w)
        # descent step
        w = w - alpha/(k+1)*grad_eval
        # record weight and cost
        weight_history.append(w)
        cost_history.append(g(w))

    return np.array(weight_history), np.array(cost_history)

def gradient_descent_mom_accel(g, alpha, beta, max_its, w):
    # compute gradient module using autograd
    gradient = grad(g)
    # gradient descent loop
    weight_history = [w] # weight history container
    cost_history = [g(w)] # cost history container
    d_vecs = []
    for k in range(max_its):
        # eval gradient
        grad_eval = gradient(w)
        # exponentially averaged descent direction
        if k == 0:
            d_vecs.append(-grad_eval)
        else:
            d_vecs.append(beta*d_vecs[-1] + (1-beta)*(-grad_eval))
        # descent step
        w = w + alpha*d_vecs[-1]
        # record weight and cost
        weight_history.append(w)
        cost_history.append(g(w))

    return np.array(weight_history), np.array(cost_history)
