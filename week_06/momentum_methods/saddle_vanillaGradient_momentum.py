# define cost function of saddle function (z = x^2 - y^2)
# set initial point of (x, y)
# define the families of gradient descent methods
# compute the gradient and cost in each step
#   record cost function
#
# plot the cost values along the optimization process
#
# Note: This program is a reproduced copy of CS231n
# http://cs231n.github.io/neural-networks-3/

import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

# Saddle function
# ref: https://en.wikipedia.org/wiki/Saddle_point

def init_start_point():
    return np.array([0.0, -1.0])

def main():
    max_iter = 100
    print_period = 10
    w_init = init_start_point()
    M = 2 # size of w_init

### vanilla gradient descent
    print "start training vanilla gradient descent"

    lr = 0.00004
    reg = 0.01

    w = theano.shared(w_init, 'w')
    cost = w[0]**2 - w[1]**2 + reg*((w*w).sum())

    update_w = w - lr*T.grad(cost, w)

    train = theano.function(
        inputs=[],
        updates=[(w, update_w)],
        outputs=[cost],
    )

    costs = [] 
    history_w = []
    for i in xrange(max_iter):
        cost_val = train()

        if i % print_period == 0:
            costs.append(cost_val)
            history_w.append(w.get_value())
            print "Cost at iteration i=%d: %.3f" % (i, cost_val[0]) 

### vanilla momentum
    print "start training vanilla momentum"

    lr = 0.00004
    reg = 0.01
    alpha = 0.9

    vw_init = np.zeros(M)

    w = theano.shared(w_init, 'w')
    vw = theano.shared(vw_init, 'vw')

    cost = w[0]**2 - w[1]**2 + reg*((w*w).sum())

    update_vw = alpha*vw - lr*T.grad(cost, w)
    update_w = w + update_vw

    train = theano.function(
        inputs=[],
        updates=[(w, update_w), (vw, update_vw)],
        outputs=[cost],
    )

    costs_momentum = [] 
    history_w_momentum = []
    for i in xrange(max_iter):
        cost_val = train()

        if i % print_period == 0:
            costs_momentum.append(cost_val)
            history_w_momentum.append(w.get_value())
            print "Cost at iteration i=%d: %.3f" % (i, cost_val[0]) 


    plt.plot(costs, label='Vanilla Gradient Descent')
    plt.plot(costs_momentum, label='Vanilla Momentum')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
