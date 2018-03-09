#Copyright 2018 Brandon H. Lin. {brandon.hy.lin.0@gamil.com}
#
#Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
#
#http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
#
#-------------------------------------------------------------------------

########################################################
#
# This example is inspired from 
#   1. Centre for Speech Technology Research at U. of Edinburgh
#       https://github.com/CSTR-Edinburgh/merlin/blob/master/src/training_schemes/rprop.py
#   2. Louis Tiao's work
#       http://tiao.io/notes/visualizing-and-animating-optimization-algorithms-with-matplotlib/
########################################################


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
import matplotlib.animation as animation
from matplotlib import cm
from matplotlib.colors import LogNorm
import mpl_toolkits.mplot3d.axes3d as p3

IS_SAVE=False
#IS_SAVE=True

# parameter: type should be theano.shared
class SGD:
    # reg: regularization
    def __init__(self, cost_func, params, lr, reg, momentum=0):
        self.params = params
        self.lr = lr
        self.reg = reg
        self.momentum = momentum

        # define theano.shared parameters
        self.w = self.params
        size_w = self.w.get_value().shape

        vw_init = np.zeros(size_w)
        self.vw= theano.shared(vw_init, 'vw')

        # define cost function
        self.cost = cost_func(self.w, self.reg)

        # deinfe gradient method
        update_vw = momentum * self.vw - self.lr * T.grad(self.cost, self.w)
        update_w = self.w + update_vw

        self.train = theano.function(
            inputs=[],
            updates=[(self.w, update_w), (self.vw, update_vw)],
            outputs=[self.cost],
        )

    def step(self):
        self.train()
        return self.cost

    def get_weights(self):
        return self.w.get_value()

class NesterovSGD:
    def __init__(self, cost_func, params, lr, reg, momentum):
        self.params = params
        self.lr = lr
        self.reg = reg
        self.momentum = momentum

        # define theano.shared parameters
        self.w = self.params
        size_w = self.w.get_value().shape

        vw_init = np.zeros(size_w)
        self.vw = theano.shared(vw_init, 'vw')

        # define cost function
        self.cost = cost_func(self.w, self.reg)

        #define gradient method
        gw = T.grad(self.cost, self.w)
        update_vw = self.momentum * self.vw - lr * gw
        update_w = self.w + self.momentum * update_vw - lr * gw

        self.train = theano.function(
            inputs=[],
            updates=[(self.w, update_w), (self.vw, update_vw)],
            outputs=[self.cost],
        )

    def step(self):
        self.train()
        return self.cost

    def get_weights(self):
        return self.w.get_value()

class RMSprop:
    # Init paramters are refereced from pytorch
    # weight_decay: r = weight_decay * v + (1 - weight_decay) * g^2
    # alpha: served as momentum ( v = alpha * v - (lr * g )/[(r+eps)^0.5]
    def __init__(self, cost_func, params, lr, alpha, eps=1e-8, weight_decay=0,
        reg=0.004):
        
        self.params = params
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.reg = reg

        # define theano.shared parameters
        self.w = self.params
        size_w = self.w.get_value().shape

        vw_init = np.zeros(size_w)
        self.vw = theano.shared(vw_init, 'vw')

        rw_init = np.zeros(size_w)
        self.rw = theano.shared(rw_init, 'rw')

        # define cost function
        self.cost = cost_func(self.w, self.reg)

        gw = T.grad(self.cost, self.w)

        update_rw = self.weight_decay * self.rw + \
            (1 - self.weight_decay) *gw *gw

        update_vw = self.alpha * self.vw - \
            self.lr*gw / (np.sqrt(update_rw) + eps)

        update_w = self.w + update_vw

        self.train = theano.function(
            inputs=[],
            updates=[(self.w, update_w), (self.vw, update_vw),
                (self.rw, update_rw)],
            outputs=[self.cost],
        )

    def step(self):
        self.train()
        return self.cost

    def get_weights(self):
        return self.w.get_value()

class NesterovRMSprop:
    def __init__(self, cost_func, params, lr, alpha, eps=1e-8, 
        weight_decay=0, reg=0.004):
        
        self.params = params
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.reg = reg

        # define theano.shared parameters
        self.w = self.params
        size_w = self.w.get_value().shape

        vw_init = np.zeros(size_w)
        self.vw = theano.shared(vw_init, 'vw')

        rw_init = np.zeros(size_w)
        self.rw = theano.shared(rw_init, 'rw')

        # define cost function
        self.cost = cost_func(self.w, self.reg)

        gw = T.grad(self.cost, self.w)

        update_rw = self.weight_decay * self.rw + \
            ( 1 - self.weight_decay) * gw * gw

        update_vw = self.alpha * self.vw - \
            self.lr*gw / (np.sqrt(update_rw) + eps)
        
        update_w = self.w + self.alpha * update_vw - \
            self.lr * gw / (np.sqrt(update_rw) + self.eps)
            
        self.train = theano.function(
            inputs=[],
            updates=[(self.w, update_w), (self.vw, update_vw),
                (self.rw, update_rw)],
            outputs=[self.cost],
        )
        
    def step(self):
        self.train()
        return self.cost

    def get_weights(self):
        return self.w.get_value()


class Adam:
    # ADAptive Momentum
    def __init__(self, cost_func, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, reg=0):
        
        self.params = params
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.reg = reg

        self.weight_decay_L1 = self.betas[0]
        self.weight_decay_L2 = self.betas[1]

        # define theano.shared parameters
        t_init = 0
        self.t = theano.shared(t_init, 't')  # correction

        self.w = self.params
        size_w = self.w.get_value().shape

        sw_init = np.zeros(size_w)
        self.sw = theano.shared(sw_init, 'sw') # 1st moment (L1)

        rw_init = np.zeros(size_w)
        self.rw = theano.shared(rw_init, 'rw') # 2nd moment (L2)

        # define cost function
        self.cost = cost_func(self.w, self.reg)

        gw = T.grad(self.cost, self.w)

        update_t = self.t + 1

        update_sw = self.weight_decay_L1 * self.sw + \
            (1 - self.weight_decay_L1) * gw

        update_rw = self.weight_decay_L2 * self.rw + \
            (1 - self.weight_decay_L2) * gw * gw

        correction_s = ( 1 - self.weight_decay_L1 ** update_t)
        correction_r = ( 1 - self.weight_decay_L2 ** update_t)
        sw_hat = update_sw / correction_s
        rw_hat = update_rw /  correction_r

        update_w = self.w - self.lr * sw_hat / (np.sqrt(rw_hat) + self.eps)

        self.train = theano.function(
            inputs=[],
            updates=[(self.w, update_w), (self.sw, update_sw),
                (self.rw, update_rw), (self.t, update_t)],
            outputs=[self.cost],
        )

    def step(self):
        self.train()
        return self.cost

    def get_weights(self):
        return self.w.get_value()


# Resilient Propagation
# This Rprop class references to 
#   https://github.com/CSTR-Edinburgh/merlin/blob/master/src/training_schemes/rprop.py
class Rprop(object):
    def __init__(self, cost_func, params, initial_step_size, \
        eta_plus, eta_minus, min_step, max_step, reg):

        self.params = params
        self.initial_step_size = initial_step_size
        self.reg = reg
        self.w = self.params
        self.eta_plus = eta_plus
        self.eta_minus = eta_minus
        self.min_step = min_step
        self.max_step = max_step
        
        # value of previous gradients (type: numpy)
        self.v_prev_gradients = np.float32(
            np.zeros_like(self.params.get_value())
        )

        # define tensor
        prev_gradients = T.dvector('prev_gradients')

        # define theano.shared variables
        self.steps = theano.shared( \
            self.initial_step_size * \
                np.ones_like(self.params.get_value()) \
        )

        # define cost function
        self.cost = cost_func(self.w, self.reg)

        # calculate gradients
        gradients = T.grad(self.cost, self.w)
        grad_mul = gradients * prev_gradients
        
        # define updates
        tmp_steps = \
            self.eta_plus   * self.steps * T.gt(grad_mul, 0.0) + \
            self.eta_minus  * self.steps * T.lt(grad_mul, 0.0) + \
                              self.steps * T.eq(grad_mul, 0.0)

        # take ceil() and floor()
        update_steps = \
            T.minimum( self.max_step, \
                T.maximum(self.min_step, tmp_steps) \
        )

        update_w = self.w - T.sgn(gradients) * update_steps

        self.train = theano.function(
            inputs=[prev_gradients],
            updates=[
                (self.steps, update_steps),
                (self.w, update_w),
            ],
            outputs=[self.cost, gradients],
        )

    def step(self):
        tmp_cost, self.v_prev_gradients =  \
            self.train(self.v_prev_gradients)

        return self.cost

    def get_weights(self):
        return self.w.get_value()

class TrainProcess:
    def __init__(self, optimizer, max_iter, print_period=10):
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.print_period = print_period
        self.costs = []
        self.history = np.array([])

    def start(self):

        for i in xrange(self.max_iter):
            cost_val = self.optimizer.step() 
            
            if i % self.print_period == 0:
                w_value = self.optimizer.get_weights()
                cost_value = cost_func_without_reg(w_value[0], w_value[1])
                value = np.append(w_value, cost_value)

                if len(self.history) == 0:
                    self.history = value
                else:
                    self.history = np.vstack((self.history, value))

    def get_history(self):
        return self.history
        
# Saddle function
# ref: https://en.wikipedia.org/wiki/Saddle_point

def init_start_point():
    #return np.array([0.8, -0.000001])
    return np.array([3.5, 4.0])

def swapaxes(list_data):
    data = np.array(list_data)
    data = data.swapaxes(0, 1)
    return data

def z_saddle(x, y):
    return x**2 - y**2

def z_cone(x, y):
    return x**2 + y**2

def z_beale(x, y):
    return (1.5 - x + x*y)**2 + \
        (2.25 - x + x*(y**2))**2 + \
        (2.625 - x + x*(y**3))**2

def func_mesh(ax, fig):
#    x_max = 2
#    x_min = -2
#    y_max = 2
#    y_min = -2
    x_max = 4.5
    x_min = -4.5
    y_max = 4.5
    y_min = -4.5


    X = np.arange(x_min, x_max, 0.1)
    Y = np.arange(y_min, y_max, 0.1)
    X, Y = np.meshgrid(X, Y)
    Z = cost_func_without_reg(X, Y)

    surf = ax.plot_surface(X, Y, Z, cmap=cm.jet, alpha=0.8,
                                norm=LogNorm(), antialiased=False,
                                cstride=1, rstride=1, edgecolor='none')

    fig.colorbar(surf, shrink=0.5, aspect=5) 

#   create animation callback
#   create animation function with callback
def update_lines(num, dataLines, lines, points, labels, ax, func_mesh):
    #func_mesh(ax)

    for line, point, data, label in zip(lines, points, dataLines, labels):
        line.set_data(data[0:2, :num])
        line.set_3d_properties(data[2, :num])
        line.set_label(label)
        color = line.get_color()
        
        point.set_data(data[0:2, num-1:num])
        point.set_3d_properties(data[2, num-1:num])
        point.set_marker('o')
        point.set_color(color)

        ax.legend(loc='upper right')

    return lines + points

def cost_func_without_reg(X, Y):
    #return z_cone(X, Y)
    return z_beale(X, Y)

def cost_func(w, reg):
    return cost_func_without_reg(w[0], w[1]) + reg*((w*w).sum())

def main():
    max_iter = 4000
    print_period = 50
    w_init = init_start_point()
    M = 2 # size of w_init

### momentum 
    print "start training using Rprop" 

    initial_step_size = 0.001
    eta_plus = 1.2
    eta_minus = 0.5
    min_step = 1e-6
    max_step = 50
    reg = 0

    w = theano.shared(w_init, 'w')
    optimizer = Rprop(cost_func, w, initial_step_size=initial_step_size, 
        eta_plus=eta_plus, eta_minus=eta_minus, 
        min_step=min_step, max_step=max_step, reg=reg)

    training = TrainProcess(optimizer, max_iter, print_period)
    training.start()
    history_rprop = training.get_history()

### RMSprop w/ momentum
    print "start training RMSprop w/ momentum"

    lr = 0.006
    alpha = 0.9 # served as momentum in RMSprop
    weight_decay = 0.999

    w = theano.shared(w_init, 'w')
    optimizer = RMSprop(cost_func, w, lr, alpha, eps=1e-8, weight_decay=weight_decay, reg=reg) 

    training = TrainProcess(optimizer, max_iter, print_period)
    training.start()
    history_rms = training.get_history()

### momentum 
    print "start training with momentum" 

    #lr = 0.00004
    lr = 0.000001
    reg = 0
    momentum = 0.9

    w = theano.shared(w_init, 'w')
    optimizer = SGD(cost_func, w, lr, reg, momentum=momentum)
    training = TrainProcess(optimizer, max_iter, print_period)
    training.start()
    history_momentum = training.get_history()


    ##############################################
    # 3D animation
    ##############################################
    # {Main Body}
    #   create 3D axis
    #   [init animation]
    #       create lines upon #set of data
    #   Decorations:
    #       set axis boundary, axis label, legend
    #   create animation function with callback
    # {Callback}
    #   create animation callback
    #   Decorations:
    #       Add static image (cost function)
    #       Add legends
    #   pick color of lines
    #   plot lines and endpoints
    #   return lines + endpoints

    # prepare data
    data_rprop = swapaxes(history_rprop)
    data_rms = swapaxes(history_rms)
    data_momentum = swapaxes(history_momentum)

    data = np.array([
        data_rprop, data_rms, data_momentum 
    ])

    labels = ['Rprop', 'RMS', 'Momentum']

    num_lines, num_axes, num_frames = data.shape

    # create 3D axis
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    
    # [init animation] create lines upon #set of data
    lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in data]
    points = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in data]
 
    #   Decorations:
    #       set axis boundary, axis label, legend
    #ax.set_xlim(-2, 2)
    #ax.set_ylim(-2, 2)
    #ax.set_zlim(-4, 4)
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-4.5, 4.5)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(40, 325)
        
    #   create animation callbac
    line_ani = animation.FuncAnimation(fig, update_lines, 
        frames=num_frames, 
        fargs=(data, lines, points, labels, ax, func_mesh), 
        interval=50, blit=True, repeat=True)


    func_mesh(ax, fig)

    if IS_SAVE == True:
        line_ani.save('./rprop_animation_3d.gif', writer='imagemagick',fps=1000/100)

    else:
        line_ani
        plt.show()

if __name__ == '__main__':
    main()
