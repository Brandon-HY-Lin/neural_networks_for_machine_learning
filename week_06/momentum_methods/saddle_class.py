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
import mpl_toolkits.mplot3d.axes3d as p3

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
                cost_value = z_saddle(w_value[0], w_value[1])
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
    return np.array([0.8, -0.000001])

def swapaxes(list_data):
    data = np.array(list_data)
    data = data.swapaxes(0, 1)
    return data

def z_saddle(x, y):
    return x**2 - y**2

def func_mesh(ax):
    x_max = 2
    x_min = -2
    y_max = 2
    y_min = -2

    X = np.arange(x_min, x_max, 0.1)
    Y = np.arange(y_min, y_max, 0.1)
    X, Y = np.meshgrid(X, Y)
    Z = z_saddle(X, Y)
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                                linewidth=10, antialiased=False)


#   create animation callback
#   create animation function with callback
def update_lines(num, dataLines, lines, points, labels, ax, func_mesh):
    func_mesh(ax)

    for line, point, data, label in zip(lines, points, dataLines, labels):
        line.set_data(data[0:2, :num])
        line.set_3d_properties(data[2, :num])
        line.set_label(label)
        color = line.get_color()
        
        point.set_data(data[0:2, num-1:num])
        point.set_3d_properties(data[2, num-1:num])
        point.set_marker('o')
        point.set_color(color)

        ax.legend()

    return lines + points

def cost_func(w, reg):
    return w[0]**2 - w[1]**2 + reg*((w*w).sum())

def main():
    max_iter = 200
    print_period = 5
    w_init = init_start_point()
    M = 2 # size of w_init

### vanilla gradient descent
    print "start training vanilla gradient descent"

    #lr = 0.00004
    lr = 0.004
    reg = 0

    w = theano.shared(w_init, 'w')
    optimizer = SGD(cost_func, w, lr, reg, momentum=0)
    training = TrainProcess(optimizer, max_iter, print_period)
    training.start()
    history_w = training.get_history()

### vanilla momentum
    print "start training vanilla momentum"

    lr = 0.004
    momentum = 0.99

    w = theano.shared(w_init, 'w')
    optimizer = SGD(cost_func, w, lr, reg, momentum=momentum) 
    training = TrainProcess(optimizer, max_iter, print_period)
    training.start()
    history_w_momentum = training.get_history()
 
 ### Nesterov momentum
    print "start training Nesterov momentum"

    lr = 0.004
    momentum = 0.99

    w = theano.shared(w_init, 'w')
    optimizer = NesterovSGD(cost_func, w, lr, reg, momentum=momentum) 
    training = TrainProcess(optimizer, max_iter, print_period)
    training.start()
    history_nesterov_momentum = training.get_history()

 ### RMSprop
    print "start training vanilla RMSprop w/o momentum"

    lr = 0.0004
    alpha = 0 # served as momentum in RMSprop
    weight_decay = 0.999

    w = theano.shared(w_init, 'w')
    optimizer = RMSprop(cost_func, w, lr, alpha, eps=1e-8, weight_decay=weight_decay, reg=reg) 

    training = TrainProcess(optimizer, max_iter, print_period)
    training.start()
    history_vanilla_rms = training.get_history()


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
    data_gradient = swapaxes(history_w)
    data_momentum = swapaxes(history_w_momentum)
    data_nesterov_momentum = swapaxes(history_nesterov_momentum)
    data_vanilla_rms = swapaxes(history_vanilla_rms)

    data = np.array([data_gradient, data_momentum, 
        data_nesterov_momentum, data_vanilla_rms])

    labels = ['gradient', 'vanilla momentum', 'Nesterov momentum',
        'vanilla RMSprop w/o momentum']

    num_lines, num_axes, num_frames = data.shape

    # create 3D axis
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    
    # [init animation] create lines upon #set of data
    lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in data]
    points = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in data]
 
    #   Decorations:
    #       set axis boundary, axis label, legend
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-4, 4)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(30, 220)
        
    #   create animation callbac
    line_ani = animation.FuncAnimation(fig, update_lines, 
        frames=num_frames, 
        fargs=(data, lines, points, labels, ax, func_mesh), 
        interval=100, blit=True, repeat=True)

    line_ani
    plt.show()

if __name__ == '__main__':
    main()