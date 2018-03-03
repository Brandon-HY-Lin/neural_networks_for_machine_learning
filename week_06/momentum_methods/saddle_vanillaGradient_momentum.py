#Copyright 2018 Brandon H. Lin. {brandon.hy.lin.0@gamil.com}
#
#Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
#
#http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
#
#-------------------------------------------------------------------------

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

# Saddle function
# ref: https://en.wikipedia.org/wiki/Saddle_point

def init_start_point():
    return np.array([0.0, -1.0])

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
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.8,
                                linewidth=0.05, antialiased=False,
                                cstride=1, rstride=1,
                                edgecolor='dimgrey')
    #ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2, color='k')


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

        #func_mesh(ax)
        ax.legend(loc='upper right')


    return lines + points

def main():
    max_iter = 100
    print_period = 10
    w_init = init_start_point()
    M = 2 # size of w_init

### vanilla gradient descent
    print "start training vanilla gradient descent"

    #lr = 0.00004
    lr = 0.004
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
    history_w = np.array([])
    for i in xrange(max_iter):
        cost_val = train()

        if i % print_period == 0:
            costs.append(cost_val)
            w_value = w.get_value()
            cost_value = z_saddle(w_value[0], w_value[1])
            value = np.append(w_value, cost_value)
            #history_w.append(np.append(w_value, z_saddle(w_value[0], w_value[1])))
            if len(history_w) == 0:
                history_w = value
            else:
                history_w = np.vstack((history_w, value))
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
    history_w_momentum = np.array([])
    for i in xrange(max_iter):
        cost_val = train()

        if i % print_period == 0:
            costs_momentum.append(cost_val)
            w_value = w.get_value()
            cost_value = z_saddle(w_value[0], w_value[1])
            value = np.append(w_value, cost_value)
            #history_w_momentum.append(np.append(w_value, z_saddle(w_value[0], w_value[1])))
            if len(history_w_momentum) == 0:
                history_w_momentum = value
            else:
                history_w_momentum = np.vstack((history_w_momentum, value))

            print "Cost at iteration i=%d: %.3f" % (i, cost_val[0]) 

    #plt.plot(costs, label='Vanilla Gradient Descent')
    #plt.plot(costs_momentum, label='Vanilla Momentum')
    #plt.legend()
    #plt.show()

   
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
    data = np.array([data_gradient, data_momentum])
    labels = ['gradient', 'vanilla momentum']
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

    func_mesh(ax)
    line_ani
    plt.show()
    #line_ani.save('./saddle_animation_3d.gif', writer='imagemagick',fps=1000/100)

if __name__ == '__main__':
    main()
