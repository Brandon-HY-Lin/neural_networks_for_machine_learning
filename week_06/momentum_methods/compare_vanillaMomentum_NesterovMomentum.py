import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from util import get_normalized_data, y2indicator

# get visible units thX and verified data thT
# get output units thY
#   define H_hidden = thX * W1 + b1
#   define H_output = H_hidden * W2 + b2
# define probability thY = T.nn.softmax(H_output)
# define gradient descent function
# define theano.function of gradient descent
# define negative log likelihood function and precision function
#   l2_reg = sum(W1*W1) + sum(b1 * b1) + sum(W2 * W2) + sum(b2 * b2)
#   cost = sum(-1 * thT * log thY) + lr * l2_reg
#   prediction = mean(thT == argmax(thY, axis = 1)
# define theano.function of cost and prediction
# 
# get data
# perform mini-batch training
#   train data of "training set"
#       append negative log likelihood result of "testing set"
#       print precision and negative log likelihood of "testing set"
# plot traning history of negative log likelihood result

def error_rate(p, t):
    return np.mean(p != t)

def relu(a):
    return a * (a > 0)

def main():
    X, Y = get_normalized_data()
    max_iter = 2
    print_period = 10

    lr = 0.00004
    reg = 0.01

    Xtrain = X[:-1000,]
    Ytrain = Y[:-1000]
    Xtest = X[-1000:,]
    Ytest = Y[-1000:]

    Ytrain_ind = y2indicator(Ytrain)
    Ytest_ind = y2indicator(Ytest)

    N, D = Xtrain.shape
    batch_sz = 500
    n_batches = N / batch_sz

    M = 300
    K = 10

############################################
    # momentum
    W1_init = np.random.randn(D, M) / 28
    b1_init = np.zeros(M)
    W2_init = np.random.randn(M,K) / np.sqrt(M)
    b2_init = np.zeros(K)

    vW1_init = np.zeros((D, M))
    vb1_init = np.zeros(M)
    vW2_init = np.zeros((M,K))
    vb2_init = np.zeros(K)


    thX = T.matrix('X')
    thT = T.matrix('T')
    W1 = theano.shared(W1_init, 'W1')
    b1 = theano.shared(b1_init, 'b1')
    W2 = theano.shared(W2_init, 'W2')
    b2 = theano.shared(b2_init, 'b2')
 
    vW1 = theano.shared(vW1_init, 'vW1')
    vb1 = theano.shared(vb1_init, 'vb1')
    vW2 = theano.shared(vW2_init, 'vW2')
    vb2 = theano.shared(vb2_init, 'vb2')
    
    thZ = relu(thX.dot(W1) + b1)
    thY = T.nnet.softmax(thZ.dot(W2)+b2)
    cost = - (thT * T.log(thY)).sum() + reg*((W1*W1).sum() + (b1*b1).sum() + (W2*W2).sum() + (b2*b2).sum())

    prediction = T.argmax(thY, axis=1)

    alpha = 0.9

    update_vW1 = alpha * vW1 - lr*T.grad(cost, W1)
    update_vb1 = alpha * vb1 - lr*T.grad(cost, b1)
    update_vW2 = alpha * vW2 - lr*T.grad(cost, W2)
    update_vb2 = alpha * vb2 - lr*T.grad(cost, b2)

    update_W1 = W1 + update_vW1
    update_b1 = b1 + update_vb1
    update_W2 = W2 + update_vW2
    update_b2 = b2 + update_vb2 

    train = theano.function(
        inputs=[thX, thT], 
        updates=[(W1, update_W1), (b1, update_b1), (W2, update_W2), (b2, update_b2)] + 
        [(vW1, update_vW1), (vb1, update_vb1), (vW2, update_vW2), (vb2, update_vb2)],
    )

    get_prediction = theano.function(
        inputs=[thX, thT], 
        outputs=[cost, prediction],
    )

    print "Starting training using vanilla momentum"
    costs_momentum = []
    for i in xrange(max_iter):
        for j in xrange(n_batches):
            Xbatch = Xtrain[j*batch_sz: (j*batch_sz + batch_sz),]
            Ybatch = Ytrain_ind[j*batch_sz: (j*batch_sz + batch_sz),]
            train(Xbatch, Ybatch)

            if j % print_period == 0:
                cost_val, prediction_val = get_prediction(Xtest, Ytest_ind)
                err = error_rate(prediction_val, Ytest)
                print "Cost / err at iteration i=%d, j=%d: %.3f / %.3f" % (i, j , cost_val, err)
                costs_momentum.append(cost_val)

############################################
    # Nesterov momentum
    W1_init = np.random.randn(D, M) / 28
    b1_init = np.zeros(M)
    W2_init = np.random.randn(M,K) / np.sqrt(M)
    b2_init = np.zeros(K)

    vW1_init = np.zeros((D, M))
    vb1_init = np.zeros(M)
    vW2_init = np.zeros((M,K))
    vb2_init = np.zeros(K)


    thX = T.matrix('X')
    thT = T.matrix('T')
    W1 = theano.shared(W1_init, 'W1')
    b1 = theano.shared(b1_init, 'b1')
    W2 = theano.shared(W2_init, 'W2')
    b2 = theano.shared(b2_init, 'b2')
 
    vW1 = theano.shared(vW1_init, 'vW1')
    vb1 = theano.shared(vb1_init, 'vb1')
    vW2 = theano.shared(vW2_init, 'vW2')
    vb2 = theano.shared(vb2_init, 'vb2')
    
    thZ = relu(thX.dot(W1) + b1)
    thY = T.nnet.softmax(thZ.dot(W2)+b2)
    cost = - (thT * T.log(thY)).sum() + reg*((W1*W1).sum() + (b1*b1).sum() + (W2*W2).sum() + (b2*b2).sum())

    prediction = T.argmax(thY, axis=1)

    alpha = 0.9

    gW1 = T.grad(cost, W1)
    gb1 = T.grad(cost, b1)
    gW2 = T.grad(cost, W2)
    gb2 = T.grad(cost, b2)

    update_vW1 = alpha * vW1 - lr*gW1
    update_vb1 = alpha * vb1 - lr*gb1
    update_vW2 = alpha * vW2 - lr*gW2
    update_vb2 = alpha * vb2 - lr*gb2

    update_W1 = W1 + alpha * update_vW1 - lr*gW1
    update_b1 = b1 + alpha * update_vb1 - lr*gb1
    update_W2 = W2 + alpha * update_vW2 - lr*gW2
    update_b2 = b2 + alpha * update_vb2 - lr*gb2

    train = theano.function(
        inputs=[thX, thT], 
        updates=[(W1, update_W1), (b1, update_b1), (W2, update_W2), (b2, update_b2)] + 
        [(vW1, update_vW1), (vb1, update_vb1), (vW2, update_vW2), (vb2, update_vb2)],
    )

    get_prediction = theano.function(
        inputs=[thX, thT], 
        outputs=[cost, prediction],
    )

    print "Starting training using Nesterov Momentum"
    costs_nesterov = []
    for i in xrange(max_iter):
        for j in xrange(n_batches):
            Xbatch = Xtrain[j*batch_sz: (j*batch_sz + batch_sz),]
            Ybatch = Ytrain_ind[j*batch_sz: (j*batch_sz + batch_sz),]
            train(Xbatch, Ybatch)

            if j % print_period == 0:
                cost_val, prediction_val = get_prediction(Xtest, Ytest_ind)
                err = error_rate(prediction_val, Ytest)
                print "Cost / err at iteration i=%d, j=%d: %.3f / %.3f" % (i, j , cost_val, err)
                costs_nesterov.append(cost_val)

    plt.plot(costs_momentum, label='Vanilla Momentum')
    plt.plot(costs_nesterov, label='Nesterov Momentum')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
