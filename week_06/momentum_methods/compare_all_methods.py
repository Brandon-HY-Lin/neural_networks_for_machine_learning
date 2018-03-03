#Copyright 2018 Brandon H. Lin. {brandon.hy.lin.0@gamil.com}
#
#Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
#
#http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
#
#-------------------------------------------------------------------------

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

    # vanilla gradient decent
    W1_init = np.random.randn(D, M) / 28
    b1_init = np.zeros(M)
    W2_init = np.random.randn(M,K) / np.sqrt(M)
    b2_init = np.zeros(K)

    thX = T.matrix('X')
    thT = T.matrix('T')
    W1 = theano.shared(W1_init, 'W1')
    b1 = theano.shared(b1_init, 'b1')
    W2 = theano.shared(W2_init, 'W2')
    b2 = theano.shared(b2_init, 'b2')
    
    thZ = relu(thX.dot(W1) + b1)
    thY = T.nnet.softmax(thZ.dot(W2)+b2)
    cost = - (thT * T.log(thY)).sum() + reg*((W1*W1).sum() + (b1*b1).sum() + (W2*W2).sum() + (b2*b2).sum())

    prediction = T.argmax(thY, axis=1)

    update_W1 = W1 - lr*T.grad(cost, W1)
    update_b1 = b1 - lr*T.grad(cost, b1)
    update_W2 = W2 - lr*T.grad(cost, W2)
    update_b2 = b2 - lr*T.grad(cost, b2)

    train = theano.function(
        inputs=[thX, thT], 
        updates=[(W1, update_W1), (b1, update_b1), (W2, update_W2), (b2, update_b2)],
    )

    get_prediction = theano.function(
        inputs=[thX, thT], 
        outputs=[cost, prediction],
    )

    print "Starting training using vanilla Gradient Descent"
    costs = []
    for i in xrange(max_iter):
        for j in xrange(n_batches):
            Xbatch = Xtrain[j*batch_sz: (j*batch_sz + batch_sz),]
            Ybatch = Ytrain_ind[j*batch_sz: (j*batch_sz + batch_sz),]
            train(Xbatch, Ybatch)

            if j % print_period == 0:
                cost_val, prediction_val = get_prediction(Xtest, Ytest_ind)
                err = error_rate(prediction_val, Ytest)
                print "Cost / err at iteration i=%d, j=%d: %.3f / %.3f" % (i, j , cost_val, err)
                costs.append(cost_val)

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

############################################
    # RMSprop with vanilla Gradient 
    W1_init = np.random.randn(D, M) / 28
    b1_init = np.zeros(M)
    W2_init = np.random.randn(M,K) / np.sqrt(M)
    b2_init = np.zeros(K)

    rW1_init = np.zeros((D, M))
    rb1_init = np.zeros(M)
    rW2_init = np.zeros((M,K))
    rb2_init = np.zeros(K)


    thX = T.matrix('X')
    thT = T.matrix('T')
    W1 = theano.shared(W1_init, 'W1')
    b1 = theano.shared(b1_init, 'b1')
    W2 = theano.shared(W2_init, 'W2')
    b2 = theano.shared(b2_init, 'b2')
 
    rW1 = theano.shared(rW1_init, 'rW1')
    rb1 = theano.shared(rb1_init, 'rb1')
    rW2 = theano.shared(rW2_init, 'rW2')
    rb2 = theano.shared(rb2_init, 'rb2')
    
    thZ = relu(thX.dot(W1) + b1)
    thY = T.nnet.softmax(thZ.dot(W2)+b2)
    cost = - (thT * T.log(thY)).sum() + reg*((W1*W1).sum() + (b1*b1).sum() + (W2*W2).sum() + (b2*b2).sum())

    prediction = T.argmax(thY, axis=1)

    decay_rate = 0.999
    eps = 0.0000000001

    gW1 = T.grad(cost, W1)
    gb1 = T.grad(cost, b1)
    gW2 = T.grad(cost, W2)
    gb2 = T.grad(cost, b2)

    update_rW1 = decay_rate * rW1 + (1 - decay_rate)*gW1*gW1
    update_rb1 = decay_rate * rb1 + (1 - decay_rate)*gb1*gb1
    update_rW2 = decay_rate * rW2 + (1 - decay_rate)*gW2*gW2
    update_rb2 = decay_rate * rb2 + (1 - decay_rate)*gb2*gb2

    update_W1 = W1 - lr*gW1 / (np.sqrt(update_rW1) + eps)
    update_b1 = b1 - lr*gb1 / (np.sqrt(update_rb1) + eps)
    update_W2 = W2 - lr*gW2 / (np.sqrt(update_rW2) + eps)
    update_b2 = b2 - lr*gb2 / (np.sqrt(update_rb2) + eps)

    train = theano.function(
        inputs=[thX, thT], 
        updates=[(W1, update_W1), (b1, update_b1), (W2, update_W2), (b2, update_b2)] + 
        [(rW1, update_rW1), (rb1, update_rb1), (rW2, update_rW2), (rb2, update_rb2)],
    )

    get_prediction = theano.function(
        inputs=[thX, thT], 
        outputs=[cost, prediction],
    )

    print "Starting training using RMSprop"
    costs_rmsprop = []
    for i in xrange(max_iter):
        for j in xrange(n_batches):
            Xbatch = Xtrain[j*batch_sz: (j*batch_sz + batch_sz),]
            Ybatch = Ytrain_ind[j*batch_sz: (j*batch_sz + batch_sz),]
            train(Xbatch, Ybatch)

            if j % print_period == 0:
                cost_val, prediction_val = get_prediction(Xtest, Ytest_ind)
                err = error_rate(prediction_val, Ytest)
                print "Cost / err at iteration i=%d, j=%d: %.3f / %.3f" % (i, j , cost_val, err)
                costs_rmsprop.append(cost_val)

############################################
    # RMSprop with vanilla momentum
    W1_init = np.random.randn(D, M) / 28
    b1_init = np.zeros(M)
    W2_init = np.random.randn(M,K) / np.sqrt(M)
    b2_init = np.zeros(K)

    vW1_init = np.zeros((D, M))
    vb1_init = np.zeros(M)
    vW2_init = np.zeros((M,K))
    vb2_init = np.zeros(K)

    rW1_init = np.zeros((D, M))
    rb1_init = np.zeros(M)
    rW2_init = np.zeros((M,K))
    rb2_init = np.zeros(K)

    thX = T.matrix('X')
    thT = T.matrix('T')
    W1 = theano.shared(W1_init, 'W1')
    b1 = theano.shared(b1_init, 'b1')
    W2 = theano.shared(W2_init, 'W2')
    b2 = theano.shared(b2_init, 'b2')
 
    vW1 = theano.shared(vW1_init, 'rW1')
    vb1 = theano.shared(vb1_init, 'rb1')
    vW2 = theano.shared(vW2_init, 'rW2')
    vb2 = theano.shared(vb2_init, 'rb2')
 
    rW1 = theano.shared(rW1_init, 'rW1')
    rb1 = theano.shared(rb1_init, 'rb1')
    rW2 = theano.shared(rW2_init, 'rW2')
    rb2 = theano.shared(rb2_init, 'rb2')
 
    thZ = relu(thX.dot(W1) + b1)
    thY = T.nnet.softmax(thZ.dot(W2)+b2)
    cost = - (thT * T.log(thY)).sum() + reg*((W1*W1).sum() + (b1*b1).sum() + (W2*W2).sum() + (b2*b2).sum())

    prediction = T.argmax(thY, axis=1)

    decay_rate = 0.999
    eps = 0.0000000001
    alpha = 0.9 # momentum

    gW1 = T.grad(cost, W1)
    gb1 = T.grad(cost, b1)
    gW2 = T.grad(cost, W2)
    gb2 = T.grad(cost, b2)

    update_rW1 = decay_rate * rW1 + (1 - decay_rate)*gW1*gW1
    update_rb1 = decay_rate * rb1 + (1 - decay_rate)*gb1*gb1
    update_rW2 = decay_rate * rW2 + (1 - decay_rate)*gW2*gW2
    update_rb2 = decay_rate * rb2 + (1 - decay_rate)*gb2*gb2

    update_vW1 = alpha * vW1 - lr*gW1 / (np.sqrt(update_rW1) + eps)
    update_vb1 = alpha * vb1 - lr*gb1 / (np.sqrt(update_rb1) + eps)
    update_vW2 = alpha * vW2 - lr*gW2 / (np.sqrt(update_rW2) + eps)
    update_vb2 = alpha * vb2 - lr*gb2 / (np.sqrt(update_rb2) + eps)

    update_W1 = W1 + update_vW1 
    update_b1 = b1 + update_vb1 
    update_W2 = W2 + update_vW2 
    update_b2 = b2 + update_vb2 

    train = theano.function(
        inputs=[thX, thT], 
        updates=[(W1, update_W1), (b1, update_b1), (W2, update_W2), (b2, update_b2)] + 
        [(vW1, update_vW1), (vb1, update_vb1), (vW2, update_vW2), (vb2, update_vb2)] +
        [(rW1, update_rW1), (rb1, update_rb1), (rW2, update_rW2), (rb2, update_rb2)],
    )

    get_prediction = theano.function(
        inputs=[thX, thT], 
        outputs=[cost, prediction],
    )

    print "Starting training using RMSprop with momentum"
    costs_rmsprop_momentum = []
    for i in xrange(max_iter):
        for j in xrange(n_batches):
            Xbatch = Xtrain[j*batch_sz: (j*batch_sz + batch_sz),]
            Ybatch = Ytrain_ind[j*batch_sz: (j*batch_sz + batch_sz),]
            train(Xbatch, Ybatch)

            if j % print_period == 0:
                cost_val, prediction_val = get_prediction(Xtest, Ytest_ind)
                err = error_rate(prediction_val, Ytest)
                print "Cost / err at iteration i=%d, j=%d: %.3f / %.3f" % (i, j , cost_val, err)
                costs_rmsprop_momentum.append(cost_val)



############################################
    # RMSprop with nesterov momentum
    W1_init = np.random.randn(D, M) / 28
    b1_init = np.zeros(M)
    W2_init = np.random.randn(M,K) / np.sqrt(M)
    b2_init = np.zeros(K)

    vW1_init = np.zeros((D, M))
    vb1_init = np.zeros(M)
    vW2_init = np.zeros((M,K))
    vb2_init = np.zeros(K)

    rW1_init = np.zeros((D, M))
    rb1_init = np.zeros(M)
    rW2_init = np.zeros((M,K))
    rb2_init = np.zeros(K)

    thX = T.matrix('X')
    thT = T.matrix('T')
    W1 = theano.shared(W1_init, 'W1')
    b1 = theano.shared(b1_init, 'b1')
    W2 = theano.shared(W2_init, 'W2')
    b2 = theano.shared(b2_init, 'b2')
 
    vW1 = theano.shared(vW1_init, 'rW1')
    vb1 = theano.shared(vb1_init, 'rb1')
    vW2 = theano.shared(vW2_init, 'rW2')
    vb2 = theano.shared(vb2_init, 'rb2')
 
    rW1 = theano.shared(rW1_init, 'rW1')
    rb1 = theano.shared(rb1_init, 'rb1')
    rW2 = theano.shared(rW2_init, 'rW2')
    rb2 = theano.shared(rb2_init, 'rb2')
 
    thZ = relu(thX.dot(W1) + b1)
    thY = T.nnet.softmax(thZ.dot(W2)+b2)
    cost = - (thT * T.log(thY)).sum() + reg*((W1*W1).sum() + (b1*b1).sum() + (W2*W2).sum() + (b2*b2).sum())

    prediction = T.argmax(thY, axis=1)

    decay_rate = 0.999
    eps = 0.0000000001
    alpha = 0.9 # momentum

    gW1 = T.grad(cost, W1)
    gb1 = T.grad(cost, b1)
    gW2 = T.grad(cost, W2)
    gb2 = T.grad(cost, b2)

    update_rW1 = decay_rate * rW1 + (1 - decay_rate)*gW1*gW1
    update_rb1 = decay_rate * rb1 + (1 - decay_rate)*gb1*gb1
    update_rW2 = decay_rate * rW2 + (1 - decay_rate)*gW2*gW2
    update_rb2 = decay_rate * rb2 + (1 - decay_rate)*gb2*gb2

    update_vW1 = alpha * vW1 - lr*gW1 / (np.sqrt(update_rW1) + eps)
    update_vb1 = alpha * vb1 - lr*gb1 / (np.sqrt(update_rb1) + eps)
    update_vW2 = alpha * vW2 - lr*gW2 / (np.sqrt(update_rW2) + eps)
    update_vb2 = alpha * vb2 - lr*gb2 / (np.sqrt(update_rb2) + eps)

    update_W1 = W1 + alpha * update_vW1 - lr*gW1 / (np.sqrt(update_rW1) + eps)
    update_b1 = b1 + alpha * update_vb1 - lr*gb1 / (np.sqrt(update_rb1) + eps)
    update_W2 = W2 + alpha * update_vW2 - lr*gW2 / (np.sqrt(update_rW2) + eps)
    update_b2 = b2 + alpha * update_vb2 - lr*gb2 / (np.sqrt(update_rb2) + eps)

    train = theano.function(
        inputs=[thX, thT], 
        updates=[(W1, update_W1), (b1, update_b1), (W2, update_W2), (b2, update_b2)] + 
        [(vW1, update_vW1), (vb1, update_vb1), (vW2, update_vW2), (vb2, update_vb2)] +
        [(rW1, update_rW1), (rb1, update_rb1), (rW2, update_rW2), (rb2, update_rb2)],
    )

    get_prediction = theano.function(
        inputs=[thX, thT], 
        outputs=[cost, prediction],
    )

    print "Starting training using RMSprop with Nesterov Momentum"
    costs_rmsprop_nesterov = []
    for i in xrange(max_iter):
        for j in xrange(n_batches):
            Xbatch = Xtrain[j*batch_sz: (j*batch_sz + batch_sz),]
            Ybatch = Ytrain_ind[j*batch_sz: (j*batch_sz + batch_sz),]
            train(Xbatch, Ybatch)

            if j % print_period == 0:
                cost_val, prediction_val = get_prediction(Xtest, Ytest_ind)
                err = error_rate(prediction_val, Ytest)
                print "Cost / err at iteration i=%d, j=%d: %.3f / %.3f" % (i, j , cost_val, err)
                costs_rmsprop_nesterov.append(cost_val)

###########################################
    # Adam (ADAptive Moments) 
    W1_init = np.random.randn(D, M) / 28
    b1_init = np.zeros(M)
    W2_init = np.random.randn(M,K) / np.sqrt(M)
    b2_init = np.zeros(K)

    sW1_init = np.zeros((D, M))
    sb1_init = np.zeros(M)
    sW2_init = np.zeros((M,K))
    sb2_init = np.zeros(K)

    rW1_init = np.zeros((D, M))
    rb1_init = np.zeros(M)
    rW2_init = np.zeros((M,K))
    rb2_init = np.zeros(K)

    t_init = 0

    thX = T.matrix('X')
    thT = T.matrix('T')
    W1 = theano.shared(W1_init, 'W1')
    b1 = theano.shared(b1_init, 'b1')
    W2 = theano.shared(W2_init, 'W2')
    b2 = theano.shared(b2_init, 'b2')
 
    sW1 = theano.shared(sW1_init, 'sW1')
    sb1 = theano.shared(sb1_init, 'sb1')
    sW2 = theano.shared(sW2_init, 'sW2')
    sb2 = theano.shared(sb2_init, 'sb2')
 
    rW1 = theano.shared(rW1_init, 'rW1')
    rb1 = theano.shared(rb1_init, 'rb1')
    rW2 = theano.shared(rW2_init, 'rW2')
    rb2 = theano.shared(rb2_init, 'rb2')

    t = theano.shared(t_init, 't')
 
    thZ = relu(thX.dot(W1) + b1)
    thY = T.nnet.softmax(thZ.dot(W2)+b2)
    cost = - (thT * T.log(thY)).sum() + reg*((W1*W1).sum() + (b1*b1).sum() + (W2*W2).sum() + (b2*b2).sum())

    prediction = T.argmax(thY, axis=1)

    lr = 0.001  #increase learning rate by order 2
    decay_rate_s = 0.9
    decay_rate_r = 0.999
    eps = 0.0000000001

    gW1 = T.grad(cost, W1)
    gb1 = T.grad(cost, b1)
    gW2 = T.grad(cost, W2)
    gb2 = T.grad(cost, b2)

    update_t = t + 1

    update_sW1 = decay_rate_s * sW1 + (1 - decay_rate_s)*gW1
    update_sb1 = decay_rate_s * sb1 + (1 - decay_rate_s)*gb1
    update_sW2 = decay_rate_s * sW2 + (1 - decay_rate_s)*gW2
    update_sb2 = decay_rate_s * sb2 + (1 - decay_rate_s)*gb2

    correction_s = ( 1 - decay_rate_s ** update_t)

    sW1_hat = update_sW1 / correction_s 
    sb1_hat = update_sb1 / correction_s
    sW2_hat = update_sW2 / correction_s
    sb2_hat = update_sb2 / correction_s

    update_rW1 = decay_rate_r * rW1 + (1 - decay_rate_r)*gW1*gW1
    update_rb1 = decay_rate_r * rb1 + (1 - decay_rate_r)*gb1*gb1
    update_rW2 = decay_rate_r * rW2 + (1 - decay_rate_r)*gW2*gW2
    update_rb2 = decay_rate_r * rb2 + (1 - decay_rate_r)*gb2*gb2

    correction_r = ( 1 - decay_rate_r ** update_t)

    rW1_hat = update_rW1 / correction_r 
    rb1_hat = update_rb1 / correction_r
    rW2_hat = update_rW2 / correction_r
    rb2_hat = update_rb2 / correction_r

#    dW1 = - lr*sW1_hat / (np.sqrt(rW1_hat) + eps)
#    db1 = - lr*sb1_hat / (np.sqrt(rb1_hat) + eps)
#    dW2 = - lr*sW2_hat / (np.sqrt(rW2_hat) + eps)
#    db2 = - lr*sb2_hat / (np.sqrt(rb2_hat) + eps)
#
#    update_W1 = W1 + dW1
#    update_b1 = b1 + db1
#    update_W2 = W2 + dW2
#    update_b2 = b2 + db2

    update_W1 = W1 - lr*sW1_hat / (np.sqrt(rW1_hat) + eps)
    update_b1 = b1 - lr*sb1_hat / (np.sqrt(rb1_hat) + eps)
    update_W2 = W2 - lr*sW2_hat / (np.sqrt(rW2_hat) + eps)
    update_b2 = b2 - lr*sb2_hat / (np.sqrt(rb2_hat) + eps)



    train = theano.function(
        inputs=[thX, thT], 
        updates=[(W1, update_W1), (b1, update_b1), (W2, update_W2), (b2, update_b2)] + 
        [(sW1, update_sW1), (sb1, update_sb1), (sW2, update_sW2), (sb2, update_sb2)] + 
        [(rW1, update_rW1), (rb1, update_rb1), (rW2, update_rW2), (rb2, update_rb2)] + 
        [(t, update_t)],
    )

    get_prediction = theano.function(
        inputs=[thX, thT], 
        outputs=[cost, prediction],
    )

    print "Starting training using ADAM"
    costs_adam = []
    for i in xrange(max_iter):
        for j in xrange(n_batches):
            Xbatch = Xtrain[j*batch_sz: (j*batch_sz + batch_sz),]
            Ybatch = Ytrain_ind[j*batch_sz: (j*batch_sz + batch_sz),]
            train(Xbatch, Ybatch)

            if j % print_period == 0:
                cost_val, prediction_val = get_prediction(Xtest, Ytest_ind)
                err = error_rate(prediction_val, Ytest)
                print t.get_value()
                print "Cost / err at iteration i=%d, j=%d: %.3f / %.3f" % (i, j , cost_val, err)
                costs_adam.append(cost_val)



    plt.plot(costs, label='Vanilla Gradient Descent')
    plt.plot(costs_momentum, label='Vanilla Momentum')
    plt.plot(costs_nesterov, label='Nesterov Momentum')
    plt.plot(costs_rmsprop, label='RMSprop')
    plt.plot(costs_rmsprop_momentum, label='RMSprop with vanilla Momentum')
    plt.plot(costs_rmsprop_nesterov, label='RMSprop with Nesterov Momentum')
    plt.plot(costs_adam, label='Adam')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
