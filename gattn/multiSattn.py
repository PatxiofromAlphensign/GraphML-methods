import warnings
import argparse
from base_gattn import BaseGAttN
import tensorflow as tf 
import numpy as np

def calulate_params(logits):
    if len(logits.shape) < 2:
        raise ValueError('must be more than 1 dimention ')
    comm_attn = []
    for i, x  in enumerate(logits):
        attnL = np.mean(x)
        if attnL in logits[i:len(logits)]:
            comm_attn.append(attnL)

    return comm_attn

base = BaseGAttN

def dummy_vals(rng): # no labels requried
    assert rng%2 == 0
    logits2x2 = np.array([n for n in np.arange(1,(rng+1))]).reshape(2,int(rng/2))
    attnHead = calulate_params(logits2x2)
    logits2x2 = tf.cast(logits2x2, tf.float32)
    labels = list(logits2x2.shape)
   
    labels[labels.index(max(labels))]-=1
    
    return logits2x2, labels


def forward(prob):
    #implement alpha teleportation 

    pass

def flows(values):
    assert len(values) == 2
    logits,labels = values
    base = BaseGAttN 
    confmat = tf.cast(base.confmat(logits, labels), dtype=tf.float64)
    batched_confmat = []
    for i in range(1, confmat.shape.ndims + 1):
        forward = 0.5 + 1/np.arctan(i)*np.pi
        
        batched_confmat.append( confmat[i+int(forward)])

    return batched_confmat


def loss_on_iter(count):
    global base
    logit, labels = dummy_vals(args.range)
    losses = []

    #base.masked_softmax_cross_entropy(logit, labels, np.mean(losses))
    for _ in range(count):
        losses.append(base.loss(logit, labels, 0, 0.5))

    return np.mean(losses)


parser = argparse.ArgumentParser()
parser.add_argument('--range', help='range', default=100, type=int)
parser.add_argument('--count', help='count', default=3, type=int)
args = parser.parse_args()
tf.compat.v1.enable_eager_execution()
dummpy_vals = '' #extract1_q, commns
if dummy_vals(args.range):
    values = dummy_vals(args.range)
#with tf.compat.v1.enable_eager_execution():


print(flows(values))
loss = loss_on_iter(args.count)
#confmat = flows(dummy_vals(args.range))
print(loss)








