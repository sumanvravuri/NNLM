'''
Created on Oct 23, 2013

@author: sumanravuri
'''

import sys
import numpy as np
import cudamat as cm
import gnumpy as gnp
import scipy.io as sp
import scipy.linalg as sl
import scipy.optimize as sopt
import math
import copy
import argparse

class Vector_Math:
    #math functions
    def sigmoid(self,inputs): #completed, expensive, should be compiled
        return inputs.logistic()#1/(1+e^-X)
    def softmax(self, inputs): #completed, expensive, should be compiled
        exp_inputs = gnp.exp(inputs - gnp.max(inputs,axis=1)[:,gnp.newaxis])
        return exp_inputs / gnp.sum(exp_inputs, axis=1)[:, gnp.newaxis]
    def weight_matrix_multiply(self,inputs,weights,biases): #completed, expensive, should be compiled
        return gnp.dot(inputs,weights) + biases#[np.newaxis, :]
    

class NNLM_Weight(object):
    def __init__(self, num_previous_tokens = 0, weights = None, bias = None):
        """num_previous_tokens
        weights - actual Neural Network weights, a dictionary with keys corresponding to corresponding weight matrices as follows
         * 1) input layer if indexed by which previous word, i.e., weights[1] is the previous, weights[2] is the one before that and so on
         * 2) output, which is the hidden to output layer - weights['output']
         * 3) one at the projection layer - weights['projection']
        bias - NN biases, same as biases, but without projection
        weights['projection'] - weight matrix to convert one-hot words to dense and compact features
        """
        self.num_previous_tokens = num_previous_tokens
        if weights == None:
            self.weights = dict()
        else:
            self.weights = copy.deepcopy(weights)
        if bias == None:
            self.bias = dict()
        else:
            self.bias = copy.deepcopy(bias)
    def clear(self):
        self.num_layers = 0
        self.weights.clear()
        self.bias.clear()
        self.weight_type.clear()
    def dot(self, nn_weight2, excluded_keys = {'bias': [], 'weights': []}):
        if type(nn_weight2) is not NNLM_Weight:
            print "argument must be of type Neural_Network_Weight... instead of type", type(nn_weight2), "Exiting now..."
            sys.exit()
        return_val = 0
        for key in self.bias.keys():
            if key in excluded_keys['bias']:
                continue
            return_val += gnp.vdot(self.bias[key], nn_weight2.bias[key])
        for key in self.weights.keys():
            if key in excluded_keys['weights']:
                continue
            return_val += gnp.vdot(self.weights[key], nn_weight2.weights[key])
        return return_val
    def __str__(self):
        string = ""
        for key in self.bias.keys():
            string = string + "bias key " + key + "\n"
            string = string + str(self.bias[key]) + "\n"
        for key in self.weights.keys():
            string = string + "weight key " + key + "\n"
            string = string + str(self.weights[key]) + "\n"
        return string
    def print_statistics(self):
        for key in self.bias.keys():
            print "min of bias[" + key + "] is", np.min(self.bias[key]) 
            print "max of bias[" + key + "] is", np.max(self.bias[key])
            print "mean of bias[" + key + "] is", np.mean(self.bias[key])
            print "var of bias[" + key + "] is", np.var(self.bias[key]), "\n"
        for key in self.weights.keys():
            print "min of weights[" + key + "] is", np.min(self.weights[key]) 
            print "max of weights[" + key + "] is", np.max(self.weights[key])
            print "mean of weights[" + key + "] is", np.mean(self.weights[key])
            print "var of weights[" + key + "] is", np.var(self.weights[key]), "\n"
    def norm(self, excluded_keys = {'bias': [], 'weights': []}):
        squared_sum = 0
        for key in self.bias.keys():
            if key in excluded_keys['bias']:
                continue
            squared_sum += gnp.sum(self.bias[key] ** 2)
        for key in self.weights.keys():
            if key in excluded_keys['weights']:
                continue  
            squared_sum += gnp.sum(self.weights[key] ** 2)
        return np.sqrt(squared_sum)
    def max(self, excluded_keys = {'bias': [], 'weights': []}):
        max_val = -float('Inf')
        for key in self.bias.keys():
            if key in excluded_keys['bias']:
                continue
            max_val = gnp.max(gnp.max(self.bias[key]), max_val)
        for key in self.weights.keys():
            if key in excluded_keys['weights']:
                continue  
            max_val = gnp.max(gnp.max(self.weights[key]), max_val)
        return max_val
    def min(self, excluded_keys = {'bias': [], 'weights': []}):
        min_val = float('Inf')
        for key in self.bias.keys():
            if key in excluded_keys['bias']:
                continue
            min_val = gnp.min(gnp.min(self.bias[key]), min_val)
        for key in self.weights.keys():
            if key in excluded_keys['weights']:
                continue  
            min_val = gnp.min(gnp.min(self.weights[key]), min_val)
        return min_val
    def clip(self, clip_min, clip_max, excluded_keys = {'bias': [], 'weights': []}):
        nn_output = copy.deepcopy(self)
        for key in self.bias.keys():
            if key in excluded_keys['bias']:
                continue
            gnp.clip(self.bias[key], clip_min, clip_max, out=nn_output.bias[key])
        for key in self.weights.keys():
            if key in excluded_keys['weights']:
                continue  
            gnp.clip(self.weights[key], clip_min, clip_max, out=nn_output.weights[key])
        return nn_output
    def get_architecture(self):
        return [self.bias[str(layer_num)].size for layer_num in range(self.num_layers+1) ]
    @property
    def size(self, excluded_keys = {'bias': [], 'weights': []}):
        numel = 0
        for key in self.bias.keys():
            if key in excluded_keys['bias']:
                continue
            numel += self.bias[key].size
        for key in self.weights.keys():
            if key in excluded_keys['weights']:
                continue  
            numel += self.weights[key].size
        return numel
    def open_weights(self, weight_matrix_name): #completed
        #the weight file format is very specific, it contains the following variables:
        #weights01, weights12, weights23, ...
        #bias0, bias1, bias2, bias3, ....
        #weights01_type, weights12_type, weights23_type, etc...
        #optional variables:
        #num_layers
        #everything else will be ignored
        try:
            weight_dict = sp.loadmat(weight_matrix_name)
        except IOError:
            print "Unable to open", weight_matrix_name, "exiting now"
            sys.exit()
        if 'num_previous_tokens' in weight_dict:
            self.num_previous_tokens = weight_dict['num_previous_tokens'][0]
            if type(self.num_previous_tokens) is not int: #hack because write_weights() stores num_layers as [[num_layers]] 
                self.num_previous_tokens = self.num_previous_tokens[0]
        else: #count number of biases for num_layers
            self.num_previous_tokens = 0
            for layer_num in range(1,101): #maximum number of layers currently is set to 100
                if str(layer_num) in weight_dict:
                    self.num_previous_tokens += 1
                else:
                    break
            if self.num_previous_tokens == 0:
                print "no weight matrices for previous layer weights found... must be greater than 0 ... Exiting now"
                sys.exit()
        for layer_num in range(1,self.num_previous_tokens+1):
            cur_layer = str(layer_num)
            self.weights[layer_num] = weight_dict[''.join(['weights', cur_layer])]
            self.bias[layer_num] = weight_dict[''.join(['bias', cur_layer])]
        self.projection_layer = weight_dict['projection_layer']
        self.weights['output'] = weight_dict['weights_output']
        self.bias['output'] = weight_dict['bias_output']
        del weight_dict
        self.check_weights()
    def init_random_weights(self, architecture, initial_bias_max, initial_bias_min, initial_weight_min, 
                           initial_weight_max, seed = 0): #completed, expensive, should be compiled
        np.random.seed(seed)
        
        self.projection_layer = np.random.randn(architecture[0], architecture[1])
        self.num_layers = len(architecture) - 1
        initial_bias_range = initial_bias_max - initial_bias_min
        initial_weight_range = initial_weight_max - initial_weight_min
        self.bias['0'] = initial_bias_min + initial_bias_range * np.random.random_sample((1,architecture[0]))
        
        for layer_num in range(1,self.num_layers+1):
            weight_cur_layer = ''.join([str(layer_num-1),str(layer_num)])
            bias_cur_layer = str(layer_num)
            #print "initializing weight layer", weight_cur_layer, "and bias layer", bias_cur_layer
            self.bias[bias_cur_layer] = initial_bias_min + initial_bias_range * np.random.random_sample((1,architecture[layer_num]))
            self.weights[weight_cur_layer]=(initial_weight_min + initial_weight_range * 
                                            np.random.random_sample( (architecture[layer_num-1],architecture[layer_num]) ))
        
        print "Finished Initializing Weights"
        self.check_weights()
    def init_zero_weights(self, architecture, last_layer_logistic=True, verbose=False):
        self.num_layers = len(architecture) - 1
        self.bias['0'] = np.zeros((1,architecture[0]))
        
        for layer_num in range(1,self.num_layers+1):
            weight_cur_layer = ''.join([str(layer_num-1),str(layer_num)])
            bias_cur_layer = str(layer_num)
            #print "initializing weight layer", weight_cur_layer, "and bias layer", bias_cur_layer
            self.bias[bias_cur_layer] = np.zeros((1,architecture[layer_num]))
            self.weights[weight_cur_layer] = np.zeros( (architecture[layer_num-1],architecture[layer_num]) )
            if layer_num == 0:
                self.weight_type[weight_cur_layer] = 'rbm_gaussian_bernoulli'
            elif layer_num == self.num_layers and last_layer_logistic == True:
                self.weight_type[weight_cur_layer] = 'logistic'
            else:
                self.weight_type[weight_cur_layer] = 'rbm_bernoulli_bernoulli'
        if verbose:
            print "Finished Initializing Weights"
        self.check_weights(False)
    def check_weights(self, verbose=True): #need to check consistency of features with weights
        #checks weights to see if following conditions are true
        # *feature dimension equal to number of rows of first layer (if weights are stored in n_rows x n_cols)
        # *n_cols of (n-1)th layer == n_rows of nth layer
        # if only one layer, that weight layer type is logistic, gaussian_bernoulli or bernoulli_bernoulli
        # check is biases match weight values
        # if multiple layers, 0 to (n-1)th layer is gaussian bernoulli RBM or bernoulli bernoulli RBM and last layer is logistic regression
        
        #if below is true, not running in logistic regression mode, so first layer must be an RBM
        if verbose:
            print "Checking weights...",
        if self.num_layers > 1: 
            if self.weight_type['01'] not in self.valid_layer_types['intermediate']:
                print self.weight_type['01'], "is not valid layer type. Must be one of the following:", self.valid_layer_types['intermediate'], "...Exiting now"
                sys.exit()
        
        #check biases
        if self.bias['0'].shape[1] != self.weights['01'].shape[0]:
            print "Number of visible bias dimensions: ", self.bias['0'].shape[1],
            print " of layer 0 does not equal visible weight dimensions ", self.weights['01'].shape[0], "... Exiting now"
            sys.exit()
        if self.bias['1'].shape[1] != self.weights['01'].shape[1]:
            print "Number of hidden bias dimensions: ", self.bias['1'].shape[1],
            print " of layer 0 does not equal hidden weight dimensions ", self.weights['01'].shape[1], "... Exiting now"
            sys.exit()
        
        #intermediate layers need to have correct shape and RBM type
        for layer_num in range(1,self.num_layers-1): 
            weight_prev_layer = ''.join([str(layer_num-1),str(layer_num)])
            weight_cur_layer = ''.join([str(layer_num),str(layer_num+1)])
            bias_prev_layer = str(layer_num)
            bias_cur_layer = str(layer_num+1)
            #check shape
            if self.weights[weight_prev_layer].shape[1] != self.weights[weight_cur_layer].shape[0]:
                print "Dimensionality of", weight_prev_layer, "\b:", self.weights[weight_prev_layer].shape, "does not match dimensionality of", weight_cur_layer, "\b:",self.weights[weight_cur_layer].shape
                print "The second dimension of", weight_prev_layer, "must equal the first dimension of", weight_cur_layer
                sys.exit()
            #check RBM type
            if self.weight_type[weight_cur_layer] not in self.valid_layer_types['intermediate']:
                print self.weight_type[weight_cur_layer], "is not valid layer type. Must be one of the following:", self.valid_layer_types['intermediate'], "...Exiting now"
                sys.exit()
            #check biases
            if self.bias[bias_prev_layer].shape[1] != self.weights[weight_cur_layer].shape[0]:
                print "Number of visible bias dimensions:", self.bias[bias_prev_layer].shape[1], "of layer", weight_cur_layer, "does not equal visible weight dimensions:", self.weights[weight_cur_layer].shape[0]
                sys.exit()
            if self.bias[bias_cur_layer].shape[1] != self.weights[weight_cur_layer].shape[1]:
                print "Number of hidden bias dimensions:", self.bias[bias_cur_layer].shape[1],"of layer", weight_cur_layer, "does not equal hidden weight dimensions", self.weights[weight_cur_layer].shape[1]
                sys.exit()
        
        #check last layer
        layer_num = self.num_layers-1
        weight_prev_layer = ''.join([str(layer_num-1),str(layer_num)])
        weight_cur_layer = ''.join([str(layer_num),str(layer_num+1)])
        bias_prev_layer = str(layer_num)
        bias_cur_layer = str(layer_num+1)
        #check if last layer is of type logistic
        if self.weight_type[weight_cur_layer] not in self.valid_layer_types['last']:
            print self.weight_type[weight_cur_layer], " is not valid type for last layer.", 
            print "Must be one of the following:", self.valid_layer_types['last'], "...Exiting now"
            sys.exit()
        #check shape if hidden layer is used
        if self.num_layers > 1:
            if self.weights[weight_prev_layer].shape[1] != self.weights[weight_cur_layer].shape[0]:
                print "Dimensionality of", weight_prev_layer, "\b:", self.weights[weight_prev_layer].shape, "does not match dimensionality of", weight_cur_layer, "\b:",self.weights[weight_cur_layer].shape
                print "The second dimension of", weight_prev_layer, "must equal the first dimension of", weight_cur_layer
                sys.exit()
            #check biases
            if self.bias[bias_prev_layer].shape[1] != self.weights[weight_cur_layer].shape[0]:
                print "Number of visible bias dimensions:", self.weights[bias_prev_layer].shape[1], "of layer", weight_cur_layer, "does not equal visible weight dimensions:", self.weights[weight_cur_layer].shape[0]
                sys.exit()
            if self.bias[bias_cur_layer].shape[1] != self.weights[weight_cur_layer].shape[1]:
                print "Number of hidden bias dimensions:", self.weights[bias_cur_layer].shape[1],"of layer", weight_cur_layer, "does not equal hidden weight dimensions", self.weights[weight_cur_layer].shape[1]
                sys.exit()
        if verbose:
            print "seems copacetic"
    def write_weights(self, output_name): #completed
        weight_dict = dict()
        weight_dict['num_previous_tokens'] = self.num_previous_tokens
        for layer_num in range(1, self.num_previous_tokens+1):
            cur_layer = str(layer_num)
            weight_dict[''.join(['bias', cur_layer])] = self.bias[layer_num]
            weight_dict[''.join(['weights', cur_layer])] = self.weights[layer_num]
        weight_dict['weights_output'] = self.weights['output']
        weight_dict['bias_output'] = self.bias['output']
        weight_dict['projection_layer'] = self.projection_layer
        try:
            sp.savemat(output_name, weight_dict, oned_as='column')
        except IOError:
            print "Unable to save ", self.output_name, "... Exiting now"
            sys.exit()
        else:
            print output_name, "successfully saved"
            del weight_dict
    def __neg__(self):
        nn_output = copy.deepcopy(self)
        for key in self.bias.keys():
            nn_output.bias[key] = -self.bias[key]
        for key in self.weights.keys():
            nn_output.weights[key] = -self.weights[key]
        return nn_output
    def __add__(self,addend):
        nn_output = copy.deepcopy(self)
        if type(addend) is NNLM_Weight:
            if self.get_architecture() != addend.get_architecture():
                print "Neural net models do not match... Exiting now"
                sys.exit()
            
            for key in self.bias.keys():
                nn_output.bias[key] = self.bias[key] + addend.bias[key]
            for key in self.weights.keys():
                nn_output.weights[key] = self.weights[key] + addend.weights[key]
            return nn_output
        #otherwise type is scalar
        addend = float(addend)
        for key in self.bias.keys():
            nn_output.bias[key] = self.bias[key] + addend
        for key in self.weights.keys():
            nn_output.weights[key] = self.weights[key] + addend
        return nn_output
        
    def __sub__(self,subtrahend):
        nn_output = copy.deepcopy(self)
        if type(subtrahend) is NNLM_Weight:
            if self.get_architecture() != subtrahend.get_architecture():
                print "Neural net models do not match... Exiting now"
                sys.exit()
            
            for key in self.bias.keys():
                nn_output.bias[key] = self.bias[key] - subtrahend.bias[key]
            for key in self.weights.keys():
                nn_output.weights[key] = self.weights[key] - subtrahend.weights[key]
            return nn_output
        #otherwise type is scalar
        subtrahend = float(subtrahend)
        for key in self.bias.keys():
            nn_output.bias[key] = self.bias[key] - subtrahend
        for key in self.weights.keys():
            nn_output.weights[key] = self.weights[key] - subtrahend
        return nn_output
    def __mul__(self, multiplier):
        #if type(scalar) is not float and type(scalar) is not int:
        #    print "__mul__ must be by a float or int. Instead it is type", type(scalar), "Exiting now"
        #    sys.exit()
        nn_output = copy.deepcopy(self)
        if type(multiplier) is NNLM_Weight:
            for key in self.bias.keys():
                nn_output.bias[key] = self.bias[key] * multiplier.bias[key]
            for key in self.weights.keys():
                nn_output.weights[key] = self.weights[key] * multiplier.weights[key]
            return nn_output
        #otherwise scalar type
        multiplier = float(multiplier)
        
        for key in self.bias.keys():
            nn_output.bias[key] = self.bias[key] * multiplier
        for key in self.weights.keys():
            nn_output.weights[key] = self.weights[key] * multiplier
        return nn_output
    def __div__(self, divisor):
        #if type(scalar) is not float and type(scalar) is not int:
        #    print "Divide must be by a float or int. Instead it is type", type(scalar), "Exiting now"
        #    sys.exit()
        nn_output = copy.deepcopy(self)
        if type(divisor) is NNLM_Weight:
            for key in self.bias.keys():
                nn_output.bias[key] = self.bias[key] / divisor.bias[key]
            for key in self.weights.keys():
                nn_output.weights[key] = self.weights[key] / divisor.weights[key]
            return nn_output
        #otherwise scalar type
        divisor = float(divisor)
        
        for key in self.bias.keys():
            nn_output.bias[key] = self.bias[key] / divisor
        for key in self.weights.keys():
            nn_output.weights[key] = self.weights[key] / divisor
        return nn_output
    def __iadd__(self, nn_weight2):
        if type(nn_weight2) is not NNLM_Weight:
            print "argument must be of type NNLM_Weight... instead of type", type(nn_weight2), "Exiting now..."
            sys.exit()
        if self.get_architecture() != nn_weight2.get_architecture():
            print "Neural net models do not match... Exiting now"
            sys.exit()

        for key in self.bias.keys():
            self.bias[key] += nn_weight2.bias[key]
        for key in self.weights.keys():
            self.weights[key] += nn_weight2.weights[key]
        return self
    def __isub__(self, nn_weight2):
        if type(nn_weight2) is not NNLM_Weight:
            print "argument must be of type NNLM_Weight... instead of type", type(nn_weight2), "Exiting now..."
            sys.exit()
        if self.get_architecture() != nn_weight2.get_architecture():
            print "Neural net models do not match... Exiting now"
            sys.exit()

        for key in self.bias.keys():
            self.bias[key] -= nn_weight2.bias[key]
        for key in self.weights.keys():
            self.weights[key] -= nn_weight2.weights[key]
        return self
    def __imul__(self, scalar):
        #if type(scalar) is not float and type(scalar) is not int:
        #    print "__imul__ must be by a float or int. Instead it is type", type(scalar), "Exiting now"
        #    sys.exit()
        scalar = float(scalar)
        for key in self.bias.keys():
            self.bias[key] *= scalar
        for key in self.weights.keys():
            self.weights[key] *= scalar
        return self
    def __idiv__(self, scalar):
        scalar = float(scalar)
        for key in self.bias.keys():
            self.bias[key] /= scalar
        for key in self.weights.keys():
            self.weights[key] /= scalar
        return self
    def __pow__(self, scalar):
        if scalar == 2:
            return self * self
        scalar = float(scalar)
        nn_output = copy.deepcopy(self)
        for key in self.bias.keys():
            nn_output.bias[key] = self.bias[key] ** scalar
        for key in self.weights.keys():
            nn_output.weights[key] = self.weights[key] ** scalar
        return nn_output
    def __copy__(self):
        return NNLM_Weight(self.num_layers, self.weights, self.bias, self.weight_type)
    def __deepcopy__(self, memo):
        return NNLM_Weight(copy.deepcopy(self.num_layers, memo), copy.deepcopy(self.weights,memo), 
                                     copy.deepcopy(self.bias,memo), copy.deepcopy(self.weight_type,memo))