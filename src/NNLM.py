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
        self.num_previous_tokens = 0
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
        return [self.weights['projection'].shape[1], self.bias['prehidden'].size, self.bias['output'].size]
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
                if 'weights' + str(layer_num) in weight_dict:
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
        self.weights['prehidden'] = weight_dict['weights_prehidden']
        self.bias['prehidden'] = weight_dict['bias_prehidden']
        self.weights['output'] = weight_dict['weights_output']
        self.bias['output'] = weight_dict['bias_output']
        del weight_dict
        self.check_weights()
    def init_random_weights(self, architecture, initial_bias_max, initial_bias_min, initial_weight_min, 
                           initial_weight_max, seed = 0, verbose = False): #completed, expensive, should be compiled
        np.random.seed(seed)
#        print architecture
        self.weights['projection'] = gnp.randn(architecture[-1], architecture[1])
        print self.weights['projection'].shape
        initial_bias_range = initial_bias_max - initial_bias_min
        initial_weight_range = initial_weight_max - initial_weight_min
        num_prehiddens = architecture[1] * self.num_previous_tokens
        self.bias['prehidden'] = gnp.garray(initial_bias_min + initial_bias_range * np.random.random_sample((1,architecture[2])))
        self.weights['prehidden'] = gnp.garray(initial_weight_min + initial_weight_range * 
                                      	     np.random.random_sample( (num_prehiddens, architecture[2]) ))
        
        self.bias['output'] = gnp.garray(initial_bias_min + initial_bias_range * np.random.random_sample((1,architecture[-1])))
        self.weights['output'] = gnp.garray(initial_weight_min + initial_weight_range * 
                                  	    np.random.random_sample( (architecture[-2],architecture[-1]) ))
        
        if verbose:
            print "Finished Initializing Weights"
        self.write_weights('init_random_weights.mat')
#        self.check_weights()
    def init_zero_weights(self, architecture, verbose=False):
        self.weights['projection'] = gnp.zeros((architecture[0], architecture[1]))

        num_prehiddens = architecture[1] * self.num_previous_tokens        

        self.bias['prehidden'] = gnp.zeros((1,architecture[2]))
        self.weights['prehidden'] = gnp.zeros( (num_prehiddens, architecture[2]) )
        
        self.bias['output'] = gnp.zeros((1,architecture[-1]))
        self.weights['output'] = gnp.zeros( (architecture[-2],architecture[-1]) )
        
        if verbose:
            print "Finished Initializing Weights"
#        self.check_weights(False)
    def check_weights(self, verbose=True): #need to check consistency of features with weights
        #checks weights to see if following conditions are true
        # *feature dimension equal to number of rows of first layer (if weights are stored in n_rows x n_cols)
        # *n_cols of (n-1)th layer == n_rows of nth layer
        # if only one layer, that weight layer type is logistic, gaussian_bernoulli or bernoulli_bernoulli
        # check is biases match weight values
        # if multiple layers, 0 to (n-1)th layer is gaussian bernoulli RBM or bernoulli bernoulli RBM and last layer is logistic regression
        
        #if below is true, not running in logistic regression mode, so first layer must be an RBM
        if verbose:
            print "Checking weights..."
        
        proto_weight_shape = self.weights[1].shape
        proto_bias_shape = self.bias[1].shape
        #intermediate layers need to have correct shape and RBM type
        for layer_num in range(1,self.num_previous_tokens+1): 
            #check shape
            if self.weights[layer_num].shape != proto_weight_shape:
                print "Dimensionality of weights at token %d do not match needed weight shape" % layer_num
                print "Needed weight shape", proto_weight_shape
                print "Given weight shape", self.weights[layer_num].shape
                print "Exiting Now..."
                sys.exit()
            #check biases
            if self.bias[layer_num].shape != proto_bias_shape:
                print "Dimensionality of biases at token %d do not match needed weight shape" % layer_num
                print "Needed bias shape", proto_bias_shape
                print "Given bias shape", self.bias[layer_num].shape
                print "Exiting Now..."
                sys.exit()
            #check consistency of weights and biases
            if self.bias[layer_num].shape[1] != self.weights[layer_num].shape[1]:
                print "Number of hidden bias dimensions:", self.bias[layer_num].shape[1],"of layer", layer_num, "does not equal hidden weight dimensions", self.weights[layer_num].shape[1]
                print "Exiting now"
                sys.exit()
            #check consistency of projection layer and layer_num
            if self.weights[layer_num].shape[0] != self.weights['projection'].shape[1]:
                print "Number of output dimensions of projection layer %d does not match number of input dimensions for first weight layer (%d) of previous token %d" % (self.weights['projection'].shape[1], self.weights[layer_num].shape[0], layer_num)
                print "Exiting now"
                sys.exit()
        
        #check last layer
        num_hiddens = self.bias[1].size * self.num_previous_tokens
        if self.weights['output'].shape[0] != num_hiddens:
            print "The number of inputs to output weight layer %d does not equal the number of hiddens %d" % (self.weights['output'].shape[0], num_hiddens)
            print "Exiting now..."
            sys.exit()
        if self.bias['output'].shape[1] != self.weights['output'].shape[1]:
            print "Number of hidden bias dimensions:", self.bias[layer_num].shape[1],"of layer", layer_num, "does not equal hidden weight dimensions", self.weights[layer_num].shape[1]
            print "Exiting now"
            sys.exit()
        if verbose:
            print "seems copacetic"
    def write_weights(self, output_name): #completed
        weight_dict = dict()
        weight_dict['num_previous_tokens'] = self.num_previous_tokens
        weight_dict['projection_layer'] = self.weights['projection'].as_numpy_array()
        weight_dict['bias_prehidden'] = self.bias['prehidden'].as_numpy_array()
        weight_dict['weights_prehidden'] = self.weights['prehidden'].as_numpy_array()
        weight_dict['weights_output'] = self.weights['output'].as_numpy_array()
        weight_dict['bias_output'] = self.bias['output'].as_numpy_array()
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
        return NNLM_Weight(self.num_previous_tokens, self.weights, self.bias, self.weight_type)
    def __deepcopy__(self, memo):
        return NNLM_Weight(copy.deepcopy(self.num_previous_tokens, memo), copy.deepcopy(self.weights,memo), 
                                     copy.deepcopy(self.bias,memo), copy.deepcopy(self.weight_type,memo))
        


class NNLM(object, Vector_Math):
    #features are stored in format ndata x nvis
    #weights are stored as nvis x nhid at feature level
    #biases are stored as 1 x nhid
    #rbm_type is either gaussian_bernoulli, bernoulli_bernoulli, notrbm_logistic
    def __init__(self, config_dictionary): #completed
        #variables for Neural Network: feature_file_name(read from)
        #required_variables - required variables for running system
        #all_variables - all valid variables for each type
        self.feature_file_name = self.default_variable_define(config_dictionary, 'feature_file_name', arg_type='string')
#        print "Amount of memory in use before reading feature file is", gnp.memory_in_use(True), "MB"
        self.features = self.read_feature_file()
        self.num_previous_tokens = self.features.shape[1]
#        print "Amount of memory in use after reading feature file is", gnp.memory_in_use(True), "MB"
        self.model = NNLM_Weight(self.num_previous_tokens)
        self.output_name = self.default_variable_define(config_dictionary, 'output_name', arg_type='string')
        
        self.required_variables = dict()
        self.all_variables = dict()
        self.required_variables['train'] = ['mode', 'feature_file_name', 'output_name']
        self.all_variables['train'] = self.required_variables['train'] + ['label_file_name', 'hiddens_structure', 'weight_matrix_name', 
                               'initial_weight_max', 'initial_weight_min', 'initial_bias_max', 'initial_bias_min', 'save_each_epoch',
                               'do_pretrain', 'pretrain_method', 'pretrain_iterations', 
                               'pretrain_learning_rate', 'pretrain_batch_size',
                               'do_backprop', 'backprop_method', 'backprop_batch_size', 'l2_regularization_const',
                               'num_epochs', 'num_line_searches', 'armijo_const', 'wolfe_const',
                               'steepest_learning_rate',
                               'conjugate_max_iterations', 'conjugate_const_type',
                               'truncated_newton_num_cg_epochs', 'truncated_newton_init_damping_factor',
                               'krylov_num_directions', 'krylov_num_batch_splits', 'krylov_num_bfgs_epochs', 'second_order_matrix',
                               'krylov_use_hessian_preconditioner', 'krylov_eigenvalue_floor_const', 
                               'fisher_preconditioner_floor_val', 'use_fisher_preconditioner']
        self.required_variables['test'] =  ['mode', 'feature_file_name', 'weight_matrix_name', 'output_name']
        self.all_variables['test'] =  self.required_variables['test'] + ['label_file_name']
    def dump_config_vals(self):
        no_attr_key = list()
        print "********************************************************************************"
        print "Neural Network configuration is as follows:"
        
        for key in self.all_variables[self.mode]:
            if hasattr(self,key):
                print key, "=", eval('self.' + key)
            else:
                no_attr_key.append(key)
                
        print "********************************************************************************"
        print "Undefined keys are as follows:"
        for key in no_attr_key:
            print key, "not set"
        print "********************************************************************************"
    def default_variable_define(self,config_dictionary,config_key, arg_type='string', 
                                default_value=None, error_string=None, exit_if_no_default=True,
                                acceptable_values=None):
        #arg_type is either int, float, string, int_comma_string, float_comma_string, boolean
        try:
            if arg_type == 'int_comma_string':
                return self.read_config_comma_string(config_dictionary[config_key], needs_int=True)
            elif arg_type == 'float_comma_string':
                return self.read_config_comma_string(config_dictionary[config_key], needs_int=False)
            elif arg_type == 'int':
                return int(config_dictionary[config_key])
            elif arg_type == 'float':
                return float(config_dictionary[config_key])
            elif arg_type == 'string':
                return config_dictionary[config_key]
            elif arg_type == 'boolean':
                if config_dictionary[config_key] == 'False' or config_dictionary[config_key] == '0' or config_dictionary[config_key] == 'F':
                    return False
                elif config_dictionary[config_key] == 'True' or config_dictionary[config_key] == '1' or config_dictionary[config_key] == 'T':
                    return True
                else:
                    print config_dictionary[config_key], "is not valid for boolean type... Acceptable values are True, False, 1, 0, T, or F... Exiting now"
                    sys.exit()
            else:
                print arg_type, "is not a valid type, arg_type can be either int, float, string, int_comma_string, float_comma_string... exiting now"
                sys.exit()
        except KeyError:
            if error_string != None:
                print error_string
            else:
                print "No", config_key, "defined,",
            if default_value == None and exit_if_no_default:
                print "since", config_key, "must be defined... exiting now"
                sys.exit()
            else:
                if acceptable_values != None and (default_value not in acceptable_values):
                    print default_value, "is not an acceptable input, acceptable inputs are", acceptable_values, "... Exiting now"
                    sys.exit()
                if error_string == None:
                    print "setting", config_key, "to", default_value
                return default_value
    def read_feature_file(self): #completed
        try:
            return sp.loadmat(self.feature_file_name)['features'] #in MATLAB format
        except IOError:
            print "Unable to open ", self.feature_file_name, "... Exiting now"
            sys.exit()
    def read_label_file(self): #completed
        try:
            return sp.loadmat(self.label_file_name)['labels'] #in MATLAB format
        except IOError:
            print "Unable to open ", self.label_file_name, "... Exiting now"
            sys.exit()
    def read_config_comma_string(self,input_string,needs_int=False):
        output_list = []
        for elem in input_string.split(','):
            if '*' in elem:
                elem_list = elem.split('*')
                if needs_int:
                    output_list.extend([int(elem_list[1])] * int(elem_list[0]))
                else:
                    output_list.extend([float(elem_list[1])] * int(elem_list[0]))
            else:
                if needs_int:
                    output_list.append(int(elem))
                else:
                    output_list.append(float(elem))
        return output_list
    def levenshtein_string_edit_distance(self, string1, string2): #completed
        dist = dict()
        string1_len = len(string1)
        string2_len = len(string2)
        
        for idx in range(-1,string1_len+1):
            dist[(idx, -1)] = idx + 1
        for idx in range(-1,string2_len+1):
            dist[(-1, idx)] = idx + 1
            
        for idx1 in range(string1_len):
            for idx2 in range(string2_len):
                if string1[idx1] == string2[idx2]:
                    cost = 0
                else:
                    cost = 1
                dist[(idx1,idx2)] = min(
                           dist[(idx1-1,idx2)] + 1, # deletion
                           dist[(idx1,idx2-1)] + 1, # insertion
                           dist[(idx1-1,idx2-1)] + cost, # substitution
                           )
                if idx1 and idx2 and string1[idx1]==string2[idx2-1] and string1[idx1-1] == string2[idx2]:
                    dist[(idx1,idx2)] = min (dist[(idx1,idx2)], dist[idx1-2,idx2-2] + cost) # transposition
        return dist[(string1_len-1, string2_len-1)]    
    def check_keys(self, config_dictionary): #completed
        print "Checking config keys...",
        exit_flag = False
        
        config_dictionary_keys = config_dictionary.keys()
        
        if self.mode == 'train':
            correct_mode = 'train'
            incorrect_mode = 'test'
        elif self.mode == 'test':
            correct_mode = 'test'
            incorrect_mode = 'train'
            
        for req_var in self.required_variables[correct_mode]:
            if req_var not in config_dictionary_keys:
                print req_var, "needs to be set for", correct_mode, "but is not."
                if exit_flag == False:
                    print "Because of above error, will exit after checking rest of keys"
                    exit_flag = True
        
        for var in config_dictionary_keys:
            if var not in self.all_variables[correct_mode]:
                print var, "in the config file given is not a valid key for", correct_mode
                if var in self.all_variables[incorrect_mode]:
                    print "but", var, "is a valid key for", incorrect_mode, "so either the mode or key is incorrect"
                else:
                    string_distances = np.array([self.levenshtein_string_edit_distance(var, string2) for string2 in self.all_variables[correct_mode]])
                    print "perhaps you meant ***", self.all_variables[correct_mode][np.argmin(string_distances)], "\b*** (levenshtein string edit distance", np.min(string_distances), "\b) instead of ***", var, "\b***?"
                if exit_flag == False:
                    print "Because of above error, will exit after checking rest of keys"
                    exit_flag = True
        
        if exit_flag:
            print "Exiting now"
            sys.exit()
        else:
            print "seems copacetic"
    def check_labels(self): #ugly, I should extend gnumpy to include a len, a unitq and bincount functions
        print "Checking labels..."
        #labels = np.array([int(x) for x in self.labels.as_numpy_array()])
        if len(self.labels.shape) != 1 and ((len(self.labels.shape) == 2 and self.labels.shape[1] != 1) or len(self.labels.shape) > 2):
            print "labels need to be in (n_samples) or (n_samples,1) format and the shape of labels is ", self.labels.shape, "... Exiting now"
            sys.exit()
        if len(self.labels.shape) == 2 and self.labels.shape[1] != 1:
            self.labels = self.labels.reshape(-1)
        if self.labels.size != self.features.shape[0]:
            print "Number of examples in feature file: ", self.features.shape[0], " does not equal size of label file, ", self.labels.size, "... Exiting now"
            sys.exit()
#        if  [i for i in np.unique(self.labels)] != range(np.max(self.labels)+1):
#            print "Labels need to be in the form 0,1,2,....,n,... Exiting now"
            sys.exit()      
        print "labels seem copacetic"
    def forward_layer(self, inputs, model = None, layer_type = None): #completed
        if model == None:
            model = self.model
        if layer_type == 'projection':
            projection_layer_size = model.weights['projection'].shape[1]
            num_examples = inputs.shape[0]
            num_outs = self.model.num_previous_tokens * projection_layer_size
            outputs = gnp.zeros((num_examples, num_outs))
            for weight_index in range(self.model.num_previous_tokens):
                start_index = projection_layer_size * weight_index
                end_index = projection_layer_size * (1 + weight_index)
#            	print inputs[:,weight_index].astype(int)
#                print model.weights['projection'][inputs[:,weight_index].astype(int),:]
#                print outputs[:,start_index:end_index].shape
                outputs[:,start_index:end_index] = model.weights['projection'][(inputs[:,weight_index]),:]
                gnp.free_reuse_cache(False)
                return outputs
        elif layer_type == 'prehidden':
            #in this case, inputs, weights, and biases are stored in a dictionary
            hiddens_per_token = self.model.get_architecture()[1]
            outputs = self.sigmoid(self.weight_matrix_multiply(inputs, self.model.weights['prehidden'], self.model.bias['prehidden']))
            gnp.free_reuse_cache(False)
            return outputs
        elif layer_type == 'output':
            outputs =  self.softmax(self.weight_matrix_multiply(inputs, model.weights['output'], model.bias['output']))
            gnp.free_reuse_cache(False)
            return outputs
        else:
            print "layer_type", layer_type, "is not a valid layer type.",
            print "Valid layer types are... projection, prehidden, and output. Exiting now..."
            sys.exit()
    def forward_pass_linear(self, inputs, model = None, layer_type = None):
        #to test finite differences calculation for pearlmutter forward pass, just like forward pass, except it spits linear outputs
        if model == None:
            model = self.model
        if layer_type == 'projection':
#            projection_layer_size = model.get_architecture()[0]
            return self.forward_layer(inputs, model, layer_type='projection')
        elif layer_type == 'prehidden':
            #in this case, inputs, weights, and biases are stored in a dictionary
            return self.forward_layer(inputs, model, layer_type='prehidden')
        elif layer_type == 'output':
            outputs =  self.weight_matrix_multiply(inputs, model.weights['output'], model.bias['output'])
            gnp.free_reuse_cache(False)
            return outputs
        else:
            print "layer_type", layer_type, "is not a valid layer type.",
            print "Valid layer types are... projection, prehidden, and output. Exiting now..."
            sys.exit()
    def forward_pass(self, inputs, verbose=True, model=None): #completed
        # forward pass each layer starting with feature level
        # inputs are size n_batch x n_prev_tokens
        if model == None:
            model = self.model 
        
        word_vectors = self.forward_layer(inputs, model, layer_type='projection')
        hiddens = self.forward_layer(word_vectors, model, layer_type='prehidden')
        outputs = self.forward_layer(hiddens, model, layer_type='output')
        del word_vectors, hiddens
        gnp.free_reuse_cache(False)
        return outputs
    def calculate_cross_entropy(self, output, labels): #completed, expensive, should be compiled
        output_cpu = output.as_numpy_array()
        #labels_cpu = np.array([int(x) for x in labels.as_numpy_array()])
        return -np.sum(np.log([max(output_cpu.item((x,labels[x])),1E-12) for x in range(labels.size)]))
    def calculate_classification_accuracy(self, output, labels): #completed, possibly expensive
        prediction = output.argmax(axis=1).reshape(labels.shape)
        classification_accuracy = np.sum(prediction == labels) / float(labels.size)
        return classification_accuracy
    

class NNLM_Tester(NNLM): #completed
    def __init__(self, config_dictionary): #completed
        """runs DNN tester soup to nuts.
        variables are
        feature_file_name - name of feature file to load from
        weight_matrix_name - initial weight matrix to load
        output_name - output predictions
        label_file_name - label file to check accuracy
        required are feature_file_name, weight_matrix_name, and output_name"""
        self.mode = 'test'
        super(NNLM_Tester,self).__init__(config_dictionary)
        self.check_keys(config_dictionary)
        
        self.weight_matrix_name = self.default_variable_define(config_dictionary, 'weight_matrix_name', arg_type='string')
        self.model.open_weights(self.weight_matrix_name)
        self.label_file_name = self.default_variable_define(config_dictionary, 'label_file_name', arg_type='string',error_string="No label_file_name defined, just running forward pass",exit_if_no_default=False)
        if self.label_file_name != None:
            self.labels = self.read_label_file()
            self.check_labels()
        else:
            del self.label_file_name
            
        self.dump_config_vals()
        self.classify()
        self.write_posterior_prob_file()
    def classify(self): #completed
        self.posterior_probs = self.forward_pass(self.features)
        try:
            avg_cross_entropy = self.calculate_cross_entropy(self.posterior_probs, self.labels) / self.labels.size
            print "Average cross-entropy is", avg_cross_entropy
            print "Classification accuracy is", self.calculate_classification_accuracy(self.posterior_probs, self.labels) * 100, "\b%"
        except AttributeError:
            print "no labels given, so skipping classification statistics"    
    def write_posterior_prob_file(self): #completed
        try:
            print "Writing to", self.output_name
            sp.savemat(self.output_name,{'targets' : self.posterior_probs}, oned_as='column') #output name should have .mat extension
        except IOError:
            print "Unable to write to ", self.output_name, "... Exiting now"
            sys.exit()

class NNLM_Trainer(NNLM):
    def __init__(self,config_dictionary): #completed
        """variables in NN_trainer object are:
        mode (set to 'train')
        feature_file_name - inherited from Neural_Network class, name of feature file (in .mat format with variable 'features' in it) to read from
        features - inherited from Neural_Network class, features
        label_file_name - name of label file (in .mat format with variable 'labels' in it) to read from
        labels - labels for backprop
        architecture - specified by n_hid, n_hid, ..., n_hid. # of feature dimensions and # of classes need not be specified
        weight_matrix_name - initial weight matrix, if specified, if not, will initialize from random
        initial_weight_max - needed if initial weight matrix not loaded
        initial_weight_min - needed if initial weight matrix not loaded
        initial_bias_max - needed if initial weight matrix not loaded
        initial_bias_min - needed if initial weight matrix not loaded
        do_pretrain - set to 1 or 0 (probably should change to boolean values)
        pretrain_method - not yet implemented, will either be 'mean_field' or 'sampling'
        pretrain_iterations - # of iterations per RBM. Must be equal to the number of hidden layers
        pretrain_learning_rate - learning rate for each epoch of pretrain. must be equal to # hidden layers * sum(pretrain_iterations)
        pretrain_batch_size - batch size for pretraining
        do_backprop - do backpropagation (set to either 0 or 1, probably should be changed to boolean value)
        backprop_method - either 'steepest_descent', 'conjugate_gradient', or '2nd_order', latter two not yet implemented
        l2_regularization_constant - strength of l2 (weight decay) regularization
        steepest_learning_rate - learning rate for steepest_descent backprop
        backprop_batch_size - batch size for backprop
        output_name - name of weight file to store to.
        At bare minimum, you'll need these variables set to train
        feature_file_name
        output_name
        this will run logistic regression using steepest descent, which is a bad idea"""
        
        #Raise error if we encounter under/overflow during training, because this is bad... code should handle this gracefully
        old_settings = np.seterr(over='raise',under='raise',invalid='raise')
        
        self.mode = 'train'
        super(NNLM_Trainer,self).__init__(config_dictionary)
        self.num_training_examples = self.features.shape[0]
        self.check_keys(config_dictionary)
        #read label file
        self.label_file_name = self.default_variable_define(config_dictionary, 'label_file_name', arg_type='string', error_string="No label_file_name defined, can only do pretraining",exit_if_no_default=False)
        if self.label_file_name != None:
            self.labels = self.read_label_file()
            self.check_labels()
        else:
            del self.label_file_name        
#        print "Amount of memory in use before reading labels file is", gnp.memory_in_use(True), "MB"
        #initialize weights
        self.weight_matrix_name = self.default_variable_define(config_dictionary, 'weight_matrix_name', exit_if_no_default=False)

        if self.weight_matrix_name != None:
            print "Since weight_matrix_name is defined, ignoring possible value of hiddens_structure"
            self.model.open_weights(self.weight_matrix_name)
        else: #initialize model
            del self.weight_matrix_name
            
            self.hiddens_structure = self.default_variable_define(config_dictionary, 'hiddens_structure', arg_type='int_comma_string', exit_if_no_default=True)
            architecture = [self.features.shape[1]] + self.hiddens_structure
            if hasattr(self, 'labels'):
                architecture.append(np.max(self.labels)+1) #will have to change later if I have soft weights
                
            self.initial_weight_max = self.default_variable_define(config_dictionary, 'initial_weight_max', arg_type='float', default_value=0.1)
            self.initial_weight_min = self.default_variable_define(config_dictionary, 'initial_weight_min', arg_type='float', default_value=-0.1)
            self.initial_bias_max = self.default_variable_define(config_dictionary, 'initial_bias_max', arg_type='float', default_value=0.1)
            self.initial_bias_min = self.default_variable_define(config_dictionary, 'initial_bias_max', arg_type='float', default_value=-0.1)
            self.model.init_random_weights(architecture, self.initial_bias_max, self.initial_bias_min, 
                                           self.initial_weight_min, self.initial_weight_max)
            del architecture #we have it in the model
        #
#        print "Amount of memory in use before reading weights file is", gnp.memory_in_use(True), "MB"
        self.save_each_epoch = self.default_variable_define(config_dictionary, 'save_each_epoch', arg_type='boolean', default_value=False)
        #pretraining configuration
#        self.do_pretrain = self.default_variable_define(config_dictionary, 'do_pretrain', default_value=False, arg_type='boolean')
#        if self.do_pretrain:
#            self.pretrain_method = self.default_variable_define(config_dictionary, 'pretrain_method', default_value='mean_field', acceptable_values=['mean_field', 'sampling'])
#            self.pretrain_iterations = self.default_variable_define(config_dictionary, 'pretrain_iterations', default_value=[5] * len(self.hiddens_structure), 
#                                                                    error_string="No pretrain_iterations defined, setting pretrain_iterations to default 5 per layer", 
#                                                                    arg_type='int_comma_string')
#
#            weight_last_layer = ''.join([str(self.model.num_layers-1), str(self.model.num_layers)])
#            if self.model.weight_type[weight_last_layer] == 'logistic' and (len(self.pretrain_iterations) != self.model.num_layers - 1):
#                print "given layer type", self.model.weight_type[weight_last_layer], "pretraining iterations length should be", self.model.num_layers-1, "but pretraining_iterations is length ", len(self.pretrain_iterations), "... Exiting now"
#                sys.exit()
#            elif self.model.weight_type[weight_last_layer] != 'logistic' and (len(self.pretrain_iterations) != self.model.num_layers):
#                print "given layer type", self.model.weight_type[weight_last_layer], "pretraining iterations length should be", self.model.num_layers, "but pretraining_iterations is length ", len(self.pretrain_iterations), "... Exiting now"
#                sys.exit()
#            self.pretrain_learning_rate = self.default_variable_define(config_dictionary, 'pretrain_learning_rate', default_value=[0.01] * sum(self.pretrain_iterations), 
#                                                                       error_string="No pretrain_learning_rate defined, setting pretrain_learning_rate to default 0.01 per iteration", 
#                                                                       arg_type='float_comma_string')
#            if len(self.pretrain_learning_rate) != sum(self.pretrain_iterations):
#                print "pretraining learning rate should have ", sum(self.pretrain_iterations), " learning rate iterations but only has ", len(self.pretrain_learning_rate), "... Exiting now"
#                sys.exit()
#            self.pretrain_batch_size = self.default_variable_define(config_dictionary, 'pretrain_batch_size', default_value=256, arg_type='int')
                    
        #backprop configuration
        self.do_backprop = self.default_variable_define(config_dictionary, 'do_backprop', default_value=True, arg_type='boolean')
        if self.do_backprop:
            if not hasattr(self, 'labels'):
                print "No labels found... cannot do backprop... Exiting now"
                sys.exit()
            self.backprop_method = self.default_variable_define(config_dictionary, 'backprop_method', default_value='steepest_descent', 
                                                                acceptable_values=['steepest_descent', 'conjugate_gradient', 'krylov_subspace', 'truncated_newton'])
            self.backprop_batch_size = self.default_variable_define(config_dictionary, 'backprop_batch_size', default_value=2048, arg_type='int')
            self.l2_regularization_const = self.default_variable_define(config_dictionary, 'l2_regularization_const', arg_type='float', default_value=0.0, exit_if_no_default=False)
            
            if self.backprop_method == 'steepest_descent':
                self.steepest_learning_rate = self.default_variable_define(config_dictionary, 'steepest_learning_rate', default_value=[0.008, 0.004, 0.002, 0.001], arg_type='float_comma_string')
            else:
                raise ValueError("%s backprop method not yet implemented" % self.backprop_method)
#                self.num_epochs = self.default_variable_define(config_dictionary, 'num_epochs', default_value=20, arg_type='int')
#                if self.backprop_method == 'conjugate_gradient':
#                    self.num_line_searches = self.default_variable_define(config_dictionary, 'num_line_searches', default_value=20, arg_type='int')
#                    self.conjugate_max_iterations = self.default_variable_define(config_dictionary, 'conjugate_max_iterations', default_value=3, 
#                                                                                 arg_type='int')
#                    self.conjugate_const_type = self.default_variable_define(config_dictionary, 'conjugate_const_type', arg_type='string', default_value='polak-ribiere', 
#                                                                             acceptable_values = ['polak-ribiere', 'polak-ribiere+', 'hestenes-stiefel', 'fletcher-reeves'])
#                    self.armijo_const = self.default_variable_define(config_dictionary, 'armijo_const', arg_type='float', default_value=0.1)
#                    self.wolfe_const = self.default_variable_define(config_dictionary, 'wolfe_const', arg_type='float', default_value=0.2)
#                elif self.backprop_method == 'krylov_subspace':
#                    self.num_line_searches = self.default_variable_define(config_dictionary, 'num_line_searches', default_value=20, arg_type='int')
#                    self.second_order_matrix = self.default_variable_define(config_dictionary, 'second_order_matrix', arg_type='string', default_value='gauss-newton', 
#                                                                            acceptable_values=['gauss-newton', 'hessian', 'fisher'])
#                    self.krylov_num_directions = self.default_variable_define(config_dictionary, 'krylov_num_directions', arg_type='int', default_value=20, 
#                                                                              acceptable_values=range(2,2000))
#                    self.krylov_num_batch_splits = self.default_variable_define(config_dictionary, 'krylov_num_batch_splits', arg_type='int', default_value=self.krylov_num_directions, 
#                                                                                acceptable_values=range(2,2000))
#                    self.krylov_num_bfgs_epochs = self.default_variable_define(config_dictionary, 'krylov_num_bfgs_epochs', arg_type='int', default_value=self.krylov_num_directions)
#                    self.krylov_use_hessian_preconditioner = self.default_variable_define(config_dictionary, 'krylov_use_hessian_preconditioner', arg_type='boolean', default_value=True)
#                    if self.krylov_use_hessian_preconditioner:
#                        self.krylov_eigenvalue_floor_const = self.default_variable_define(config_dictionary, 'krylov_eigenvalue_floor_const', arg_type='float', default_value=1E-4)
#                    self.use_fisher_preconditioner = self.default_variable_define(config_dictionary, 'use_fisher_preconditioner', arg_type='boolean', default_value=False)
#                    if self.use_fisher_preconditioner:
#                        self.fisher_preconditioner_floor_val = self.default_variable_define(config_dictionary, 'fisher_preconditioner_floor_val', arg_type='float', default_value=1E-4)
#                    self.armijo_const = self.default_variable_define(config_dictionary, 'armijo_const', arg_type='float', default_value=0.0001)
#                    self.wolfe_const = self.default_variable_define(config_dictionary, 'wolfe_const', arg_type='float', default_value=0.9)
#                elif self.backprop_method == 'truncated_newton':
#                    self.second_order_matrix = self.default_variable_define(config_dictionary, 'second_order_matrix', arg_type='string', default_value='gauss-newton', 
#                                                                            acceptable_values=['gauss-newton', 'hessian'])
#                    self.use_fisher_preconditioner = self.default_variable_define(config_dictionary, 'use_fisher_preconditioner', arg_type='boolean', default_value=False)
#                    if self.use_fisher_preconditioner:
#                        self.fisher_preconditioner_floor_val = self.default_variable_define(config_dictionary, 'fisher_preconditioner_floor_val', arg_type='float', default_value=1E-4)
#                    self.truncated_newton_num_cg_epochs = self.default_variable_define(config_dictionary, 'truncated_newton_num_cg_epochs', arg_type='int', default_value=20)
#                    self.truncated_newton_init_damping_factor = self.default_variable_define(config_dictionary, 'truncated_newton_init_damping_factor', arg_type='float', default_value=0.1)
        self.dump_config_vals()
    def backprop_steepest_descent(self): #need to test regularization
        self.memory_management()
        print "starting backprop using steepest descent"
        print "Number of previous tokens are", self.model.num_previous_tokens
        
        cross_entropy, num_correct, num_examples, loss = self.calculate_classification_statistics(self.features, self.labels, self.model)
        print "cross-entropy before steepest descent is", cross_entropy
        if self.l2_regularization_const > 0.0:
            print "regularized loss is", loss
        print "number correctly classified is", num_correct, "of", num_examples

        for epoch_num in range(len(self.steepest_learning_rate)):
            print "At epoch", epoch_num+1, "of", len(self.steepest_learning_rate), "with learning rate", self.steepest_learning_rate[epoch_num]
            batch_index = 0
            end_index = 0

            while end_index < self.num_training_examples: #run through the batches
                per_done = min(float(batch_index)/self.num_training_examples*100, 100.0)
                sys.stdout.write("%.1f%% done\n" % per_done), sys.stdout.flush()
                end_index = min(batch_index+self.backprop_batch_size,self.num_training_examples)
                batch_size = end_index - batch_index
                batch_inputs = self.features[batch_index:end_index]
                word_vectors, hiddens, outputs = self.forward_first_order_methods(batch_inputs, self.model)
                #calculating negative gradient of log softmax
                
                hidden_output_weight_vec = -outputs #batchsize x n_outputs
#                hidden_output_weight_vec[np.arange(batch_size), self.labels[batch_index:end_index].astype(int)] += 1
                for label_index in range(batch_index,end_index):
                    data_index = label_index - batch_index
                    hidden_output_weight_vec[data_index, int(self.labels[label_index])] += 1 #the int is to enforce proper indexing
                #averaging batches 
                self.model.weights['output'] += (self.steepest_learning_rate[epoch_num] / batch_size) * (gnp.dot(hiddens.T, hidden_output_weight_vec) - self.l2_regularization_const * self.model.weights['output'])
                self.model.bias['output'][0] += (self.steepest_learning_rate[epoch_num] / batch_size) * (gnp.sum(hidden_output_weight_vec, axis=0) - self.l2_regularization_const * self.model.bias['output'][0])
                gnp.free_reuse_cache(False)
                #I don't use calculate_gradient because structure allows me to store only one layer of weights
                
                projection_hidden_weight_vec = gnp.dot(hidden_output_weight_vec, self.model.weights['output'].T) * hiddens * (1-hiddens) #n_hid x n_out * (batchsize x n_out), do the biases get involved in this calculation???
                num_hiddens_per_token = self.model.get_architecture()[0]
                del hidden_output_weight_vec
                gnp.free_reuse_cache(True)
                self.model.weights['prehidden'] += self.steepest_learning_rate[epoch_num] / batch_size * (gnp.dot(word_vectors.T,projection_hidden_weight_vec) - self.l2_regularization_const * self.model.weights['prehidden'])
                self.model.bias['prehidden'][0] += self.steepest_learning_rate[epoch_num] / batch_size * (gnp.sum(projection_hidden_weight_vec,axis=0) - self.l2_regularization_const * self.model.bias['prehidden'][0])
                projection_layer_weight_vec = gnp.dot(projection_hidden_weight_vec, self.model.weights['prehidden'].T)
                del projection_hidden_weight_vec
                gnp.free_reuse_cache(True)

                for token_num in range(1, self.model.num_previous_tokens+1):
                    start_dim = (token_num - 1) * num_hiddens_per_token
                    end_dim = token_num * num_hiddens_per_token
                    token_inputs = batch_inputs[:,token_num-1]
#                   weight_vec_per_token = projection_hidden_weight_vec[:,start_dim:end_dim] 
#                   gnp.free_reuse_cache(False)
#                   self.model.weights[token_num] += self.steepest_learning_rate[epoch_num] / batch_size * (gnp.dot(word_vectors[token_num].T,weight_vec_per_token) - self.l2_regularization_const * self.model.weights[token_num])
#                    self.model.bias[token_num][0] += self.steepest_learning_rate[epoch_num] / batch_size * (gnp.sum(hidden_output_weight_vec,axis=0) - self.l2_regularization_const * self.model.bias[token_num][0])
                    projection_layer_weight_vec_per_token = projection_layer_weight_vec[:, start_dim:end_dim]
                    projection_layer_update = gnp.zeros(self.model.weights['projection'].shape)
#		            projection_layer_update[token_inputs,:] += projection_layer_weight_vec_per_token
                    for sent_index, token_input in enumerate(token_inputs):
                        projection_layer_update[token_input] += projection_layer_weight_vec_per_token[sent_index]
                        self.model.weights['projection'] += self.steepest_learning_rate[epoch_num] / batch_size * (projection_layer_update - self.l2_regularization_const * self.model.weights['projection'])
                        gnp.free_reuse_cache(False)
                    
                batch_index += self.backprop_batch_size
            sys.stdout.write("\r100.0% done \r"), sys.stdout.flush()
            
            cross_entropy, num_correct, num_examples, loss = self.calculate_classification_statistics(self.features, self.labels, self.model)
            print "cross-entropy at the end of the epoch is", cross_entropy
            if self.l2_regularization_const > 0.0:
                print "regularized loss is", loss
            print "number correctly classified is", num_correct, "of", num_examples
            
            if self.save_each_epoch:
                self.model.write_weights(''.join([self.output_name, '_epoch_', str(epoch_num+1)]))
    def calculate_classification_statistics(self, features, labels, model=None):
        if model == None:
            model = self.model
        
        excluded_keys = {'bias': ['0'], 'weights': []}
        
        if self.do_backprop == False:
            classification_batch_size = 4096 
        else:
            classification_batch_size = max(self.backprop_batch_size, 4096)
        
        batch_index = 0
        end_index = 0
        cross_entropy = 0.0
        num_correct = 0
        num_examples = features.shape[0]
        while end_index < num_examples: #run through the batches
            end_index = min(batch_index+classification_batch_size, num_examples)
            output = self.forward_pass(features[batch_index:end_index], verbose=False, model=model)
            cross_entropy += self.calculate_cross_entropy(output, labels[batch_index:end_index])
            
            #don't use calculate_classification_accuracy() because of possible rounding error
            prediction = output.argmax(axis=1).reshape(labels[batch_index:end_index].shape)
            num_correct += np.sum(prediction == labels[batch_index:end_index])
            batch_index += classification_batch_size
        
        loss = cross_entropy
        if self.l2_regularization_const > 0.0:
            loss += (model.norm(excluded_keys) ** 2) * self.l2_regularization_const
        return cross_entropy, num_correct, num_examples, loss
    def forward_first_order_methods(self, inputs, verbose=True, model=None): #completed
        # forward pass each layer starting with feature level
        # inputs are size n_batch x n_prev_tokens
        if model == None:
            model = self.model 
        
        word_vectors = self.forward_layer(inputs, model, layer_type='projection')
        hiddens = self.forward_layer(word_vectors, model, layer_type='prehidden')
        outputs = self.forward_layer(hiddens, model, layer_type='output')
        gnp.free_reuse_cache(False)
        return word_vectors, hiddens, outputs
    def memory_management(self):
        print "WARNING: Memory Management Not Implemented yet"

if __name__ == '__main__':
    script_name, config_filename = sys.argv
    print "Opening config file: %s" % config_filename
#    script_name = sys.argv[0]
#    parser = init_arg_parser()
#    config_dictionary = vars(parser.parse_args())
    
#    if config_dictionary['config_file'] != None :
#        config_filename = config_dictionary['config_file']
#        print "Since", config_filename, "is specified, ignoring other arguments"
    try:
        config_file=open(config_filename)
    except IOError:
        print "Could open file", config_filename, ". Usage is ", script_name, "<config file>... Exiting Now"
        sys.exit()
        
#    del config_dictionary
        
        #read lines into a configuration dictionary, skipping lines that begin with #
    config_dictionary = dict([line.replace(" ", "").strip(' \n\t').split('=') for line in config_file 
                              if not line.replace(" ", "").strip(' \n\t').startswith('#') and '=' in line])
    config_file.close()
#    else:
        #remove empty keys
#        config_dictionary = dict([(arg,value) for arg,value in config_dictionary.items() if value != None])

    try:
        mode=config_dictionary['mode']
    except KeyError:
        print 'No mode found, must be train or test... Exiting now'
        sys.exit()
    else:
        if (mode != 'train') and (mode != 'test'):
            print "Mode", mode, "not understood. Should be either train or test... Exiting now"
            sys.exit()
    
    if mode == 'test':
        test_object = NNLM_Tester(config_dictionary)
    else: #mode ='train'
        train_object = NNLM_Trainer(config_dictionary)
        train_object.backprop_steepest_descent()
        
    print "Finished without Runtime Error!" 