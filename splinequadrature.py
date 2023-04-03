#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 17:13:47 2023

This file defines several utility functions to find quadrature rules for
different spline spaces. Everything is implemented in JAX.

@author: T. Teijeiro
"""

from collections import namedtuple as nt
import jax
import jax.numpy as jnp
import numpy as np
import optax

###############################################
# Definition of the basis functions
###############################################

def splinesbfns(degree, partition, cont):
    """
    Creates the B-splines basis functions of a given degree and partition.

    Implementation based initially on: https://github.com/johntfoster/bspline
    Code adapted and optimized to work in JAX.

    Returns a function that evaluates all the basis functions on a vector.
    """
    #Multiplicity of the initial and final knots according to the degree.
    #Multiplicity of interior knots according to degree and continuity.
    knots = jnp.concatenate((jnp.repeat(partition[0], degree+1),
                             jnp.repeat(partition[1:-1], degree-cont),
                             jnp.repeat(partition[-1], degree+1)))

    def basis(degree):
        """
        Definition of the univariate basis functions, using Cox-de-Boor recursion
        """

        def B0(x):
            """Base case, degree 0"""
            return jnp.logical_and(knots[:-1] <= x,
                                   x < knots[1:]).astype(jnp.float64)

        def B1plus(x):
            """Higher-order cases, defined recursively."""
            basis_m1 = basis(degree-1)(x)
            #First and second term (left and right) numerator and denominator
            lnum = x - knots[:-degree]
            ldenom = knots[degree:]-knots[:-degree]
            rnum = knots[degree+1:] - x
            rdenom = knots[degree+1:] - knots[1:-degree]
            #Calculation of both terms avoiding divisions by zero
            #Double `where` is required to avoid nans both in the result and
            #**during** the calculation.
            safe_denom = jnp.where(ldenom != 0.0, ldenom, 1.0)
            lprod = jnp.where(ldenom != 0.0, lnum/safe_denom, 0.0)
            safe_denom = jnp.where(rdenom != 0.0, rdenom, 1.0)
            rprod = jnp.where(rdenom != 0.0, rnum/safe_denom, 0.0)
            return lprod[:-1] * basis_m1[:-1] + rprod * basis_m1[1:]

        return B0 if degree == 0 else B1plus

    #Vectorized version of the basis functions (output as column vectors)
    return jax.vmap(basis(degree), 0, 1)


#################################################
## Functions related to exact Gaussian Quadrature
#################################################

def gaussint(int_points=4):
    """
    Returns a function that allows the calculation of Gaussian quadrature of
    an arbitrary function on an arbitrary interval, using a predefined number
    of quadrature points.
    """
    # Values in double precision downloaded from
    # https://pomax.github.io/bezierinfo/legendre-gauss.html#n6
    X_COEFFS = {1: [0.0],
                2: [-0.5773502691896257, 0.5773502691896257],
                3: [0.0, -0.7745966692414834, 0.7745966692414834],
                4: [-0.3399810435848563, 0.3399810435848563,
                    -0.8611363115940526, 0.8611363115940526],
                5: [0.0, -0.5384693101056831, 0.5384693101056831,
                         -0.9061798459386640, 0.9061798459386640],
                6: [-0.6612093864662645, 0.6612093864662645,
                    -0.2386191860831969, 0.2386191860831969,
                    -0.9324695142031521, 0.9324695142031521],
                7: [0.0, -0.4058451513773972, 0.4058451513773972,
                         -0.7415311855993945, 0.7415311855993945,
                         -0.9491079123427585, 0.9491079123427585],
                8: [-0.1834346424956498, 0.1834346424956498,
                    -0.5255324099163290, 0.5255324099163290,
                    -0.7966664774136267, 0.7966664774136267,
                    -0.9602898564975363, 0.9602898564975363],
                9: [0.0, -0.8360311073266358, 0.8360311073266358,
                         -0.9681602395076261, 0.9681602395076261,
                         -0.3242534234038089, 0.3242534234038089,
                         -0.6133714327005904, 0.6133714327005904],
               10: [-0.1488743389816312, 0.1488743389816312,
                    -0.4333953941292472, 0.4333953941292472,
                    -0.6794095682990244, 0.6794095682990244,
                    -0.8650633666889845, 0.8650633666889845,
                    -0.9739065285171717, 0.9739065285171717],
               11: [0.0, -0.2695431559523450, 0.2695431559523450,
                         -0.5190961292068118, 0.5190961292068118,
                         -0.7301520055740494, 0.7301520055740494,
                         -0.8870625997680953, 0.8870625997680953,
                         -0.9782286581460570, 0.9782286581460570],
               12: [-0.1252334085114689, 0.1252334085114689,
                    -0.3678314989981802, 0.3678314989981802,
                    -0.5873179542866175, 0.5873179542866175,
                    -0.7699026741943047, 0.7699026741943047,
                    -0.9041172563704749, 0.9041172563704749,
                    -0.9815606342467192, 0.9815606342467192],
               13: [0.0, -0.2304583159551348, 0.2304583159551348,
                         -0.4484927510364469, 0.4484927510364469,
                         -0.6423493394403402, 0.6423493394403402,
                         -0.8015780907333099, 0.8015780907333099,
                         -0.9175983992229779, 0.9175983992229779,
                         -0.9841830547185881, 0.9841830547185881],
               14: [-0.1080549487073437, 0.1080549487073437,
                    -0.3191123689278897, 0.3191123689278897,
                    -0.5152486363581541, 0.5152486363581541,
                    -0.6872929048116855, 0.6872929048116855,
                    -0.8272013150697650, 0.8272013150697650,
                    -0.9284348836635735, 0.9284348836635735,
                    -0.9862838086968123, 0.9862838086968123],
               15: [0.0, -0.2011940939974345, 0.2011940939974345,
                         -0.3941513470775634, 0.3941513470775634,
                         -0.5709721726085388, 0.5709721726085388,
                         -0.7244177313601701, 0.7244177313601701,
                         -0.8482065834104272, 0.8482065834104272,
                         -0.9372733924007060, 0.9372733924007060,
                         -0.9879925180204854, 0.9879925180204854],
               16: [-0.0950125098376374, 0.0950125098376374,
                    -0.2816035507792589, 0.2816035507792589,
                    -0.4580167776572274, 0.4580167776572274,
                    -0.6178762444026438, 0.6178762444026438,
                    -0.7554044083550030, 0.7554044083550030,
                    -0.8656312023878318, 0.8656312023878318,
                    -0.9445750230732326, 0.9445750230732326,
                    -0.9894009349916499, 0.9894009349916499],
               17: [0.0, -0.1784841814958479, 0.1784841814958479,
                         -0.3512317634538763, 0.3512317634538763,
                         -0.5126905370864769, 0.5126905370864769,
                         -0.6576711592166907, 0.6576711592166907,
                         -0.7815140038968014, 0.7815140038968014,
                         -0.8802391537269859, 0.8802391537269859,
                         -0.9506755217687678, 0.9506755217687678,
                         -0.9905754753144174, 0.9905754753144174]}


    W_COEFFS = {1: [2.0],
                2: [1.0, 1.0],
                3: [0.8888888888888888, 0.5555555555555556, 0.5555555555555556],
                4: [0.6521451548625461, 0.6521451548625461,
                    0.3478548451374538, 0.3478548451374538],
                5: [0.5688888888888889, 0.4786286704993665, 0.4786286704993665,
                    0.2369268850561891, 0.2369268850561891],
                6: [0.3607615730481386, 0.3607615730481386,
                    0.4679139345726910, 0.4679139345726910,
                    0.1713244923791704, 0.1713244923791704],
                7: [0.4179591836734694, 0.3818300505051189, 0.3818300505051189,
                    0.2797053914892766, 0.2797053914892766,
                    0.1294849661688697, 0.1294849661688697],
                8: [0.3626837833783620, 0.3626837833783620,
                    0.3137066458778873, 0.3137066458778873,
                    0.2223810344533745, 0.2223810344533745,
                    0.1012285362903763, 0.1012285362903763],
                9: [0.3302393550012598, 0.1806481606948574, 0.1806481606948574,
                    0.0812743883615744, 0.0812743883615744,
                    0.3123470770400029, 0.3123470770400029,
                    0.2606106964029354, 0.2606106964029354],
               10: [0.2955242247147529, 0.2955242247147529,
                    0.2692667193099963, 0.2692667193099963,
                    0.2190863625159820, 0.2190863625159820,
                    0.1494513491505806, 0.1494513491505806,
                    0.0666713443086881, 0.0666713443086881],
               11: [0.2729250867779006, 0.2628045445102467, 0.2628045445102467,
                    0.2331937645919905, 0.2331937645919905,
                    0.1862902109277343, 0.1862902109277343,
                    0.1255803694649046, 0.1255803694649046,
                    0.0556685671161737, 0.0556685671161737],
               12: [0.2491470458134028, 0.2491470458134028,
                    0.2334925365383548, 0.2334925365383548,
                    0.2031674267230659, 0.2031674267230659,
                    0.1600783285433462, 0.1600783285433462,
                    0.1069393259953184, 0.1069393259953184,
                    0.0471753363865118, 0.0471753363865118],
               13: [0.2325515532308739, 0.2262831802628972, 0.2262831802628972,
                    0.2078160475368885, 0.2078160475368885,
                    0.1781459807619457, 0.1781459807619457,
                    0.1388735102197872, 0.1388735102197872,
                    0.0921214998377285, 0.0921214998377285,
                    0.0404840047653159, 0.0404840047653159],
               14: [0.2152638534631578, 0.2152638534631578,
                    0.2051984637212956, 0.2051984637212956,
                    0.1855383974779378, 0.1855383974779378,
                    0.1572031671581935, 0.1572031671581935,
                    0.1215185706879032, 0.1215185706879032,
                    0.0801580871597602, 0.0801580871597602,
                    0.0351194603317519, 0.0351194603317519],
               15: [0.2025782419255613, 0.1984314853271116, 0.1984314853271116,
                    0.1861610000155622, 0.1861610000155622,
                    0.1662692058169939, 0.1662692058169939,
                    0.1395706779261543, 0.1395706779261543,
                    0.1071592204671719, 0.1071592204671719,
                    0.0703660474881081, 0.0703660474881081,
                    0.0307532419961173, 0.0307532419961173],
               16: [0.1894506104550685, 0.1894506104550685,
                    0.1826034150449236, 0.1826034150449236,
                    0.1691565193950025, 0.1691565193950025,
                    0.1495959888165767, 0.1495959888165767,
                    0.1246289712555339, 0.1246289712555339,
                    0.0951585116824928, 0.0951585116824928,
                    0.0622535239386479, 0.0622535239386479,
                    0.0271524594117541, 0.0271524594117541],
               17: [0.1794464703562065, 0.1765627053669926, 0.1765627053669926,
                    0.1680041021564500, 0.1680041021564500,
                    0.1540457610768103, 0.1540457610768103,
                    0.1351363684685255, 0.1351363684685255,
                    0.1118838471934040, 0.1118838471934040,
                    0.0850361483171792, 0.0850361483171792,
                    0.0554595293739872, 0.0554595293739872,
                    0.0241483028685479, 0.0241483028685479]}

    GAUSS_X0 = jnp.array(X_COEFFS[int_points])
    GAUSS_W0 = jnp.array(W_COEFFS[int_points])

    def gauss(f, a, b):
        """
        Function that performs the integration of a function `f` as the dot 
        product of the value of `f` at the points and the weights, 
        adjusted to a specific interval [a, b].
        """
        X = (b-a)/2.0*GAUSS_X0+(a+b)/2.0
        W = GAUSS_W0*(b-a)/2.0
        return jnp.dot(f(X), W)

    return gauss

def gaussint_partition(f, partition, n_points=4):
    """
    Calculates the gaussian quadrature for a specific function given a partition
    of the domain, by summing the Gauss quadrature on each element of the
    partition. The number of points in the quadrature can be specified.

    Parameters
    ----------
    f: Function to integrate (or a basis function).
    partition: Array with the partition.
    n_points: Number of points to use in the quadrature rule. It depends on the
              maximum degree of the function to give an exact result. A rule
              with n quadrature points can integrate up to a degree 2*n-1.

    Returns
    -------
    Array with the exact integral value for each basis function.

    """
    ints = jnp.stack([gaussint(n_points)(f, partition[i], partition[i+1])
                      for i in range(len(partition)-1)])
    return jnp.sum(ints, axis=0)

###############################################
# Gram matrix definition
###############################################

def l2norm_mat(basis, degree, partition):
    """
    Given a basis of functions of a given degree and a partition, outputs the
    bilinear form matrix on the basis functions and at each location. This is
    the matrix named `M` in the paper.
    """
    def B(x):
        b = basis(x)
        return jnp.einsum("Ii,Ji->IJi", b, b)
    T = gaussint_partition(B, partition, degree+1)
    return T

###############################################
# Parameter initialization
###############################################

def initialize_uniform(degree, cont, n, x1, w1):
    """
    Generates a starting point for the optimization process for a space with a
    given degree, continuity and number of elements (`n`). This starting point
    is generated from an optimal quadrature rule of a space with the same
    degree, continuity but `n-1` elements. This optimal rule is (`x1`, `w1`).
    """
    #Optimal number of points in the whole solution
    npts = int(np.ceil((degree+1+(degree-cont)*(n-1))/2.))
    #Number of points to learn due to symmetry
    nhalf = int(np.ceil(npts/2.))
    #Special case for continuity = 0. Simpler initialization
    if cont == 0:
        x = jnp.arange(1, npts+1)/npts-1/(2*npts)
        w = jnp.ones(npts)/npts
    else:
        x = x1*(n-1)/n
        w = w1*(n-1)/n
    #If the optimal number of points is odd, we need to learn one less x value.
    x = x[:npts//2]
    w = w[:nhalf]
    w = w/(2*jnp.sum(w))
    return x, w

def initialize_nonuniform(partition, x1, w1):
    """
    Generates the starting point for the optimization process for a space with
    non-uniform elements, based on the optimal points and weights for a 
    space with the same degree, continuity and number of elements, but in which
    elements are uniform. This optimal rule is (`x1`, `w1`).
    """
    n = len(partition)-1
    #Uniform partition
    upart = np.linspace(0, 1, n+1)
    #Now the points corresponding to each element are displaced according to
    #the scaling of the element.
    indices = np.searchsorted(upart, x1) - 1
    x, w = [], []
    for elem in range(n):
        #Linear translation and scaling of each element
        scale = (partition[elem+1]-partition[elem])/(upart[elem+1]-upart[elem])
        x.append(partition[elem]+(x1[indices==elem]-upart[elem])*scale)
        w.append(w1[indices==elem]*scale)
    return jnp.concatenate(x), jnp.concatenate(w)

###############################################
# Parameter optimization (rule discovery)
###############################################

Constants = nt('Constants', ['basis', 'degree', 'cont', 'partition'])
Constants.__doc__ = '''\
    Constant values configuring the integration problem.

    basis - Basis of functions
    cont - Continuity of functions between consecutive elements
    degree - Degree of polynomial space
    partition - Internal knots used for the partition (without repetitions)
'''

def _clip_params(max_value=1.0):
    """
    Single gradient transformation to ensure that parameters never grow over a 
    specific value. To be applied in the global optimizer.

    Parameters
    ----------
    max_value : Value to clip the parameters. The default is 1.0.

    Returns
    -------
    GradientTransformation object that can be chained with any other optimizer.
    """
    return optax.stateless_with_tree_map(
        lambda u, p: jnp.where((p + u) > max_value, max_value-p, u))

def get_optimizer(constants):
    """
    Creates an optimizer for the search process. It is an instance of 
    `optax.yogi` with a learning rate adapted to the number of training 
    parameters, and with additional constraints on the parameter values.
    """
    q = int(np.ceil((constants.degree + 1 + (constants.degree-constants.cont)
                     *(len(constants.partition)-1))/2.))
    return optax.chain(optax.yogi(learning_rate = 1e-2/(q*np.log10(q))),
                       optax.keep_params_nonnegative(),
                       optax.masked(_clip_params(1.0), {'x':True, 'w':False}))

def fit_quadrature_rule(constants, params, optimizer, epochs=10000, early_stop=1e-25):
    """
    Main optimization function.

    Parameters
    ----------
    constants : Constant parameters for the training. Expected to be an instance of the
        'Constants' namedtuple.
    params : Dictionary with the parameters to optimize.
    optimizer : Optimizer to be used in the training process.
    epochs : Number of epochs. The default is 10000.
    early_stop: Threshold used to stop the training when the error goes below
                this limit.

    Returns
    -------
    Final value of the parameters, loss obtained with those parameters, and
    number of optimization epochs required to obtain the loss.
    """

    opt_state = optimizer.init(params)
    is_unif = np.allclose(np.diff(constants.partition, 2), 0)
    #Exact integral value for each basis function, with the minimum number of
    #points required by the Gauss rule.
    reference = gaussint_partition(constants.basis, constants.partition, 
                                   np.ceil((constants.degree+1)/2.))
    #Inverse of the Gram matrix required in the loss function.
    T = l2norm_mat(constants.basis, constants.degree, constants.partition)
    Tm1 = jnp.linalg.inv(T)

    #Utility functions to get the full 'x' and 'w' arrays from the simplified
    #representation due to symmetry if the elements are uniform.
    def get_x_w_odd(parameters):
        x = parameters['x']
        x = jnp.concatenate((jnp.append(x, 0.5), 1-x))
        w = parameters['w']
        return (x, jnp.concatenate((w, w[:-1])))

    def get_x_w_even(parameters):
        x = parameters['x']
        w = parameters['w']
        return (jnp.concatenate((x, 1-x)), jnp.concatenate((w, w)))
    #Dummy function with the same interface to get the 'x' and 'w' arrays, but 
    #for non-uniform partitions.
    def get_x_w_nunif(parameters):
        return parameters['x'], parameters['w']

    get_x_w = (get_x_w_nunif if not is_unif 
               else (get_x_w_even if len(params['w']) == len(params['x'])
                     else get_x_w_odd))

    def loss(parameters):
        """Loss function to be minimized"""
        x, w = get_x_w(parameters)
        b = constants.basis(x)
        #Weights need to be scaled to ensure a total value of 1.0
        w = w/jnp.sum(w)
        #Error on integrating each of the basis functions
        err = reference - jnp.einsum("Ii,i->I", b, w)
        #Maximum estimation of the error using the inverse of the Gram matrix
        loss = jnp.einsum("IJ,I,J", Tm1, err, err)
        return loss

    loss_value_grad = jax.jit(jax.value_and_grad(loss))

    @jax.jit
    def gradient_step(grads, opt_state, params):
        """
        Performs a single step in the optimization process following
        gradient descent.
        """
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    #Optimization loop
    i = 0
    loss_value, grads = loss_value_grad(params)
    while i < epochs and loss_value >= early_stop:
        params, opt_state = gradient_step(grads, opt_state, params)
        loss_value, grads = loss_value_grad(params)
        i += 1

    #Points and weights are returned sorted and normalized
    final_x, final_w = get_x_w(params)
    order = jnp.argsort(final_x)
    params['x'] = final_x[order]
    params['w'] = final_w[order]/jnp.sum(final_w)
    return params, loss_value, i+1
