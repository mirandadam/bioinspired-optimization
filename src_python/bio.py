#!/usr/bin/python3
# -*- coding: utf8 -*-

# Copyright (C) 2015 Daniel Araujo Miranda
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
#@author: Daniel Araujo Miranda

import numpy as np
import fitnessfunctions  # needed for testing.

# TODO: cite the references to the algorithms.
# TODO: have somebody else review this code.


class Optimizer:
    """
    ## These variables are commented out on purpose to generate errors
    ##  in case they are used before being initialized.
    #problem variables:
    costfunction=None      #the cost function to evaluate
    dimensions=None        #number of dimensions
    maxiter=500            #maximum number of iterations
    target_cost=None       #stopping criterion for cost
    lb=None                #range domain for fitness function - lower bound for each dimension
    ub=None                #range domain for fitness function - upper bound for each dimension

    #state variables:
    _X=None       #current particle solutions
    _Y=None       #current particle cost
    _bestidx=None #index of best particle in X
    _bestx=None   #solution of best particle in X
    _besty=None   #cost of best particle in X
    _iter=None    #current iteration
    """

    def run(self):
        """
          Iterate the algorithm until a stop condition is met.
          Returns the final cost and the final solution found.
        """
        s = self
        while(s._iter < s.maxiter and
              (s.target_cost is None or s._ybest > s.target_cost)):
            i = s._iter
            s.iterate_one()
            # protect against s.iterate_one incrementing the s._iter counter:
            if(s._iter == i):
                s._iter += 1
        return (s._besty, s._bestx)

    def run_with_history(self):
        """
          Iterate the algorithm until a stop condition is met.

          Returns the cost history and the solution history of each iteration in
           chronological order
        """
        s = self
        x_history = []
        y_history = []
        while(s._iter < s.maxiter and
              (s.target_cost is None or s._ybest > s.target_cost)):
            i = s._iter
            s.iterate_one()
            x_history.append(s._bestx)
            y_history.append(s._besty)
            # protect against s.iterate_one incrementing the s._iter counter:
            if(s._iter == i):
                s._iter += 1
        return (y_history, x_history)


class PSO(Optimizer):
    name = 'PSO'
    description = 'Particle Swarm Optimization (PSO) algorithm'
    # algorithm tuning:
    n = 10  # number of particles
    w0 = 0.9  # initial inertia coefficient (weight)
    wf = 0.1  # final inertia coefficient (weight)
    c1 = 2  # cognitive coefficient
    c2 = 2  # social coefficient
    max_v = 5  # maximum velocity
    ini_v = max_v / 10  # max_v/10 #initial velocity

    # state variables:
    _iter = 0

    def __init__(self, costfunction, dimensions, lb, ub, maxiter=500, target_cost=None,
                 n=10, w0=0.9, wf=0.1, c1=2, c2=2, max_v=5, ini_v=5 / 10):
        """
        The cost function has to take arrays with (m,n) shape as inputs, where
         m is the number of particles and n is the number of dimensions.
        lb is the lower bound for each dimension
        ub is the upper bound for each dimension
        """
        s = self
        # TODO: do some serious input checking here.

        # problem parameters:
        s.costfunction = costfunction
        s.dimensions = dimensions
        s.lb = lb.copy()
        s.ub = ub.copy()
        s.maxiter = maxiter
        s.target_cost = target_cost

        # algorithm tuning:
        s.n = n
        s.w0 = w0
        s.wf = wf
        s.c1 = c1
        s.c2 = c2
        s.max_v = max_v
        s.ini_v = ini_v

        # initial conditions:
        s._X = np.random.random((n, dimensions)) * (ub - lb) + lb  # current particle solutions
        s._Y = s.costfunction(s._X)                  # current particle cost
        s._V = np.ones((n, dimensions)) * ini_v               # current particle speeds

        s._Xmemory = s._X.copy()              # memory of best individual solution
        s._Ymemory = s._Y.copy()              # memory of best individual fitness
        # print(s._X.shape,s._X)
        s._bestidx = np.argmin(s._Ymemory)         # index of best particle in Xmemory
        s._bestx = s._Xmemory[s._bestidx].copy()  # solution of best particle in Xmemory
        s._besty = s._Ymemory[s._bestidx]        # cost of best particle in Xmemory

        s._iter = 0

    def iterate_one(self):
        s = self
        # calculating inertia weight:
        w = s.w0 + s._iter * (s.wf - s.w0) / s.maxiter

        # particle movement:
        r1 = np.random.random((s.n, s.dimensions))
        r2 = np.random.random((s.n, s.dimensions))
        s._V = w * s._V + s.c1 * r1 * (s._Xmemory - s._X) + s.c2 * r2 * (s._bestx - s._X)

        # applying speed limit:
        vnorm = ((s._V**2).sum(axis=1))**0.5  # norm of the speed
        aux = np.where(vnorm > s.max_v)  # particles with velocity greater than expected
        s._V[aux] = s._V[aux] * s.max_v / (vnorm[aux].reshape((-1, 1)))  # clipping the speed to the maximum speed

        # update solutions:
        s._X = s._X + s._V

        # clipping the search space
        s._X = np.minimum(s.ub, s._X)
        s._X = np.maximum(s.lb, s._X)

        # fitness value calculation:
        s._Y = s.costfunction(s._X)  # current particle cost

        # update memories:
        aux = np.where(s._Y < s._Ymemory)
        s._Xmemory[aux] = s._X[aux].copy()           # memory of best individual solution
        s._Ymemory[aux] = s._Y[aux].copy()           # memory of best individual fitness
        s._bestidx = np.argmin(s._Ymemory)           # index of best particle in Xmemory
        s._bestx = s._Xmemory[s._bestidx].copy()  # solution of best particle in Xmemory
        s._besty = s._Ymemory[s._bestidx]         # cost of best particle in Xmemory
        return


class ABC(Optimizer):
    name = 'ABC'
    description = 'Artificial Bee Colony (ABC) algorithm'

    nb = 24  # Number of bees (employed bees plus onlooker bees).
    nf = 12  # Number of food sources. Default is nb/2.
    abandon_threshold = 20  # Number of consecutive improvement trials that a food source undergoes before being abandoned.

    def __init__(self, costfunction, dimensions, lb, ub, maxiter=500, target_cost=None,
                 n=12, nb=24, abandon_threshold=20):
        # TODO: do input checking here.
        s = self

        # problem parameters:
        s.costfunction = costfunction
        s.dimensions = dimensions
        s.lb = lb.copy()
        s.ub = ub.copy()
        s.maxiter = maxiter
        s.target_cost = target_cost

        s.nb = nb                               # Number of bees (employed bees plus onlooker bees).
        s.nf = n                                # Number of food sources. Default is nb/2.
        s.abandon_threshold = abandon_threshold  # Number of consecutive improvement trials that a food source undergoes before being abandoned.

        # initial conditions:
        s._X = np.random.random((s.nf, dimensions)) * (ub - lb) + lb  # current food sources
        s._Y = s.costfunction(s._X)                         # current source cost
        s._trials = np.zeros(s.nf)                            # number of attempts to improve each solution

        # index of best particle (food source) - watch out!!! if scout bees destroy this solution, the index is no longer valid, even though the besty and bestx variables are.
        s._bestidx = np.argmin(s._Y)
        s._bestx = s._X[s._bestidx].copy()  # solution of best particle (food source) ever
        s._besty = s._Y[s._bestidx]        # cost of best particle (food source)

        s._iter = 0
        return

    def iterate_one(self):
        # TODO: cleanup this code. refactor. document. use references and terminology from the article
        s = self

        #### Employed bee phase ####
        # The parameter to be changed is determined randomly
        parameters_to_change = np.random.randint(0, s.dimensions, s.nf)
        # A randomly chosen solution is used in producing a mutant solution of the solution i
        aux = np.arange(s.nf, dtype='int')
        neighbours = (np.random.randint(1, s.nf, s.nf) + aux) % s.nf  # neighbour indices
        new_foods = s._X.copy()
        new_foods[(aux, parameters_to_change)] = s._X[(aux, parameters_to_change)] + (np.random.rand(s.nf) * 2 - 1) * (s._X[(aux, parameters_to_change)] - s._X[(neighbours, parameters_to_change)])
        del aux, parameters_to_change, neighbours
        # if generated parameter value is out of boundaries, it is shifted onto the boundaries
        new_foods = np.maximum(new_foods, s.lb)
        new_foods = np.minimum(new_foods, s.ub)
        # evaluate new solution
        new_food_costs = s.costfunction(new_foods)
        # a greedy selection is applied between the current solution i and its mutant
        aux = (new_food_costs < s._Y)  # mask to show where the cost has improved
        # If the mutant solution is better than the current solution i, replace the solution with the mutant and reset the trial counter of solution i
        s._X = aux.reshape((-1, 1)) * new_foods + (1 - aux.reshape((-1, 1))) * s._X
        s._Y = aux * new_food_costs + (1 - aux) * s._Y
        # increase trial count of solutions which not improve, zero the counter of the improved ones.
        s._trials = (1 - aux) * (s._trials + 1)
        del aux, new_food_costs, new_foods

        #### Calculate probabilities ####
        # A food source is chosen with the probability which is proportional to its quality
        # Different schemes can be used to calculate the probability values
        # For example prob(i)=fitness(i)/sum(fitness)
        # or in a way used in the metot below prob(i)=a*fitness(i)/max(fitness)+b
        # probability values are calculated by using fitness values and normalized by dividing maximum fitness value
        prob = (0.9 * s._Y / np.max(s._Y)) + 0.1  # the higher the cost, the higher the probability for change

        #### Onlooker bee phase ####
        # TODO: fix the number of onlooker bees. It is currently hardcoded to be the same number of food sources.
        aux = np.where(prob > np.random.rand(s.nf))[0]  # chosen food sources by probability
        n_aux = len(aux)  # number of randomly chosen food sources
        # The parameter to be changed is determined randomly
        parameters_to_change = np.random.randint(0, s.dimensions, n_aux)
        # A randomly chosen solution is used in producing a mutant solution of the solution i
        # Randomly selected solution must be different from the solution i
        neighbours = (np.random.randint(1, s.nf, n_aux) + aux) % s.nf
        new_foods = s._X.copy()
        new_foods[(aux, parameters_to_change)] = s._X[(aux, parameters_to_change)] + (np.random.rand(n_aux) * 2 - 1) * (s._X[(aux, parameters_to_change)] - s._X[(neighbours, parameters_to_change)])
        del aux, n_aux, parameters_to_change, neighbours
        # if generated parameter value is out of boundaries, it is shifted onto the boundaries
        new_foods = np.maximum(new_foods, s.lb)
        new_foods = np.minimum(new_foods, s.ub)
        # evaluate new solution
        new_food_costs = s.costfunction(new_foods)
        # a greedy selection is applied between the current solution i and its mutant
        aux = (new_food_costs < s._Y)  # mask to show where the cost has improved
        # If the mutant solution is better than the current solution i, replace the solution with the mutant and reset the trial counter of solution i
        s._X = aux.reshape((-1, 1)) * new_foods + (1 - aux.reshape((-1, 1))) * s._X
        s._Y = aux * new_food_costs + (1 - aux) * s._Y
        # increase trial count of solutions which not improve, zero the counter of the improved ones.
        s._trials = (1 - aux) * (s._trials + 1)
        del aux, new_food_costs, new_foods

        # The best food source is identified
        s._bestidx = np.argmin(s._Y)
        if(s._Y[s._bestidx] < s._besty):
            # record only the best food source ever, even though there may be no more bees on it
            # this may depart from the original algorithm.
            # TODO: check the paper!
            s._bestx = s._X[s._bestidx].copy()
            s._besty = s._Y[s._bestidx]

        #### Scout bee phase ####
        # determine the food sources whose trial counter exceeds the "abandon_threshold" value.
        # In Basic ABC, only one scout is allowed to occur in each cycle
        aux = np.argmax(s._trials)
        if(s._trials[aux] > s.abandon_threshold):
            new_food = np.random.rand(s.dimensions) * (s.ub - s.lb) + s.lb
            new_cost = s.costfunction(np.array([new_food]))[0]
            # Since the best solution cannot be improved upon, it will eventually
            # hit the maximum trial count and be abandoned.
            # in this algorithm, scout bees destroy the best solution.
            s._X[aux] = new_food.copy()
            s._Y[aux] = new_cost
            s._trials[aux] = 0
            # best food source is verified again, only against the scout:
            if(new_cost < s._besty):
                s._bestidx = aux
                s._bestx = s._X[s._bestidx].copy()
                s._besty = s._Y[s._bestidx]
        del aux

        # end of algorithm
        return


class DE(Optimizer):
    name = 'DE'
    description = 'Differential Evolution (DE) algorithm'
    # algorithm tuning:
    n = 10  # number of particles
    f = 0.5  # mutation factor
    c = 0.9  # crossover rate

    # state variables:
    _iter = 0

    def __init__(self, costfunction, dimensions, lb, ub, maxiter=500, target_cost=None,
                 n=50, f=0.5, c=0.9):
        """
        The cost function has to take arrays with (m,n) shape as inputs, where
         m is the number of particles and n is the number of dimensions.
        lb is the lower bound for each dimension
        ub is the upper bound for each dimension
        """
        s = self
        # TODO: do some serious input checking here.

        # problem parameters:
        s.costfunction = costfunction
        s.dimensions = dimensions
        s.lb = lb.copy()
        s.ub = ub.copy()
        s.maxiter = maxiter
        s.target_cost = target_cost

        # algorithm tuning:
        s.n = n
        s.f = f
        s.c = c

        # initial conditions:
        s._X = np.random.random((n, dimensions)) * (ub - lb) + lb  # current individuals
        s._Y = s.costfunction(s._X)                        # current cost

        s._bestidx = np.argmin(s._Y)             # index of best individual
        s._bestx = s._X[s._bestidx].copy()  # solution of best individual
        s._besty = s._Y[s._bestidx]        # cost of best individual

        s._iter = 0

    def _mutate(self, direction):
        """
          Selects neighbours and apply mutation.
        """
        # def mutate(X,F,direction=1):#div):
        #div_min = 0.25
        #div_max = 0.50
        # if(div<div_min):
        #  direction=-1
        # elif(div>div_max):
        #  direction=1

        s = self
        n = s.n
        assert(n > 3)  # only works with 4 or more individuals

        # getting random neighbours to permutate
        neighbours = np.array([np.random.permutation(n - 1)[:3] for i in range(n)]).transpose()
        neighbours = (neighbours + np.arange(n) + 1) % n
        n0 = np.arange(n)  # self
        n1 = neighbours[0, :]  # neighbours different from self
        n2 = neighbours[1, :]  # neighbours different from self and n1
        n3 = neighbours[2, :]  # neighbours different from self, n1 and n2
        '''
    #DEBUG:
    assert((n0!=n1).all())
    assert((n0!=n2).all())
    assert((n0!=n3).all())
    assert((n1!=n2).all())
    assert((n1!=n3).all())
    assert((n2!=n3).all())
    '''
        # mutation:
        m = s._X[n1] + direction * s.f * (s._X[n2] - s._X[n3])
        return m

    def iterate_one(self):
        s = self

        # Mutation:
        mutated_X = s._mutate(direction=1)
        # TODO: implement changing the mutation direction

        # If generated parameter value is out of boundaries, it is shifted onto the boundaries:
        mutated_X = np.minimum(s.ub, mutated_X)
        mutated_X = np.maximum(s.lb, mutated_X)

        # Crossover
        d = np.random.randint(s.dimensions, size=(s.n,))  # dimensions chosen for each individual
        r1 = np.random.rand(s.n, s.dimensions)  # calculating random variables
        aux = (r1 < s.c)  # dimensions to crossover based on probability
        aux[(np.arange(s.n), d)] = True  # choose one dimension of each individual to crossover regardless os probability
        crossover_X = aux * mutated_X + (1 - aux) * s._X  # take elements from the mutated solution and assign them to the population

        # Selection
        fc = s.costfunction(crossover_X)
        aux = (fc < s._Y)  # individuals where the cost function decreased
        s._X = aux.reshape((-1, 1)) * crossover_X + (1 - aux).reshape((-1, 1)) * s._X
        s._Y = aux * fc + (1 - aux) * s._Y

        s._bestidx = np.argmin(s._Y)        # index of best particle in Xmemory
        s._bestx = s._X[s._bestidx].copy()  # solution of best particle in Xmemory
        s._besty = s._Y[s._bestidx]         # cost of best particle in Xmemory
        return


class FA(Optimizer):
    name = 'FA'
    description = 'Firefly (FA) algorithm, revised'
    # algorithm tuning:
    n = 20  # Swarm size
    alpha0 = 0.5  # coefficient of random movement, is multiplied by the scale
    delta = 0.99  # randomization dampening coefficient
    exponent = 1  # gamma, light absorption exponent, should be related to the range of possible values. 1/(range^0.5)
    # suggested exponent is (1/(ub-lb))**0.5
    beta0 = 1  # constant that multiplies the attraction
    betamin = 0.2  # minimum possible attraction value. see calculation.

    # state variables:
    _iter = 0

    def __init__(self, costfunction, dimensions, lb, ub, maxiter=500, target_cost=None,
                 n=20, alpha0=0.5, delta=.99, exponent=1, beta0=1, betamin=0.2):
        """
        The cost function has to take arrays with (m,n) shape as inputs, where
         m is the number of particles and n is the number of dimensions.
        lb is the lower bound for each dimension
        ub is the upper bound for each dimension
        """
        s = self
        # TODO: do some serious input checking here.

        # problem parameters:
        s.costfunction = costfunction
        s.dimensions = dimensions
        s.lb = lb.copy()
        s.ub = ub.copy()
        s.maxiter = maxiter
        s.target_cost = target_cost

        s.scale = (ub - lb)  # scale for calculating random part
        # algorithm tuning:
        s.n = n

        u = ub.max()
        l = lb.min()

        s.alpha0 = alpha0
        s.delta = delta
        if exponent is None:
            s.exponent = (1 / (u - l))**0.5  # suggested value for the exponent
        else:
            s.exponent = exponent
        s.beta0 = beta0

        # initial conditions:
        s._X = np.random.random((n, dimensions)) * (ub - lb) + lb  # current particle solutions
        s._Y = s.costfunction(s._X)                        # current particle cost

        s._bestidx = np.argmin(s._Y)       # index of best particle in Xmemory
        s._bestx = s._X[s._bestidx].copy()  # solution of best particle in Xmemory
        s._besty = s._Y[s._bestidx]        # cost of best particle in Xmemory

        s._alpha = s.alpha0

        s._iter = 0

    def iterate_one(self):
        s = self

        # order fireflies by cost starting with the lower cost (brightest) ones:
        sequence = np.argsort(s._Y)
        s._Y = s._Y[sequence]
        s._X = s._X[sequence]
        s._bestidx = np.where(sequence == s._bestidx)[0][0]

        for i in range(s.n):  # for all fireflies
            position_of_brightest = s._X[i]
            for j in range(i + 1, s.n):  # iterate over the ones which are less bright
                # moving the firefly
                # calculating random part:
                random_part = s._alpha * (np.random.rand(s.dimensions) - 0.5) * s.scale
                initial_position = s._X[j]
                r2 = np.sum((position_of_brightest - initial_position)**2)
                # calculation of the attraction (beta) contribution:
                beta = (s.beta0 - s.betamin) * np.exp(-s.exponent * r2) + s.betamin
                # updating position:
                s._X[j] = initial_position + beta * (position_of_brightest - initial_position) + random_part
                del j, random_part, initial_position, r2, beta
            del i, position_of_brightest

        # update alpha value to migrate from exploration to exploitation gradually
        s._alpha = s._alpha * s.delta

        # applying search space limits:
        s._X = np.minimum(s.ub, s._X)
        s._X = np.maximum(s.lb, s._X)
        # calculating new costs:
        s._Y = s.costfunction(s._X)

        # WARNING!!! firefly algorithm does not check if individual solutions were improved

        s._bestidx = np.argmin(s._Y)        # index of best particle in current solution
        # if(s._besty>s._Y[s._bestidx]):
        s._bestx = s._X[s._bestidx].copy()  # solution of best particle
        s._besty = s._Y[s._bestidx]         # cost of best particle
        return


class FA_OBL(FA):
    """
      This is an attempt to addapt the OBL (Opposition-Based Learning) strategy to the firefly algorithm.

      This is original work by the programmer (Daniel), not a published algorithm from a reputed author.
    """

    name = 'FA_OBL'
    description = 'Firefly (FA) algorithm with the Opposition-Based Learning strategy'

    obl_iteration_threshold = 80  # number of iterations without improvement to trigger OBL
    obl_randomness = 0.1  # randomness to aply
    obl_probability = 0.5  # probability of a coordinate to be flipped by the OBL

    def __init__(self, *args, **kwargs):
        s = self
        kwargs2 = kwargs.copy()
        if('obl_iteration_threshold' in kwargs2.keys()):
            s.obl_iteration_threshold = kwargs2.pop('obl_iteration_threshold')
        if('obl_randomness' in kwargs2.keys()):
            s.obl_randomness = kwargs2.pop('obl_randomness')
        if('obl_probability' in kwargs2.keys()):
            s.obl_probability = kwargs2.pop('obl_probability')

        s._last_alpha = s.alpha0
        s._count_for_obl = 0
        # init in the superclass:
        FA.__init__(self, *args, **kwargs2)

    def iterate_one(self):
        s = self
        old_best = s._besty
        FA.iterate_one(self)
        new_best = s._besty
        if(new_best >= old_best):
            s._count_for_obl += 1
        else:
            s._last_alpha = s._alpha  # record the last alpha that produced an improvement
        if(s._count_for_obl >= s.obl_iteration_threshold):
            ### do Opposition-Based Learning ###
            bestidx = np.argmin(s._Y)
            invert = (np.random.rand(*(s._X.shape)) < s.obl_probability)
            # preserve the best individual:
            invert[bestidx, :] = False
            # calculate noise:
            noise = s.obl_randomness * (s.ub - s.lb) * (np.random.rand(*(s._X.shape)) - 0.5)  # scale noise according to the search limits
            # mirror X coordinates:
            #-x is  (lb+ub)/2 - (x-(lb+ub)/2) which equals lb+ub-x to mirror x about the center of the search limits
            minusX = ((s.ub + s.lb) - s._X)
            s._X = (1 - invert) * s._X + invert * (minusX + noise)
            # checking bounds:
            s._X = np.minimum(s.ub, s._X)
            s._X = np.maximum(s.lb, s._X)
            # updating costs:
            s._Y = s.costfunction(s._X)
            s._bestidx = np.argmin(s._Y)
            s._bestx = s._X[s._bestidx].copy()  # solution of best particle
            s._besty = s._Y[s._bestidx]

            # commented out to make it easier to isolate the effects of the OBL itself:
            # restore the alpha value of the firefly algorithm to the last one that produced an improvement
            # average it with the current alpha, because it may be too much
            # s._alpha=(0.8*s._last_alpha+0.2*s._alpha)
            s._count_for_obl = 0
            #print("Triggered OBL in FA_OBL at iteration",s._iter,s._alpha)


class DE_OBL(DE):
    """
      This is an attempt to addapt the OBL (Opposition-Based Learning) strategy to the differential evolution algorithm.

      This is original work by the programmer (Daniel), not a published algorithm from a reputed author.
    """

    name = 'DE_OBL'
    description = 'Differential Evolution (DE) algorithm with the Opposition-Based Learning strategy'

    obl_iteration_threshold = 80  # number of iterations without improvement to trigger OBL
    obl_randomness = 0.1  # randomness to aply
    obl_probability = 0.5  # probability of a coordinate to be flipped by the OBL

    def __init__(self, *args, **kwargs):
        s = self
        kwargs2 = kwargs.copy()
        if('obl_iteration_threshold' in kwargs2.keys()):
            s.obl_iteration_threshold = kwargs2.pop('obl_iteration_threshold')
        if('obl_randomness' in kwargs2.keys()):
            s.obl_randomness = kwargs2.pop('obl_randomness')
        if('obl_probability' in kwargs2.keys()):
            s.obl_probability = kwargs2.pop('obl_probability')

        s._count_for_obl = 0
        # init in the superclass:
        DE.__init__(self, *args, **kwargs2)

    def iterate_one(self):
        s = self
        old_best = s._besty
        DE.iterate_one(self)
        new_best = s._besty
        if(new_best >= old_best):
            s._count_for_obl += 1
        if(s._count_for_obl >= s.obl_iteration_threshold):
            ### do Opposition-Based Learning ###
            bestidx = np.argmin(s._Y)
            invert = (np.random.rand(*(s._X.shape)) < s.obl_probability)
            # preserve the best individual:
            invert[bestidx, :] = False
            # calculate noise:
            noise = s.obl_randomness * (s.ub - s.lb) * (np.random.rand(*(s._X.shape)) - 0.5)  # scale noise according to the search limits
            # mirror X coordinates:
            #-x is  (lb+ub)/2 - (x-(lb+ub)/2) which equals lb+ub-x to mirror x about the center of the search limits
            minusX = ((s.ub + s.lb) - s._X)
            s._X = (1 - invert) * s._X + invert * (minusX + noise)
            # checking bounds:
            s._X = np.minimum(s.ub, s._X)
            s._X = np.maximum(s.lb, s._X)
            # updating costs:
            s._Y = s.costfunction(s._X)
            s._bestidx = np.argmin(s._Y)
            s._bestx = s._X[s._bestidx].copy()  # solution of best particle
            s._besty = s._Y[s._bestidx]

            s._count_for_obl = 0
            #print("Triggered OBL in DE_OBL at iteration",s._iter)


class PSO_OBL(PSO):
    """
      This is an attempt to addapt the OBL (Opposition-Based Learning) strategy to the differential evolution algorithm.

      This is original work by the programmer (Daniel), not a published algorithm from a reputed author.
    """

    name = 'PSO_OBL'
    description = 'Particle Swarm Optimization (PSO) algorithm with the Opposition-Based Learning strategy'

    obl_iteration_threshold = 80  # number of iterations without improvement to trigger OBL
    obl_randomness = 0.1  # randomness to aply
    obl_probability = 0.5  # probability of a coordinate to be flipped by the OBL

    def __init__(self, *args, **kwargs):
        s = self
        kwargs2 = kwargs.copy()
        if('obl_iteration_threshold' in kwargs2.keys()):
            s.obl_iteration_threshold = kwargs2.pop('obl_iteration_threshold')
        if('obl_randomness' in kwargs2.keys()):
            s.obl_randomness = kwargs2.pop('obl_randomness')
        if('obl_probability' in kwargs2.keys()):
            s.obl_probability = kwargs2.pop('obl_probability')

        s._count_for_obl = 0
        # init in the superclass:
        PSO.__init__(self, *args, **kwargs2)

    def iterate_one(self):
        s = self
        old_best = s._besty
        PSO.iterate_one(self)
        new_best = s._besty
        if(new_best >= old_best):
            s._count_for_obl += 1
        if(s._count_for_obl >= s.obl_iteration_threshold):
            ### do Opposition-Based Learning ###
            bestidx = np.argmin(s._Y)
            invert = (np.random.rand(*(s._X.shape)) < s.obl_probability)
            # preserve the best individual:
            invert[bestidx, :] = False
            # calculate noise:
            noise = s.obl_randomness * (s.ub - s.lb) * (np.random.rand(*(s._X.shape)) - 0.5)  # scale noise according to the search limits
            # mirror X coordinates:
            #-x is  (lb+ub)/2 - (x-(lb+ub)/2) which equals lb+ub-x to mirror x about the center of the search limits
            minusX = ((s.ub + s.lb) - s._X)
            s._X = (1 - invert) * s._X + invert * (minusX + noise)
            # checking bounds:
            s._X = np.minimum(s.ub, s._X)
            s._X = np.maximum(s.lb, s._X)
            # updating costs:

            s._Y = s.costfunction(s._X)

            aux = np.where(s._Y < s._Ymemory)
            s._Xmemory[aux] = s._X[aux].copy()           # memory of best individual solution
            s._Ymemory[aux] = s._Y[aux].copy()           # memory of best individual fitness
            s._bestidx = np.argmin(s._Ymemory)           # index of best particle in Xmemory
            s._bestx = s._Xmemory[s._bestidx].copy()  # solution of best particle in Xmemory
            s._besty = s._Ymemory[s._bestidx]         # cost of best particle in Xmemory

            s._count_for_obl = 0
            #print("Triggered OBL in PSO_OBL at iteration",s._iter)


class FA_CP(FA):
    """
      This is an attempt to addapt the CP (Passive Congregation) strategy to the firefly algorithm.

      This is original work by the programmer (Daniel), not a published algorithm from a reputed author.
    """
    name = 'FA_CP'
    description = 'Firefly (FA) algorithm with the Passive Congregation (CP) strategy'

    cp_confidence_in_second = 0.5  # Passive Congregation confidence in the second best individual

    def __init__(self, *args, **kwargs):
        s = self
        kwargs2 = kwargs.copy()
        if('cp_confidence_in_second' in kwargs2.keys()):
            s.cp_confidence_in_second = kwargs2.pop('cp_confidence_in_second')

        # init in the superclass:
        FA.__init__(self, *args, **kwargs2)

    def iterate_one(self):
        s = self

        # order fireflies by cost starting with the lower cost (brightest) ones:
        sequence = np.argsort(s._Y)
        s._Y = s._Y[sequence]
        s._X = s._X[sequence]
        s._bestidx = np.where(sequence == s._bestidx)[0][0]

        for i in range(s.n):  # for all fireflies
            position_of_brightest = s._X[i]
            for j in range(max(0, i - 1), s.n):  # MODIFIED BY THE Passive Congregation CP strategy.
                # moving the firefly
                # calculating random part:
                random_part = s._alpha * (np.random.rand(s.dimensions) - 0.5) * s.scale
                initial_position = s._X[j]
                r2 = np.sum((position_of_brightest - initial_position)**2)
                # calculation of the attraction (beta) contribution:
                beta = (s.beta0 - s.betamin) * np.exp(-s.exponent * r2) + s.betamin
                # updating position:
                if(i > j):  # attract to the less bright, weight by confidence
                    s._X[j] = initial_position + s.cp_confidence_in_second * beta * (position_of_brightest - initial_position) + random_part
                # if i==j do not update the position.
                if(i < j):  # do regular algorithm
                    s._X[j] = initial_position + beta * (position_of_brightest - initial_position) + random_part
                del j, random_part, initial_position, r2, beta
            del i, position_of_brightest

        # update alpha value to migrate from exploration to exploitation gradually
        s._alpha = s._alpha * s.delta

        # applying search space limits:
        s._X = np.minimum(s.ub, s._X)
        s._X = np.maximum(s.lb, s._X)
        # calculating new costs:
        s._Y = s.costfunction(s._X)

        # WARNING!!! firefly algorithm does not check if individual solutions were improved

        s._bestidx = np.argmin(s._Y)        # index of best particle in current solution
        # if(s._besty>s._Y[s._bestidx]):
        s._bestx = s._X[s._bestidx].copy()  # solution of best particle
        s._besty = s._Y[s._bestidx]         # cost of best particle
        return


class DE_CP(DE):
    """
      This is an attempt to addapt the CP (Passive Congregation) strategy to the Differential Evolution algorithm.

      This is original work by the programmer (Daniel), not a published algorithm from a reputed author.
    """

    name = 'DE_CP'
    description = 'Differential Evolution (DE) algorithm with the Passive Congregation (CP) strategy'

    cp_confidence_in_second = 0.5  # Passive Congregation confidence in the second best individual

    def __init__(self, *args, **kwargs):
        s = self
        kwargs2 = kwargs.copy()
        if('cp_confidence_in_second' in kwargs2.keys()):
            s.cp_confidence_in_second = kwargs2.pop('cp_confidence_in_second')

        # init in the superclass:
        DE.__init__(self, *args, **kwargs2)

    def _mutate(self, direction):
        """
          Selects neighbours and apply mutation.
        """

        s = self
        n = s.n
        assert(n > 3)  # only works with 4 or more individuals
        # getting neighbours to permutate with probabilities biased to the best fitting:
        a, b = (np.min(s._Y), np.max(s._Y))
        weight = s.cp_confidence_in_second
        b = max(a + 0.001, b)  # avoiding division by zero
        p = (1 - weight) + weight * (s._Y - a) / (b - a)  # relative probability varies between 1-weight and 1
        p = p / p.sum()  # normalized probabilites
        p = np.cumsum(p)  # cumulative probabilities to make it easy to toss a random number and search
        n0 = np.arange(n)  # self
        n1 = p.searchsorted(np.random.rand(n))  # neighbours different from self
        aux = np.where(n1 == n0)[0]
        while(len(aux) > 0):
            n1[aux] = p.searchsorted(np.random.rand(len(aux)))[:]
            aux = np.where(n1 == n0)[0]
        n2 = p.searchsorted(np.random.rand(n))  # neighbours different from self and n1
        aux = np.where((n2 == n1) + (n2 == n0))[0]
        while(len(aux) > 0):
            n2[aux] = p.searchsorted(np.random.rand(len(aux)))[:]
            aux = np.where((n2 == n1) + (n2 == n0))[0]
        n3 = p.searchsorted(np.random.rand(n))  # neighbours different from self, n1 and n2
        aux = np.where((n3 == n2) + (n3 == n1) + (n3 == n0))[0]
        while(len(aux) > 0):
            n3[aux] = p.searchsorted(np.random.rand(len(aux)))[:]
            aux = np.where((n3 == n2) + (n3 == n1) + (n3 == n0))[0]
        '''
    #DEBUG:
    assert((n0!=n1).all())
    assert((n0!=n2).all())
    assert((n0!=n3).all())
    assert((n1!=n2).all())
    assert((n1!=n3).all())
    assert((n2!=n3).all())
    #'''
        # mutation:
        m = s._X[n1] + direction * s.f * (s._X[n2] - s._X[n3])
        return m


all_algorithms = {i[0]: i[1] for i in vars().copy().items() if
                  hasattr(i[1], 'iterate_one') and
                  hasattr(i[1], 'run') and
                  hasattr(i[1], 'run_with_history')}


def test(algorithm, Fitnessfunc, dimensions, tolerance=1e-3, **kwargs):
    """
      Does basic testing of the algorithm and plots a convergence curve.

      Do not forget to call plt.show() after running all the tests you want to show.
      Example:
        test(PSO,
             fitnessfunctions.Rastrigin,
             ndim=2,
             maxiter=500,
             tolerance=1e-2,
             n=30)
        from matplotlib import pyplot as plt
        plt.show()
    """
    # TODO: check the fitnessfunction test and see if there are any tests that can be applied here
    # TODO: do a lot more tests
    # TODO: check if the name attribute is the same as the name of the class.
    f = Fitnessfunc.evaluate
    lb, ub = Fitnessfunc.default_bounds(dimensions)
    ymin, xmin = Fitnessfunc.default_minimum(dimensions)
    a = algorithm(f, dimensions, lb, ub, **kwargs)
    y, x = a.run_with_history()
    cost_delta = ((y[-1] - ymin)**2).sum()**0.5
    solution_delta = ((x[-1] - xmin)**2).sum()**0.5
    print('cost difference to ideal:   ', cost_delta)
    print('solution distance to ideal: ', solution_delta)
    print('converged within tolerance? ', cost_delta < tolerance)
    print('Solution found:\n', x[-1])
    print('Theoretical best solution possible:\n', xmin)
    print('cost achieved:\n', y[-1])
    print('Theoretical best cost:\n', ymin)

    from matplotlib import pyplot as plt
    fig = plt.figure(Fitnessfunc.name + ' cost function, ' + a.description)
    ax1 = fig.add_subplot(211)
    # ax1.plot(np.log10(cost_history))
    ax1.plot(y)
    # ax1.set_label(a.name)
    ax1.set_ylabel("$mincost$")
    ax2 = fig.add_subplot(212)
    ax2.plot(np.log10(np.array(y) - ymin))
    ax2.set_ylabel("$log_{10}(mincost-theoreticalminimum)$")
    ax2.set_xlabel("$iteration$")
    # plt.show()


def test_all():
    # c=fitnessfunctions.Sphere
    # c=fitnessfunctions.Rastrigin
    c = fitnessfunctions.Schwefel
    # c=fitnessfunctions.Michalewicz
    # c=fitnessfunctions.Rosenbrock
    ndim = 6
    nparticles = 30
    for i in all_algorithms.items():
        test(i[1], c, ndim, maxiter=1000, tolerance=1e-2, n=nparticles)
    #from matplotlib import pyplot as plt
    # plt.show()

#all_algorithms={i[0]:i[1] for i in all_algorithms.items() if 'FA' in i[0]}
# test_all()
#from matplotlib import pyplot as plt
# plt.show()
