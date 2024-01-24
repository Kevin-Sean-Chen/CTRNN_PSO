# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 04:11:57 2024

@author: kevin
"""

from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import gamma

import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)

# %%
# CTRNN with three neurons, each corresponding to run, rev, and turn
# PSO_test class would perform particle swarm optimization given RNN target output
# the target has state-dependent IO

# %% CTRNN class
class CTRNN:
    def __init__(self, N, dt, lt, K):
        """
        network with N neurons, dt step, and lt length
        K repeats for statsitics
        """
        self.N = N
        self.dt = dt
        self.lt = lt
        self.K = K

    def sim_RNN(self, x):
        """
        ipnput parameter vector x and unpack for RNN
        ---
        parameters:
        wij: N x N
        tau_i: N
        b_i: N
        sig_ni: N
        tau_ni: N
        thre: scalar
        ---
        return time series simulated for RNN
        """
        wij, tau_i, b_i, sig_ni, tau_ni = self.unpack_vec(x) #,thre
        lt, dt, N = self.lt, self.dt, self.N
        xt = np.zeros((N,lt))
        xt[:,0] = np.random.rand(N)
        It = xt*1
        rt = xt*1
        x_out = np.zeros(lt)
        # print(len(b_i))
        for tt in range(lt-1):
            xt[:,tt+1] = xt[:,tt] + dt/tau_i*(-xt[:,tt] + wij @ self.NL(xt[:,tt] + b_i) + It[:,tt])
            It[:,tt+1] = It[:,tt] + dt*(-It[:,tt]/tau_ni + dt**0.5*sig_ni*1*np.random.randn(N))
            rt[:,tt+1] = self.NL(xt[:,tt+1] + b_i)
            
            x_out[tt+1] = np.argmax(rt[:,tt+1])  # competetive circuit assumption
        return x_out, rt
    
    def rep_RNN(self, x):
        """
        repeat sim_RNN for better statsitics
        """
        outs = []
        for i in range(self.K):
            x_out, _ = self.sim_RNN(x)
            outs.append(x_out)
        return outs

    def unpack_vec(self, x):
        """
        return params given vector x, modify when model changes...
        """
        N = self.N
        wij = x[:N**2].reshape(N,N)
        tau_i = x[N**2:N**2+N]
        b_i = x[N**2+N:N**2+N*2]
        sig_ni = x[N**2+N*2:N**2+N*3]
        tau_ni = x[N**2+N*3:N**2+N*4]
        # thre = x[-N:]
        return wij, tau_i, b_i, sig_ni, tau_ni  #, thre
    
    def make_bounds(self):
        # hand-tuned for now...
        # wij, tauij, bij, sigij, tauNij, thre
        
        U_ws = np.ones(self.N**2)*20
        L_ws = np.ones(self.N**2)*-20
        U_tau = np.ones(self.N)*1
        L_tau = np.ones(self.N)*.1
        U_b = np.ones(self.N)*10
        L_b = np.ones(self.N)*-10
        U_sig = np.ones(self.N)*100
        L_sig = np.ones(self.N)*1 
        U_tauN = np.ones(self.N)*1
        L_tauN = np.ones(self.N)*.1
        # U_thre = np.ones(self.N)*1 
        # L_thre = np.ones(self.N)*0
        
        xU = np.concatenate((U_ws, U_tau, U_b, U_sig, U_tauN))
        xL = np.concatenate((L_ws, L_tau, L_b, L_sig, L_tauN))
        
        return xU, xL
    
    def NL(self, x):
        """
        simple nonlinearity for continuous RNN
        """
        nl = 1/(1 + np.exp(-x))
        return nl

    def fitness(self, xt, tt):
        """
        input time series xt and target tt 
        return fittness between them
        """
        # targ_hist, bins = tt
        targ_hist, bins, _ = tt
        hist = self.dwell_time(xt, bins)
        ft = -np.sum(np.abs(hist - targ_hist))#*bins[1:])  # can adjust later...
        return ft
    
    def fitness_hmm(self, xt, tt):
        """
        input time series xt and target tt 
        return fittness between them
        """
        _,_,tt = tt
        M_est = self.compute_transition_matrix(xt)
        ft = -np.sum(np.abs(M_est - tt)**2)*1
        # ft = -np.linalg.norm(M_est-tt)
        return ft
    
    def dwell_time(self, state_t, bins):
        """
        given binary time-series, compute dwell time histogram
        """
        reps = len(state_t)
        rep_dwell_time = np.array([])
        for rr in range(reps):
            change_indices = np.where(np.diff(state_t[rr]) != 0)[0]
            dwell_times = np.diff(np.concatenate(([0], change_indices, [len(state_t[rr])])))
            rep_dwell_time = np.concatenate([rep_dwell_time, dwell_times])
        hist, bin_edges = np.histogram(np.array(rep_dwell_time), bins=bins)
        ##########
        # check this~~
        ##########
        return hist/np.sum(hist)  # normalized ocupency!


    def compute_transition_matrix(self, state_t):
        """
        compute state transition matrix
        """
        reps = len(state_t)
        transition_matrix = np.zeros((self.N, self.N))+1
        
        for rr in range(reps):
            state_i = state_t[rr]
            for i in range(len(state_i) - 1):
                current_state_ind = int(state_i[i])
                next_state_ind = int(state_i[i + 1])
        
                transition_matrix[current_state_ind, next_state_ind] += 1
        
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_matrix_prob = transition_matrix / row_sums
    
        return transition_matrix_prob

    
    def particles_sim_and_eva(self, x, target_bin_count):
        """
        putting simulation together
        input parameter vector x (D x Np) and target bin-counts
        return Np particles of fittness fts
        """
        Np = x.shape[1]  # D x Np
        fts = np.zeros(Np)
        for pp in  range(Np):
            out = self.rep_RNN(x[:,pp])
            fts[pp] = self.fitness(out, target_bin_count)
            fts[pp] += self.fitness_hmm(out, target_bin_count)
        return fts
    

# %% PSO class

class PSO_test:
    def __init__(self, sim_and_eva_class, target_bin_count):
        ### initialize pso settings, tuned for now
        self.w = 0.5                   # Intertial weight, can change later
        self.c1 = 2.0                  # Weight of searching based on the optima found by a particle
        self.c2 = 2.0                  # Weight of searching based on the optima found by the swarm
        self.v_fct = 0.02*5
        self.Np = 50                   # Number of particles
        self.max_iter = 20             # Maximum iteration
        self.D = 13                    # Parameter dimensions
        self.model_class = sim_and_eva_class  # simulation and evaluations are in this class
        self.target_bin_count = target_bin_count
        
    def run_pso(self):
        
        ### initialize vectors
        pbest_val = np.zeros(self.Np)            # Personal best fintess value. One pbest value per particle.
        gbest_val = np.zeros(self.max_iter)      # Global best fintess value. One gbest value per iteration (stored).

        pbest = np.zeros((self.D, self.Np))      # pbest solution
        gbest = np.zeros(self.D)                 # gbest solution

        gbest_store = np.zeros((self.D, self.max_iter)) # storing gbest solution at each iteration

        x = np.random.rand(self.D, self.Np)            # Initial position of the particles
        v = np.zeros((self.D, self.Np))                # Initial velocity of the particles

        x_store = np.zeros((self.max_iter, self.D, self.Np))  # iter x dim x particles


        # Setting the initial position of the particles over the given bounds [xL,xU]
        xU, xL = self.model_class.make_bounds()
        for m in range(self.D):    
            x[m,:] = xL[m] + (xU[m] - xL[m])*x[m,:]

        # Function call. Evaluates the fitness of the initial swarms    
        fit = self.model_class.particles_sim_and_eva(x, self.target_bin_count)           # vector of size Np

        pbest_val = np.copy(fit)   # initial personal best = initial fitness values. Vector of size Np
        pbest = np.copy(x)         # initial pbest solution = initial position. Matrix of size D x Np

        # Calculating gbest_val and gbest. Note that gbest is the best solution within pbest                                                                                                                      
        ind = np.argmax(pbest_val)                # index where pbest_val is maximum. 
        gbest_val[0] = np.copy(pbest_val[ind])    # set initial gbest_val
        gbest = np.copy(pbest[:,ind])

        print("Iter. =", 0, ". gbest_val = ", gbest_val[0])
        print("gbest_val = ",gbest_val[0])

        x_store[0,:,:] = x

        # Loop over the generations
        for iter in range(1, self.max_iter):          
          
            r1 = np.random.rand(self.D, self.Np)           # random numbers [0,1], matrix D x Np
            r2 = np.random.rand(self.D, self.Np)           # random numbers [0,1], matrix D x Np   
            v_global = np.multiply(((x.transpose()-gbest).transpose()),r2)*self.c2*(-1.0)    # velocity towards global optima
            v_local = np.multiply((pbest- x),r1)*self.c1           # velocity towards local optima (pbest)

            w = 0.9 - 0.7*iter/self.max_iter  # test annealing
            v = w*v + v_local + v_global        # velocity update
          
            x = x + v*self.v_fct                     # position update
             
            fit = self.model_class.particles_sim_and_eva(x, self.target_bin_count)              # fitness function call (once per iteration). Vector Np
            
            # pbest and pbest_val update
            ind = np.argwhere(fit > pbest_val)  # indices where current fitness value set is greater than pbset
            pbest_val[ind] = np.copy(fit[ind])  # update pbset_val at those particle indices where fit > pbest_val
            pbest[:,ind] = np.copy(x[:,ind])    # update pbest for those particle indices where fit > pbest_val
                  
            # gbest and gbest_val update
            ind2 = np.argmax(pbest_val)                       # index where the fitness is maximum
            gbest_val[iter] = np.copy(pbest_val[ind2])        # store gbest value at each iteration
            gbest = np.copy(pbest[:,ind2])                    # global best solution, gbest
            
            gbest_store[:,iter] = np.copy(gbest)              # store gbest solution

            print("Iter. =", iter, ". gbest_val = ", gbest_val[iter])  # print iteration no. and best solution at each iteration
            x_store[iter,:,:] = x
            
        return gbest_val, gbest_store

# %% test rounds...
# %% rnn simulation
N, dt, lt, K = 3, 0.1, 100, 30
nparam = N**2 + N*4
myrnn = CTRNN(N, dt, lt, K)
test_sim, _ = myrnn.sim_RNN(np.random.randn(nparam)*1)
plt.plot(test_sim.T)

# %% particles
bins = np.arange(0, 101, 5)  # bins for dwell time
targ_hist = np.exp(-bins/10) # normalized occupency
# targ_hist = gamma.pdf(bins, a=5, scale=1/.2)
targ_hist = targ_hist[:-1]  # to match bins
targ_hist = targ_hist/np.sum(targ_hist)
Mt = np.array([[0.8, 0.1,0.1],
               [0.2, 0.6,0.2],
               [0.05, 0.05,0.9]])
target_bin_count = targ_hist, bins  , Mt
xs = np.random.randn(nparam,K)
test_ft = myrnn.particles_sim_and_eva(xs, target_bin_count)

# %% test pso
my_pso = PSO_test(myrnn, target_bin_count)
my_pso.D = nparam
gbest_val, gbest_store = my_pso.run_pso()
plt.figure()
plt.plot(gbest_val)
plt.xlabel('iterations')
plt.ylabel('fitness, gbest_val')

# %% test run
x_best = gbest_store[:,-1]
best_sim, xt = myrnn.sim_RNN(x_best)
plt.figure()
plt.subplot(211)
plt.plot(best_sim.T)
plt.subplot(212)
plt.plot(xt.T)

# %% check stats
out = myrnn.rep_RNN(x_best)
hist = myrnn.dwell_time(out, bins)
Mt_inf = myrnn.compute_transition_matrix(out)
plt.figure(figsize=(7, 7))
plt.subplot(2, 2, (1, 2))
plt.title('dwell time', fontsize=20)
plt.plot(bins[:-1], hist, label='fit')
plt.plot(bins[:-1], targ_hist,'--', label='target')
plt.legend(fontsize=20)
plt.subplot(2, 2, 3)
plt.title('target transition', fontsize=20)
plt.imshow(Mt)
plt.xticks([]); plt.yticks([])
plt.subplot(2, 2, 4)
plt.imshow(Mt_inf)
plt.xticks([]); plt.yticks([])
plt.title('fit transition', fontsize=20)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

# %%
# next steps:
# add input
# check trained RNN
# constrain with HMM
