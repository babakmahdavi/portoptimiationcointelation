# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 13:12:47 2019

@author: babak
"""

import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import tensorflow as tf

import DGMnets



# Set random seeds
np.random.seed(42)
tf.set_random_seed(42)


# PDE parameters
r = 0.01/1000.0            # Interest rate
sigma = 0.15#((0.15)**2)/364.0
eta = sigma       # Volatility
mu = 0.1/1000.0           # Mean
kappa = 0.01
rho = -0.4
lambd = sigma**2 - 2*sigma*eta*rho + eta**2
gamma = .95         # Utility decay

# Time limits
T0 = 0.0 + 1e-10    # Initial time
T  = 1.0            # Terminal time
#T0 = 0.0 + 1e-10    # Initial time
#T  = 0.01            # Terminal time

# Space limits
#S1 = 0.5 + 1e-10    # Low boundary
#S2 = 2.0              # High boundary
S1 = 0.5     # Low boundary
S2 = 2.0              # High boundary


# Merton's analytical known solution
#def analytical_solution(t, x):
 #   return -np.exp(-x*gamma*np.exp(r*(T-t)) - (T-t)*0.5*lambd**2)

#def analytical_dVdx(t,x):
 #  return gamma*np.exp(-0.5*(T-t)*lambd**2 + r*(T-t))*np.exp(-x*gamma*np.exp(r*(T-t)))

#def analytical_dV2dxx(t,x):
  # return -gamma**2*np.exp(-0.5*(T-t)*lambd**2 + 2*r*(T-t))*np.exp(-x*gamma*np.exp(r*(T-t)))

# Merton's final utility function
#def utility(x):
 #   return -tf.exp(-gamma*x)



# Loss function
def loss(model, t1, x1, t2, x2, t3, x3):
    # Loss term #1: PDE
    V = model(t1, x1)
    V_t = tf.gradients(V, t1)[0]
    V_x = tf.gradients(V, x1)[0]
    V_xx = tf.gradients(V_x, x1)[0]
    f = lambd*(gamma -1)*V*V_t - 0.5*(lambd**2)*gamma*(x1**2)*(V_x**2) - 0.5*gamma*(mu - kappa*(x1 - 1))**2*V + \
    0.5*lambd*(gamma -1)*(x1**2)*V*V_xx - lambd*gamma*(mu - kappa*(x1-1))*x1*V*V_x + \
    lambd*gamma*(gamma -1)*r*(V**2) + lambd*(gamma -1)*(mu + eta**2 - sigma*eta*rho - kappa*(x1-1))*V*V_x                          

    L1 = tf.reduce_mean(tf.square(f))

    # Loss term #2: boundary condition
    L2 = 0.0
    
    # Loss term #3: initial/terminal condition
    L3 = tf.reduce_mean(tf.square(model(t3, x3) - 1))
    return (L1, L2, L3)


# Sampling
def sampler(N1, N2, N3):
    # Sampler #1: PDE domain
    t1 = np.random.uniform(low=T0 - 0.5*(T - T0),
                           high=T,
                           size=[N1,1])
    s1 = np.random.uniform(low=S1 - (S2 - S1)*0.5,
                           high=S2 + (S2 - S1)*0.5,
                           size=[N1,1])

    # Sampler #2: boundary condition
    t2 = np.zeros(shape=(1, 1))
    s2 = np.zeros(shape=(1, 1))
    
    # Sampler #3: initial/terminal condition
    t3 = T * np.ones((N3,1)) #Terminal condition
    s3 = np.random.uniform(low=S1 - (S2 - S1)*0.5,
                           high=S2 + (S2 - S1)*0.5,
                           size=[N3,1])
    
    return (t1, s1, t2, s2, t3, s3)


# Neural Network definition
num_layers = 4#3
nodes_per_layer = 20#50
model = DGMnets.DGMNet(num_layers, nodes_per_layer)

t1_t = tf.placeholder(tf.float32, [None,1])
x1_t = tf.placeholder(tf.float32, [None,1])
t2_t = tf.placeholder(tf.float32, [None,1])
x2_t = tf.placeholder(tf.float32, [None,1])
t3_t = tf.placeholder(tf.float32, [None,1])
x3_t = tf.placeholder(tf.float32, [None,1])

L1_t, L2_t, L3_t = loss(model, t1_t, x1_t, t2_t, x2_t, t3_t, x3_t)
loss_t = L1_t + L2_t + L3_t


# Optimizer parameters
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.001
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           100000, 0.96, staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_t)


# Training parameters
steps_per_sample = 10
sampling_stages = 50#200

# Number of samples
NS_1 = 1000
NS_2 = 0
NS_3 = 100

# Plot tensors
tplot_t = tf.placeholder(tf.float32, [None,1], name="tplot_t") # We name to recover it later
xplot_t = tf.placeholder(tf.float32, [None,1], name="xplot_t")
vplot_t = tf.identity(model(tplot_t, xplot_t), name="vplot_t") # Trick for naming the trained model
dplot_t = tf.identity(tf.gradients(model(tplot_t, xplot_t), xplot_t)[0], name="dplot_t") 
dtplot_t = tf.identity(tf.gradients(model(tplot_t, xplot_t), tplot_t)[0], name="dtplot_t") 

# Training data holders
sampling_stages_list = []
elapsed_time_list = []
loss_list = []
L1_list = []
L3_list = []


# Train network!!
init_op = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init_op)

for i in range(sampling_stages):
    t1, x1, t2, x2, t3, x3 = sampler(NS_1, NS_2, NS_3)

    start_time = time.clock()
    for _ in range(steps_per_sample):
        loss, L1, L3, _ = sess.run([loss_t, L1_t, L3_t, optimizer],
                               feed_dict = {t1_t:t1, x1_t:x1, t2_t:t2, x2_t:x2, t3_t:t3, x3_t:x3})
    end_time = time.clock()
    elapsed_time = end_time - start_time

    sampling_stages_list.append(i)
    elapsed_time_list.append(elapsed_time)
    loss_list.append(loss)
    L1_list.append(L1)
    L3_list.append(L3)
    
    text = "Stage: {:04d}, Loss: {:e}, L1: {:e}, L3: {:e}, {:f} seconds".format(i, loss, L1, L3, elapsed_time)
    print(text)



# Plot results
N = 1000      # Points on plot grid

times_to_plot = [-2.5*T, -1.5*T, 0.0*T, 0.33*T, 0.66*T, T]
tplott = np.linspace(T0, T, N)
xplott = np.linspace(S1, S2, N)

#plt.figure(figsize=(10,8))
#i = 1
#for t in tplot:
   # for x in xplot:
    
    #solution_plot = analytical_solution(t, xplot)
     #   tt = t*np.ones_like(xplot.reshape(-1,1))
      #  xx = x*np.ones_like(tplot.reshape(-1,1))
      #  nn_plot, = sess.run([vplot_t],
                    #      feed_dict={tplot_t:tt, xplot_t:xplot.reshape(-1,1)})

    #plt.subplot(3,2,i)
    #plt.plot(xplot, solution_plot, 'b')
    #plt.plot(xplot, nn_plot, 'r')

    #plt.ylim(-3.1, 1.2)
    #plt.xlabel("z")
   # plt.ylabel("f")
  #  plt.title("t = %.2f"%t, loc="left")
 #   i = i+1

#plt.subplots_adjust(wspace=0.3, hspace=0.3)
#plt.show()

tplot, xplot = np.meshgrid(tplott, xplott)
#sol_an = analytical_solution(tplot, xplot)
appro_sol  = sess.run([vplot_t],
                        feed_dict={tplot_t:tplot.reshape(-1,1), xplot_t:xplot.reshape(-1,1)})
approx_sol = np.asarray(appro_sol)
approx_soll = approx_sol.reshape(-N,N)

deriv_x = sess.run([dplot_t],
                        feed_dict={tplot_t:tplot.reshape(-1,1), xplot_t:xplot.reshape(-1,1)})

derivat_x = np.asarray(deriv_x)
derivatt_x = derivat_x.reshape(-N,N)


deriv_t = sess.run([dtplot_t],
                        feed_dict={tplot_t:tplot.reshape(-1,1), xplot_t:xplot.reshape(-1,1)})

derivat_t = np.asarray(deriv_t)
derivatt_t = derivat_t.reshape(-N,N)



print(derivat_x)
print(derivat_x.size)
print(derivat_t)
print(derivat_t.size)
print(approx_soll)
print(approx_soll.size)

#print(tplot.reshape(-1,1))
#print(sol_an)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(tplot, xplot, approx_soll, cmap=cm.coolwarm)
ax.set_xlabel('$t$', fontsize=10)
ax.set_ylabel('$z$', fontsize=10)
ax.set_zlabel('$f$', fontsize=10)
fig.suptitle('Plot of DGM solution', fontsize=14)
fig.colorbar(surf, shrink=0.5, aspect=15)
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surf_x = ax.plot_surface(tplot, xplot, derivatt_x, cmap=cm.coolwarm)
ax.set_xlabel('$t$', fontsize=10)
ax.set_ylabel('$z$', fontsize=10)
ax.set_zlabel('$f_z$', fontsize=10)
fig.suptitle('Plot of derivative wrt to z ', fontsize=14)
fig.colorbar(surf_x, shrink=0.5, aspect=15)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surf_t = ax.plot_surface(tplot, xplot, derivatt_t, cmap=cm.coolwarm)
ax.set_xlabel('$t$', fontsize=10)
ax.set_ylabel('$z$', fontsize=10)
ax.set_zlabel('$f_t$', fontsize=10)
fig.suptitle('Plot of derivative wrt to t ', fontsize=14)
fig.colorbar(surf_t, shrink=0.5, aspect=15)
plt.show()

pi = np.zeros((N,N))

for i in range(0,N):
    for j in range(0,N):
        pi[i,j] =  - (xplott[j]*derivatt_x[i,j])/((gamma-1)*approx_soll[i,j]) - (mu - kappa*(xplott[j] - 1))/(lambd*(gamma - 1))


print(pi)
#pi_ = pi[500:,500:]
#tplot_ = tplot[500:]
#xplot_ = xplot[500:]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#surf_pi = ax.plot_surface(tplot_, xplot_, pi_, cmap=cm.coolwarm)
surf_pi = ax.plot_surface(tplot, xplot, pi, cmap=cm.coolwarm)
ax.set_xlabel('Time $t$', fontsize=10)
ax.set_ylabel('$z$', fontsize=10)
ax.set_zlabel('Optimal weight $\pi$', fontsize=10)
#fig.suptitle('Weights ', fontsize=14)
titleString = 'weights with r=' + str(r) +', $\sigma$=' + str(sigma) + ', $\eta$=' + str(eta) + ', $\mu$=' + str(mu) + ', $\\rho$=' + str(rho) + ', $\gamma$=' + str(gamma) + ' and $\kappa$=' + str(kappa)
#print(titleString)
#fig.suptitle('Weights ', fontsize=14)
fig.suptitle(titleString, fontsize=10)
fig.colorbar(surf_pi, shrink=0.5, aspect=15)
plt.show()


piLast = pi[:,0]
#piLast = pi[:,-1]
fig2 = plt.figure()
ax = fig2.add_subplot(111)
ax.plot(xplot, piLast)
ax.set_xlabel('$z$', fontsize=10)
ax.set_ylabel('$w$', fontsize=10)
fig2.suptitle(titleString, fontsize=10)

plt.show()
           
# Save the trained model
#saver = tf.train.Saver()
#saver.save(sess, './SavedNets/merton_1d')


# Save the time tracking
#np.save('./TimeTracks/merton_1d',
        #(sampling_stages_list, elapsed_time_list, loss_list, L1_list, L3_list))



# Plot losses X stages
#plt.figure(figsize=(12,5))

#plt.subplot(1,3,1)
#plt.semilogy(sampling_stages_list,loss_list)
#plt.title("Loss", loc="left")
#plt.xlabel("Stage")
#plt.ylim(1e-8, 1)

#plt.subplot(1,3,2)
#plt.semilogy(sampling_stages_list,L1_list)
#plt.title("L1 loss", loc="left")
#plt.xlabel("Stage")
#plt.ylim(1e-8, 1)

#plt.subplot(1,3,3)
#plt.semilogy(sampling_stages_list,L3_list)
#plt.title("L3 loss", loc="left")
#plt.xlabel("Stage")
#plt.ylim(1e-8, 1)

#plt.show()
