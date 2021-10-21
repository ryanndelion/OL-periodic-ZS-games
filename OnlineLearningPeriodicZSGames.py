#!/usr/bin/env python
# coding: utf-8

# # Online Learning in Periodic Zero-Sum Games

# In[1]:


import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import odeint, ode, quad, trapz
from scipy import optimize
from scipy.spatial import distance
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
sns.set_style("darkgrid")
from PIL import Image
from scipy.stats import entropy
from IPython.display import HTML

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Alternating Gradient Descent-Ascent 

# In[2]:


def MPderivGDA(s,t, scale, func, util):
    
    x = np.array([[s[0]], [s[1]]])
    y = np.array([[s[2]], [s[3]]])
    
    A = lambda i: 1*np.array([[func(i), -func(i)*scale],
                        [-func(i)*scale, func(i)]])
    B = lambda i: -1*np.array([[func(i), -func(i)*scale],
                        [-func(i)*scale, func(i)]])
    dxdt = (A(t)@y).flatten().tolist()
    dydt = (-A(t).T@x).flatten().tolist()
    util.append(x.T@A(t)@y)
    return np.concatenate((dxdt,dydt))
    


# In[3]:


def MPTrajectory(f=MPderivGDA, s=[0.45, 0.55, 0.55, 0.45], numperiod=10, 
                 numstep=2000, scale=1, plot=True, func=np.sin, util=[]) :
    partuple=(scale, func)    # Converts parameters to a tuple in the right order
    tvals=np.linspace(0,numperiod*2*np.pi,numstep)
    func_tvals = [func(x) for x in tvals]
    traj=odeint(f,s,tvals, args=(scale, func, util))
    # Store the results of odeint in a dictionary
    data={}
    data["times"]=tvals
    data["x1"]=traj[:,0]
    data["x2"]=traj[:,1]
    data["y1"]=traj[:,2]
    data["y2"]=traj[:,3]
    data['periodic']=func_tvals
    data['utils']=util
    if plot:
        fig = plt.figure(figsize=(12,10)) 
        ax1 = fig.add_subplot(211)
        ax1.plot(func_tvals)
        ax1.set_xlabel("Time")
        ax1.set_ylabel("State")
        ax1.set_title('Periodic Function', fontsize=16)
        ax2 = fig.add_subplot(212)
        ax2.plot(traj[:,0])
        ax2.plot(traj[:,2])
        ax2.set_xlabel("Time")
        ax2.set_ylabel("System State")
        ax2.set_title('Trajectories', fontsize=16)
        ax2.legend(['x_11','x_21'], loc='best');
        fig.tight_layout();
    return data


# ### Example with time average game = 0

# With the typical Matching Pennies payoff matrix, time average of the payoff matrix A(t) = 0. This results in gradient descent trajectories that do not converge in time average to 0.

# In[4]:


def f_changing_speed(x):
    val = f_modified_sin(x)
    
    if 0 <= x%(2*np.pi) <= 3*np.pi/2:
        val = np.sin(x)
    else: val=(2/np.pi)*(x%(2*np.pi)-2*np.pi)
    return val


# In[5]:


def f_modified_sin(x):
    if 0 <= x%(2*np.pi) <= 3*np.pi/2:
        val = np.sin(x)
    else: val=(2/np.pi)*(x%(2*np.pi)-2*np.pi)
    return val


# In[6]:


data_1_gda = MPTrajectory(f=MPderivGDA, s = [2.35, 0.65, 1.65, 0.35], scale=1, numperiod=100, numstep=5000, func=f_modified_sin)


# In[7]:


print('Time average of x_11:', np.mean(data_1_gda['x1']))
print('Time average of x_12:', np.mean(data_1_gda['x2']))
print('Time average of x_21:', np.mean(data_1_gda['y1']))
print('Time average of x_22:', np.mean(data_1_gda['y2']))


# In[8]:


x = data_1_gda['x1']
y = data_1_gda['y1']
z = data_1_gda['periodic']


fig = go.Figure([go.Scatter3d(mode='markers', x=x, y=y, z=z, marker=dict(size=2,color=z, colorscale='agsunset'),showlegend=False)
                ])

fig.update_layout(template="seaborn",
                  title='GDA with Periodically Rescaled Game',
                   font=dict(size=15),
                   scene=dict(xaxis_title='x_11',
                              yaxis_title='x_21',
                              zaxis_title='periodic function',
                              aspectratio = dict(x=1, y=1, z=0.7),))


# We rescale the off diagonal elements of the payoff matrices. We see that the gradient descent trajectories converge in time average to 0.

# In[9]:


data_2_gda = MPTrajectory(f=MPderivGDA, s = [2.35, 0.65, 1.65, 0.35], scale=0.8, numperiod=1000, numstep=50000, func=f_modified_sin)


# In[10]:


print('Time average of x_11:', np.mean(data_2_gda['x1']))
print('Time average of x_12:', np.mean(data_2_gda['x2']))
print('Time average of x_21:', np.mean(data_2_gda['y1']))
print('Time average of x_22:', np.mean(data_2_gda['y2']))


# In[11]:


x = data_2_gda['x1']
y = data_2_gda['y1']
z = data_2_gda['periodic']
color = data_2_gda['periodic']

fig = go.Figure([go.Scatter3d(mode='markers', x=x, y=y, z=z, marker=dict(size=2,color=color, colorscale='agsunset'),showlegend=False)
                ])

fig.update_layout(template="seaborn",
                  title='GDA with Periodically Rescaled Game (Off Diagonal Scale = 0.8)',
                  autosize=False,
                  width=1000,
                  height=1000,
                  font=dict(size=15),
                  scene=dict(xaxis_title='x_11',
                              yaxis_title='x_21',
                              zaxis_title='periodic function',
                              aspectratio = dict(x=1, y=1, z=0.7),))


# ### Time Averages

# In[12]:


P = np.matrix([ [1, -1],
                [-1, 1] ])


# In[13]:


def getTimeAverageData(num_init=1, f=MPderivGDA, numperiod=100, numstep=2000, scale=1, plot=False, 
                       func=np.sin, util=[]):
    x_regret = []
    y_regret = []

    x_regret_loop = []
    y_regret_loop = []


#     num_init = 10
    entropies = []

#     N=10
    x1_inits = np.linspace(0.1,0.7, num_init)
    x = np.asarray([[0.75-x1, 0.25] for x1 in x1_inits])
    y = np.random.rand(num_init,2)
    initial_conditions=np.hstack((x,y)) #.append([x,w])
    yutilities=[]
    xutilities=[]
    xvalstime=[]
    yvals1=[];  yvals2=[];
    xvals1=[];  xvals2=[];
    xutilvals=[];
    yutilvals=[];
    for x0 in initial_conditions:
        x=x0[0:2]
        y=x0[2:4]

        x = x/x.sum()
        y = y/y.sum()
        vec = np.concatenate([x, y])

        print('Initial Conditions: ', vec)
        mpdata = MPTrajectory(f=f, s = vec, scale=scale, numperiod=numperiod, numstep=numstep, func=func, plot=plot)
        A = np.eye(2)
        B = -np.eye(2)
        A = lambda i: 1*np.array([[func(i), -func(i)*scale],
                        [-func(i)*scale, func(i)]])
        B = lambda i: -1*np.array([[func(i), -func(i)*scale],
                        [-func(i)*scale, func(i)]])
        times = np.linspace(0,numperiod*2*np.pi,numstep)
        regret_x = []
        utils_x=[]

        x1s=[]; x2s=[]; # local storage
        for i in range(len(mpdata['x1'])):
            utility = (np.array([mpdata['x1'][i], mpdata['x2'][i]])@A(times[i])@np.array([mpdata['y1'][i], mpdata['y2'][i]]))
            xval = np.array([mpdata['x1'][i], mpdata['x2'][i]])
            utils_x.append(utility)
            x1s.append(xval[0])
            x2s.append(xval[1])
        xvals1.append(x1s)
        xvals2.append(x2s)
        xutilvals.append(utils_x)

        regret_y = []
        utils_y=[]

        y1s=[]; y2s=[];
        for i in range(len(mpdata['y1'])):
            utility = (np.array([mpdata['x1'][i], mpdata['x2'][i]])@B(times[i])@np.array([mpdata['y1'][i], mpdata['y2'][i]])) 
            utils_y.append(utility)
            yval = np.array([mpdata['y1'][i], mpdata['y2'][i]])
            y1s.append(yval[0])
            y2s.append(yval[1])
        yvals1.append(y1s)
        yvals2.append(y2s)
        yutilvals.append(utils_y)
        

    cumsums=[]
    for utilval in xutilvals:
        cumsums.append(np.cumsum(utilval)/np.arange(1, len(utilval)+1))

    cumsumsy=[]
    for utilval in yutilvals:
        cumsumsy.append(np.cumsum(utilval)/np.arange(1, len(utilval)+1))

    cxvals1=[]
    for xvalit in xvals1:
        cxvals1.append(np.cumsum(xvalit)/np.arange(1, len(xvalit)+1))

    cxvals2=[]
    for xvalit in xvals2:
        cxvals2.append(np.cumsum(xvalit)/np.arange(1, len(xvalit)+1))

    cyvals1=[]
    for utilval in yvals1:
        cyvals1.append(np.cumsum(utilval)/np.arange(1, len(utilval)+1))

    cyvals2=[]
    for utilval in yvals2:
        cyvals2.append(np.cumsum(utilval)/np.arange(1, len(utilval)+1))
    
    data = {}
    data['cumsums'] = cumsums
    data['cumsumsy'] = cumsumsy
    data['cxvals1'] = cxvals1
    data['cxvals2'] = cxvals2
    data['cyvals1'] = cyvals1
    data['cyvals2'] = cyvals2
    return data


# In[14]:


def plotTimeAverageUtility(data):
    uxmeans=np.mean(np.asarray(data['cumsums']),axis=0)
    uxvars=np.std(np.asarray(data['cumsums']),axis=0)

    uymeans=np.mean(np.asarray(data['cumsumsy']),axis=0)
    uyvars=np.std(np.asarray(data['cumsumsy']),axis=0)

    plt.figure(figsize=(10,8));
    plt.plot(uxmeans, color='tab:red', linewidth=3, label=r'$\hat{u}_{x1}(t)$');
#     plt.plot(uxmeans+uxvars,color='tab:blue', label=r'$\pm 1$std');
#     plt.plot(uxmeans-uxvars,color='tab:blue');

    plt.fill_between(np.arange(0,len(data['cumsums'][0])),uxmeans-uxvars,uxmeans+uxvars,color='tab:blue', alpha=0.5)

    plt.tick_params(labelsize=22)
    plt.legend(fontsize=22)
    plt.title(r'Time Average Utility ($x_1$-player)', fontsize=22);
    
    plt.figure(figsize=(10,8));
    plt.plot(uymeans, color='tab:red', linewidth=3, label=r'$\hat{u}_{x2}(t)$');
#     plt.plot(uymeans+uyvars,color='tab:blue', label=r'$\pm 1$std');
#     plt.plot(uymeans-uyvars,color='tab:blue');

    plt.fill_between(np.arange(0,len(data['cumsumsy'][0])),uymeans-uyvars,uymeans+uyvars,color='tab:blue', alpha=0.5)

    plt.tick_params(labelsize=22)
    plt.legend(fontsize=22)
    plt.title(r'Time Average Utility ($x_2$-player)', fontsize=22);
    
    return


# In[15]:


def plotTimeAverageActions(data):
    xmeans1=np.mean(np.asarray(data['cxvals1']),axis=0)
    xvars1=np.std(np.asarray(data['cxvals1']),axis=0)
    xmeans2=np.mean(np.asarray(data['cxvals2']),axis=0)
    xvars2=np.std(np.asarray(data['cxvals2']),axis=0)
    plt.figure(figsize=(10,8))

    plt.plot(xmeans1, color='tab:red', linewidth=3, label=r'$\hat{x}_{11}(t)$')
    plt.plot(xmeans2, color='tab:blue', linewidth=3, label=r'$\hat{x}_{12}(t)$')

    plt.tick_params(labelsize=22)
    plt.legend(fontsize=22)
    plt.title(r'Time Average Actions ($x_1$-player)', fontsize=22);
    
    ymeans1=np.mean(np.asarray(data['cyvals1']),axis=0)
    yvars1=np.std(np.asarray(data['cyvals1']),axis=0)
    ymeans2=np.mean(np.asarray(data['cyvals2']),axis=0)
    yvars2=np.std(np.asarray(data['cyvals2']),axis=0)

    plt.figure(figsize=(10,8));

    plt.plot(ymeans1, color='tab:red', linewidth=3, label=r'$\hat{x}_{21}(t)$');
    plt.plot(ymeans2, color='tab:blue', linewidth=3, label=r'$\hat{x}_{22}(t)$');

    plt.tick_params(labelsize=22)
    plt.legend(fontsize=22)
    plt.title(r'Time Average Actions ($x_2$-player)', fontsize=22);
    
    return


# #### Matching Pennies game with periodic rescaling

# In[16]:


dataGDAtimeavg1 = getTimeAverageData(num_init=1, f=MPderivGDA, numperiod=100, numstep=1000, scale=1, func=f_modified_sin)


# In[17]:


plotTimeAverageUtility(dataGDAtimeavg1)


# In[18]:


plotTimeAverageActions(dataGDAtimeavg1)


# #### Rescaled off diagonal Matching Pennies

# In[19]:


dataGDAtimeavg2 = getTimeAverageData(num_init=1, f=MPderivGDA, numperiod=500, numstep=5000, scale=0.8, func=f_modified_sin)


# In[20]:


plotTimeAverageUtility(dataGDAtimeavg2)


# In[21]:


plotTimeAverageActions(dataGDAtimeavg2)


# ### Piecewise function counterexample

# In[22]:


def f_piecewise(x):
    test = np.piecewise(x, [x%(3*np.pi) < np.pi, np.pi <= x%(3*np.pi) < 3*np.pi/2, 3*np.pi/2 <= x%(3*np.pi) <= 3*np.pi], [-1, 1, -1])
    return test.item()


# In[23]:


plt.plot([f_piecewise(x) for x in np.linspace(0,6*np.pi, 10000)])


# In[24]:


data_3_gda = MPTrajectory(f=MPderivGDA, s = [2.35, 0.65, 1.65, 0.35], scale=1, numperiod=100, numstep=10000, func=f_piecewise)


# In[25]:


dataGDAtimeavg3 = getTimeAverageData(num_init=1, f=MPderivGDA, numperiod=100, numstep=1000, scale=1, func=f_piecewise)


# In[26]:


plotTimeAverageUtility(dataGDAtimeavg3)


# In[27]:


plotTimeAverageActions(dataGDAtimeavg3)


# ## Replicator Dynamics - Matching Pennies

# In[28]:


def MPderivRD(s,t, scale, func, util):    
    x = np.array([[s[0]], [s[1]]])
    y = np.array([[s[2]], [s[3]]])
    
    A = lambda i: 1*np.array([[func(i), -func(i)*scale],
                        [-func(i)*scale, func(i)]])
    B = lambda i: -1*np.array([[func(i), -func(i)*scale],
                        [-func(i)*scale, func(i)]])

    dxdt = np.multiply(x,A(t)@y-x.T@A(t)@y).flatten()
    dydt = np.multiply(y,B(t).T@x-x.T@B(t).T@y).flatten()
    util.append(x.T@A(t)@y)
    return np.concatenate((dxdt,dydt))


# In[29]:


data_1_rd = MPTrajectory(f=MPderivRD, s = [0.45, 0.55, 0.55, 0.45], scale=1, numperiod=100, numstep=5000, func=f_modified_sin)


# In[30]:


x = data_1_rd['x1']
y = data_1_rd['y1']
z = data_1_rd['periodic']

# p1=[1, 0, 0, 1]
# p2=[0, 1, 0, 0]
# p3=[0, 0, 1, 0]

fig = go.Figure([go.Scatter3d(mode='markers', x=x, y=y, z=z, marker=dict(size=2,color=z, colorscale='agsunset'),showlegend=False)
#                  go.Scatter3d(mode='lines', x=p1, y=p2, z=p3, line=dict(color='black', width=2),showlegend=False)
                ])

fig.update_layout(template="seaborn",
                  title='Replicator with Periodically Rescaled Game',
                  autosize=False,
                  width=1000,
                  height=1000,
                   font=dict(size=15),
                   scene=dict(xaxis_title='x_11',
                              yaxis_title='x_21',
                              zaxis_title='periodic function',
                              aspectratio = dict(x=1, y=1, z=0.7),))


# In[31]:


data_x = np.array([data_1_rd[i] for i in ['x1', 'x2']]).T
data_y = np.array([data_1_rd[i] for i in ['y1', 'y2']]).T


# In[32]:


div_x = []
for i in data_x:
    kl_div_x = entropy([1/2, 1/2], qk=i)
    div_x.append(kl_div_x)

x_weighted = [x for x in div_x]

div_y = []
for i in data_y:
    kl_div_y = entropy([1/2, 1/2], qk=i)
    div_y.append(kl_div_y)

y_weighted = [y for y in div_y]

div_combined = np.add(x_weighted, y_weighted)


# In[33]:


fig = go.Figure([go.Scatter(y=y_weighted[:1000],
                    mode='lines', line=dict(width=0.5, color='#4a69bb'),
                    name='Weighted x_1', fill='tozeroy'), 
                 go.Scatter(y=div_combined[:1000],
                    mode='lines',
                    name='Weighted x_2', line=dict(width=0.5, color='#6ece58'), fill='tonexty'),
                 go.Scatter(y=div_combined[:1000],
                    mode='lines',
                    name='Sum of Divergences', line = dict(width = 3, color='#440154'), opacity=1)
                ])

# Edit layout
fig.update_layout(title='Constant of Motion for 2-player MP game',
                  xaxis_title='Time Steps',
                  yaxis_title='KL-Divergence',
                  legend_orientation='h', 
                  legend=dict( y=-0.2),
                  font=dict(size=15))


# ### Time Averages

# In[34]:


P = np.matrix([ [1, -1],
                [-1, 1] ])


# In[35]:


dataRDtimeavg1 = getTimeAverageData(num_init=1,f=MPderivRD, numperiod=500, numstep=5000, scale=1, func=f_modified_sin)


# In[36]:


plotTimeAverageUtility(dataRDtimeavg1)


# In[37]:


plotTimeAverageActions(dataRDtimeavg1)


# In[38]:


dataRDtimeavg2 = getTimeAverageData(num_init=1, f=MPderivRD, numperiod=100, numstep=1000, scale=1, func=np.sin)


# In[39]:


plotTimeAverageUtility(dataRDtimeavg2)


# In[40]:


plotTimeAverageActions(dataRDtimeavg2)


# ## Large Scale Simulation

# In[41]:


im = Image.open('ghostBW.jpg','r') # Can be many different formats.
pix = im.load()
print (im.size)  # Get the width and hight of the image for iterating over
print (pix[0,0])
plt.imshow(im)
plt.show()


# In[42]:


sigmoid = lambda x: 1/(1 + np.exp(-5*(x-0.5)))


# In[43]:


sig_color_list = []
for j in range(8):
    for i in range(8):
        rgb_val = pix[(75+i*150, 75+j*150)]
        if rgb_val[1] == 0:
            bw_val = 0.5
        elif rgb_val[1] == 255:
            bw_val = 0.515
        elif rgb_val[0]/255 > 0.5:
            bw_val = 0.51
        else:
            bw_val = 0.49
        sig_color_list.append(bw_val)


# In[44]:


sig_color_list = [sigmoid(x) for x in sig_color_list]
s_sig = []
for i in sig_color_list:
    init = [i, 1-i]
    s_sig.append(init)
s_sig = np.array(s_sig)


# In[45]:


plt.rcParams["axes.grid"] = False
plt.imshow(np.array(sig_color_list).reshape(8,8), cmap='inferno')


# In[46]:


def MPDerivNNodes(s,t, scale, func, N, graph, sin_scale):
    A = lambda i: 1*np.array([[func(i), -func(i)*scale],
                        [-func(i)*scale, func(i)]])
    B = lambda i: -1*np.array([[func(i), -func(i)*scale],
                        [-func(i)*scale, func(i)]])
    vals = s.reshape(N, 2)
    
    payoffs = []
    for i in range(N):
        payoffs.append(A(t))

    utils = []
    ddt=np.zeros(2*N)
    for i in range(N):
        for j in range(N):
            if graph[i][j] != 0:
                x=vals[i]
                y=vals[j]
                dxdt = np.multiply(x,(sin_scale[i]*payoffs[i])@y-x.T@(sin_scale[i]*payoffs[i])@y).flatten()
                ddt[2*i:2*i+2] += dxdt
                dydt = np.multiply(y,(-sin_scale[i]*payoffs[j])@x-x.T@(-sin_scale[i]*payoffs[j].T)@y).flatten()
                ddt[2*j:2*j+2] += dydt
    return np.array(ddt).flatten()


# In[47]:


def MPTrajectoryNNodes(N, graph, f=MPDerivNNodes, s=np.array([0.45, 0.55, 0.55, 0.45]), numperiod=10, numstep=2000, 
                       scale=1, plot=True, func=np.sin, sin_scale=np.ones(64)):
    x = s.flatten()
    partuple=(scale, func, N, graph, sin_scale)
    tvals=np.linspace(0,numperiod*2*np.pi,numstep)
    func_tvals = [func(x) for x in tvals]
    traj=odeint(f,x,tvals, partuple)

    return traj.reshape(numstep, N, 2)


# In[48]:


def GetEntropyVals(data, mu):
    x_weighted = []
    weight = 1
    for i in (range(data.shape[1])):
        div_x = []
        for j in data[:,i]:
            kl_div_x = entropy([1/2, 1/2], qk=j)
            div_x.append(kl_div_x)
        x_weighted.append(np.array([weight*x for x in div_x]))
        weight = weight*mu[i]
    return x_weighted

def PlotEntropyVals(entropy_vals):
    cumsum = 0
    lines=[go.Scatter(y=entropy_vals[0], mode='lines', line=dict(width=0.5), fill='tozeroy')]
    for i in range(len(entropy_vals)):
        cumsum += entropy_vals[i]
        lines.append(go.Scatter(y=cumsum, mode='lines', line=dict(width=0.5), fill='tonexty'))
    lines.append(go.Scatter(y=cumsum, mode='lines',line = dict(width = 3, color='#440154'),opacity=1))
    fig = go.Figure(lines)

    # Edit layout
    fig.update_layout(title='KL-Divergence for {}-node polymatrix game'.format(len(entropy_vals)),
                      xaxis_title='Time Steps',
                      yaxis_title='KL-Divergence', 
                      # legend_orientation='h', 
                      # legend=dict( y=-0.2),
                      font=dict(size=15))
    return fig


# ### Create time invariant plot 
# Note that we reduce the number of simulation steps from 50000 to 500 to reduce the data size. Full sized plot can be found in the paper

# In[49]:


graph1 = np.zeros((64, 64))
graph1[0][1] = 1


# In[50]:


for i in (range(1,63)):
    graph1[i][i+1] = 1


# In[51]:


sin_scale_rand = 3*np.random.rand(64)


# In[52]:


data_64node = MPTrajectoryNNodes(s=s_sig,numperiod=10, numstep=500, f=MPDerivNNodes, scale=0.8, 
                                 func=f_modified_sin, N=64, graph=graph1, sin_scale=sin_scale_rand)


# In[53]:


PlotEntropyVals(GetEntropyVals(data_64node, np.ones(64)))


# Use a sparser graph without randomized periodic functions, to reduce number of iterations needed to achieve recurrence.

# In[54]:


graph2 = np.zeros((64, 64))
graph2[0][1] = 1
for i in (range(1,63)):
    if i%2==0:
        graph2[i][i+1] = 1


# In[55]:


data_64node_simplified = MPTrajectoryNNodes(s=s_sig,numperiod=50, numstep=10000, f=MPDerivNNodes, scale=0.8, 
                                 func=f_modified_sin, N=64, graph=graph2)


# In[56]:


PlotEntropyVals(GetEntropyVals(data_64node_simplified, np.ones(64)))


# In[57]:


def save_img(data, name='clyde.png'):
    
    new_data = np.zeros(np.array(data.shape) * 100)

    for j in range(data.shape[0]):
        for k in range(data.shape[1]):
            new_data[j * 100: (j+1) * 100, k * 100: (k+1) * 100] = data[j, k]

    plt.imsave('clyde/'+name, new_data, cmap='inferno')


# In[58]:


fps = 60
nSeconds = 10

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure( figsize=(8,8) )

a = (sigmoid(data_64node_simplified[0][:,0].reshape(8,8)))
im = plt.imshow(a, cmap='inferno')
plt.axis('off')
def animate_func(i):
    if i % fps == 0:
        print( '.', end ='' )
    im.set_array(sigmoid(data_64node_simplified[i][:,0].reshape(8,8)))
    plt.axis('off')
    return [im]

anim = animation.FuncAnimation(
                               fig, 
                               animate_func, 
                               frames = nSeconds * fps,
                               interval = 1000 / fps, # in ms
                               )

# Uncomment to save the resulting animation as an mp4 file
# anim.save('clyde_animation3.mp4', fps=fps, extra_args=['-vcodec', 'libx264'])

print('Done!')


# First 600 iterations of the simplified polymatrix simulation. Simulation takes about 6226 iterations to recur.

# In[59]:


HTML(anim.to_html5_video())

