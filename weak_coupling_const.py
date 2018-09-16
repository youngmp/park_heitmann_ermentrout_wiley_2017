# generate long movie of weakly coupled traub models
# by default, the data files correspond to figure 8 of the paper.
# starttime sets the time at which you start saving frames.

# avconv -r 10 -start_number 1 -i test%d.jpg -b:v 1000k test.mp4


import numpy as np

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

from xppcall import xpprun,read_pars_values_from_file,read_init_values_from_file

x_label = [r"$0$", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"]

from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif', serif=['Computer Modern Roman'])

#from matplotlib import rcParams

#matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
#matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{bm}"]

matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath \usepackage{bm} \usepackage{xcolor} \definecolor{blue1}{HTML}{3399FF}']
matplotlib.rcParams.update({'figure.autolayout': True})

sizeOfFont = 20
fontProperties = {'weight' : 'bold', 'size' : sizeOfFont}

lamomfsize=15 #lambda omega figure size


import euler
import phase_model



movdir = 'mov/' # save frames to this dir
#starttime = 7666 # starting time in ms
#endtime = 7966 # starting time in ms


## TII numerical phase diff
ml_mem2_prc2_t = np.loadtxt("ml_mem2_prc2_vt1_t20k_eps.0025.dat")[:,0]
t = ml_mem2_prc2_t
T = t[-1]
ml_mem2_prc2_vt1 = np.loadtxt("ml_mem2_prc2_vt1_t20k_eps.0025.dat")
ml_mem2_prc2_wt1 = np.loadtxt("ml_mem2_prc2_wt1_t20k_eps.0025.dat")
ml_mem2_prc2_vt2 = np.loadtxt("ml_mem2_prc2_vt2_t20k_eps.0025.dat")
ml_mem2_prc2_wt2 = np.loadtxt("ml_mem2_prc2_wt2_t20k_eps.0025.dat")

if False:
    plt.figure()
    plt.plot(t,ml_mem2_prc2_vt1[:,1])
    plt.show()
# reference model data
#refT2 = np.loadtxt("ml_mem2_prc2_LCv_full.tab")[2]

mem2v = np.loadtxt("ml_mem2_prc2_LCv_full.tab")
mem2w = np.loadtxt("ml_mem2_prc2_LCw_full.tab")
ml_mem2_prc2_vtref = mem2v[3:]
ml_mem2_prc2_wtref = mem2w[3:]
refT2 = mem2v[2]#114.55
    

# poincare section: will be at max v, corresponding w, and v > 0
max_v_idx = np.argmax(ml_mem2_prc2_vtref) # max v index
max_w = ml_mem2_prc2_wtref[max_v_idx] # corresponding w value
    
# crossing index of vt1 and vt2
crossing_idx1 = (ml_mem2_prc2_wt1[:,1][1:]>max_w)*(ml_mem2_prc2_wt1[:,1][:-1]<max_w)*(ml_mem2_prc2_vt1[:,1][1:]>0)
# remove first crossing time (since v starts on spike)
crossing_idx1 = crossing_idx1[1:]
crossing_idx2 = (ml_mem2_prc2_wt2[:,1][1:]>max_w)*(ml_mem2_prc2_wt2[:,1][:-1]<max_w)*(ml_mem2_prc2_vt2[:,1][1:]>0)

# match number of crossings
min_idx = np.amin([sum(crossing_idx1),sum(crossing_idx2)])
t1_crossing2 = ml_mem2_prc2_t[:-2][crossing_idx1][:min_idx]
t2_crossing2 = ml_mem2_prc2_t[:-1][crossing_idx2][:min_idx]
print 'total crossing times', sum(crossing_idx1),sum(crossing_idx2)

#prc = np.mod(np.mod(t2_crossing,T)/T - np.mod(t1_crossing,T)/T+.5,1)-.5
prc2_u = t2_crossing2 - t1_crossing2
prc2 = np.mod((prc2_u)/refT2+.5,1)-.5



ml_mem2_prc2_psi_theory = np.loadtxt('ml_phs2ee.dat')


skipn = 100
counter = 0
j = 0

while j < (len(t)):

    fig = plt.figure(figsize=(5,5))

    ax11 = plt.subplot2grid((2,2),(0,0),colspan=2)
    ax21 = plt.subplot2grid((2,2),(1,0),colspan=2)
    #ax21b = ax21.twinx()

    ax11.set_title(r"\textbf{Weakly Coupled Oscillators}")
    #ax12.set_title(r"\textbf{Oscillator 2}")
    ax21.set_title(r"\textbf{Phase Difference}")

    ax11.set_xticks([])
    ax11.set_yticks([])
    #ax11.set_title('Weakly Coupled Oscillators')

    ax11.text(-3,1.5,r'\textbf{Neuron 1}')
    ax11.text(2.,1.5,r'\textbf{Neuron 2}')

    # right to left
    ax11.annotate('',xy=(2.5-np.sqrt(2)/2,np.sqrt(2)/2),xytext=(-2.5+np.sqrt(2)/2,np.sqrt(2)/2),
                  size=40,
        arrowprops=dict(arrowstyle='simple',
                        shrinkA=5,
                        shrinkB=5,
                        fc="k", ec="k",                        
                        connectionstyle="arc3,rad=-.3")
    )

    # left to right

    ax11.annotate('',xytext=(2.5-np.sqrt(2)/2,-np.sqrt(2)/2),xy=(-2.5+np.sqrt(2)/2,-np.sqrt(2)/2),
                  size=40,
        arrowprops=dict(arrowstyle='simple',
                        shrinkA=5,
                        shrinkB=5,
                        fc="k", ec="k",
                        connectionstyle="arc3,rad=-.3")
    )

    
    ax21.set_xlabel(r"$\mathbf{t (ms)}$",fontsize=15)
    ax21.set_ylabel(r"$\mathbf{\phi}$",fontsize=15)
    #ax21b.set_ylabel(r'$\bm{q(t)}$',fontsize=15,color='red')

    ax11.set_xlim(-4,4)
    ax11.set_ylim(-2,2)
    #ax12.set_xlim(vmin,vmax)
    
    ax21.set_xlim(t[0],t[-1])
    #ax21b.set_xlim(t[0],t[-1])

    #ax21.set_yticks(np.arange(0,0.5+.125,.125)*2*np.pi)
    #ax21.set_yticklabels(x_label, fontsize=15)
    ax21.set_ylim(-.01,np.amax(ml_mem2_prc2_psi_theory[:int(50/.01),1]*112)+5)

    #ax21.set_ylim(0,1)
    ax21.set_xlim(0,T)

    j += skipn
    k = j

    # plot neuron shapes
    print ml_mem2_prc2_vt1[k,1],t[k]
    if ml_mem2_prc2_vt1[k,1] > 0:
        fc1 = 'white'
    else:
        fc1 = 'black'

    if ml_mem2_prc2_vt2[k,1] > 0:
        fc2 = 'white'
    else:
        fc2 = 'black'
        
    circle1 = Circle((-2.5,0), 1, edgecolor='k',facecolor=fc1)
    circle2 = Circle(( 2.5,0), 1, ec='k',fc=fc2)

    #ax11.plot()
    #p = PatchCollection([circle1,circle2])
    #ax11.add_collection(p)
    ax11.add_artist(circle1)
    ax11.add_artist(circle2)
    

    # find index in numerical spike estimation
    numk = np.argmin(np.abs(t[k]-t1_crossing2))
    ax21.scatter(t1_crossing2[:numk],np.abs(prc2)[:numk]*112,color='k',zorder=3,label='Numerics')

    ax21.plot(ml_mem2_prc2_psi_theory[:int(50/.01),0]/.0025,
              ml_mem2_prc2_psi_theory[:int(50/.01),1]*112,
              lw=5,color="#3399ff",label='Theory')

    

    #ax21b.plot(t[:k][::skipn2],gm[:k][::skipn2],color='red',lw=2,label='Parameter')
    ax21.legend()
    
    fig.savefig(movdir+str(counter)+".png",dpi=150)
    
    #plt.pause(.01)
    #print(t[k],k,counter)


    plt.cla()
    plt.close()

    counter += 1




plt.show()
