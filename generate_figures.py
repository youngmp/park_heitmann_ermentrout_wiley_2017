
import ml
from lib_ml import *
import numpy as np
import numpy.linalg as linalg

import matplotlib
from matplotlib import pyplot as plt
import matplotlib.pylab as mp
from matplotlib import rc
import matplotlib.ticker as mticker

rc('text', usetex=True)
rc('font', family='serif', serif=['Computer Modern Roman'],size=20)
matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath \usepackage{bm}']

cos = np.cos
sin = np.sin
pi = np.pi



def phase_plane(raw=False):
    """
    Voltage over time + phase + phase plane
    raw: set true to plot only the limit cycle (not yet implemented)
    """
    #fig = mp.figure()

    ## create fig 1 canvas (single figure) # might to account for no colors
    fig = plt.figure(1,figsize=(10,7.5)) # width, height


    iapp = 55.# SNIC bifufcation at 40.8
    T, init, err, lc_max = ml_limit_cycle(dy_dt_SNIC, [0,.1], iapp=iapp)
    dt = .01 # time step
    numcycle = 4
    tot = T*numcycle # total integration time
    steps = int(tot/dt)+1 # total steps
    t = np.linspace(0,tot,steps)
    sol = integrate.odeint(ml.ml_rhs, lc_max, t, args=(iapp,))
    #sol = integrate.odeint(lamom, [0,1], t)
    lcsol = sol[:int(T/dt),:]


    ax1a = plt.subplot2grid((3,4),(0,0),colspan=2) # top left
    ax1b = plt.subplot2grid((3,4),(0,2),colspan=2) # bot left
    ax1c = plt.subplot2grid((3,4),(1,0),colspan=2,rowspan=2) # top right
    ax1d = plt.subplot2grid((3,4),(1,2),colspan=2,rowspan=2) # top right

    # first tight layout command to fit axis tick labels
    #plt.tight_layout()
    #plt.figure(1,figsize=(8,4)) # width, height
    #plt.plot(t,sol[:,0],lw=5);plt.title('Fig1a')
    #plt.savefig('fig1a.png')


    # limit cycle solution over time
    ax1a.plot(t,sol[:,0],lw=4,color='black');
    myLocatora = mticker.MultipleLocator(20)
    ax1a.yaxis.set_major_locator(myLocatora)
    ax1a.set_xlabel(r'\textbf{t (ms)}')
    ax1a.set_ylabel(r'\textbf{V}')
    ax1a.set_xlim(0,t[-1])
    
    #ax1a.set_xticks([])
    ax1a.text(-60,40,r'\textbf{(a)}')
    #ax1a.set_title('Title 1')
    
    #plt.savefig('fig1a.ps')
    
    #plt.figure(2,figsize=(6,3)) # width, height
    #plt.plot(t,np.mod(t,T),lw=5);plt.title('Fig1b')
    #plt.savefig('fig1b.png')

    # phase model plot over time
    ax1b.plot(t,np.mod(t,T),lw=4,color='black')
    myLocatorb = mticker.MultipleLocator(20)
    ax1b.yaxis.set_major_locator(myLocatorb)
    #ax1b.set_title('Title 2')
    ax1b.set_xlabel(r'\textbf{t (ms)}')
    ax1b.set_ylabel(r'$\bm{\theta}$')
    ax1b.set_xlim(0,t[-1])
    ax1b.text(-60,70,r'\textbf{(b)}')
    #plt.savefig('fig1b.ps')
    
    #plt.figure(3,figsize=(6,6)) # width, height
    

    # nullclines
    """
    vrange = np.linspace(np.amin(lcsol[:,0])-15,np.amax(lcsol[:,0])+5,300)
    nrange = np.linspace(np.amin(lcsol[:,1])-.1,np.amax(lcsol[:,1])+.05,300)
    ax1c.set_xlim(vrange[0],vrange[-1])
    ax1c.set_ylim(nrange[0],nrange[-1])
    #plt.xlim(vrange[0],vrange[-1])
    #plt.ylim(nrange[0],nrange[-1])
    #plt.plot(vrange,nullcline1(vrange,iapp),lw=3,ls='dashed',dashes=(3,1))
    #plt.plot(vrange,nullcline2(vrange,iapp),lw=3,ls='dashed',dashes=(3,1))
    ax1c.plot(vrange,nullcline1(vrange,iapp),lw=3,ls='dashed',dashes=(3,1))
    ax1c.plot(vrange,nullcline2(vrange,iapp),lw=3,ls='dashed',dashes=(3,1))
    """

    # fixed pts
    """
    saddle = saddle_point(iapp)
    ax1c.scatter(saddle[0],saddle[1],facecolor='white',edgecolor='black',s=45,zorder=10)
    #plt.scatter(saddle[0],saddle[1],facecolor='white',edgecolor='black',s=50,zorder=10)
    u_foc = unstable_focus(iapp)
    ax1c.scatter(u_foc[0],u_foc[1],facecolor='white',edgecolor='black',s=45,zorder=10)
    #plt.scatter(u_foc[0],u_foc[1],facecolor='white',edgecolor='black',s=50,zorder=10)
    """

    # limit cycle phase plane
    ax1c.plot(lcsol[:,0],lcsol[:,1],lw=3,color='black')
    k = int(.92*len(lcsol[:,0]))
    vecx = -(lcsol[k,0] - lcsol[k+10,0])
    vecy = -(lcsol[k,1] - lcsol[k+10,1])
    sf = 1

    #ax1c.arrow(lcsol[k,0],lcsol[k,1],vecx*sf,vecy*sf,head_width=.05,head_length=5,fc='k',ec='k')
    ax1c.annotate('',(lcsol[k+50,0],lcsol[k+50,1]),(lcsol[k,0],lcsol[k,1]),
                  arrowprops=dict(facecolor='black',shrink=0,
                                  headwidth=15,
                                  frac=1,
                              ))


    #ax1c.set_title('Title 3')    
    #plt.plot(lcsol[:,0],lcsol[:,1],lw=5);plt.title('Fig1c')

    # add time ticks
    # get spacing
    # divide period T into 'div' even intervals
    div = 20.
    interval = T/div
    d_idx = int(interval/dt)
    
    # for each interval, mark with circle
    k = 0
    color = 0.
    while k < T/dt-d_idx:
        color = k/(1.*T/dt)
        ax1c.scatter(lcsol[k,0],lcsol[k,1],
                     facecolor=str(color),
                     edgecolor='black',
                     s=90,
                     zorder=10)
        k += d_idx

    


    # separatrix
    """
    sepu1,sepu2 = separatrix(iapp)
    ax1c.plot(sepu1[:,0],sepu1[:,1],color='teal',lw=5)
    ax1c.plot(sepu2[:,0],sepu2[:,1],color='teal',lw=5)
    #plt.plot(sepu1[:,0],sepu1[:,1],color='teal',lw=5)
    #plt.plot(sepu2[:,0],sepu2[:,1],color='teal',lw=5)
    #plt.savefig('fig1c.ps')
    ax1c.set_xlabel('V')
    ax1c.set_ylabel('w')
    """
    # fix x axis label spacing
    myLocatorc = mticker.MultipleLocator(20)
    ax1c.xaxis.set_major_locator(myLocatorc)
    ax1c.set_xlabel(r'\textbf{V}')
    ax1c.set_ylabel(r'\textbf{w}')
    ax1c.text(-75,0.5,r'\textbf{(c)}')


    # phase model
    # draw unit circle with same time spacing as before
    Nth = 500
    Tth = 2*np.pi
    theta = np.linspace(0,Tth,Nth)
    dth = Tth/Nth
    #ax1d.plot([0,1],[0,1])
    
    # define phase model solution
    phase_model=np.zeros((Nth,2))
    phase_model[:,0]=np.cos(theta);phase_model[:,1]=np.sin(theta)

    ax1d.plot(phase_model[:,0],phase_model[:,1],lw=3,color='black')
    intervalth = Tth/div
    dth_idx = int(intervalth/dth)
    
    # for each interval, mark with circle
    k = 0
    color = 0.
    while k < Tth/dth:
        color = k/(1.*Tth/dth)
        ax1d.scatter(phase_model[k,0],phase_model[k,1],
                     facecolor=str(color),
                     edgecolor='black',
                     s=90,
                     zorder=10)
        k += dth_idx

    # fix axis tick label
    myLocatord = mticker.MultipleLocator(0.75)
    ax1d.xaxis.set_major_locator(myLocatord)
    ax1d.text(-2,1.5,r'\textbf{(d)}')
    
    # add arrow at some phase value
    k2 = int(.92*len(phase_model[:,0]))
    vecx = -(phase_model[k2,0] - phase_model[k2+1,0])
    vecy = -(phase_model[k2,1] - phase_model[k2+1,1])
    sf = 1

    
    #ax1d.arrow(phase_model[k2,0],phase_model[k2,1],vecx*sf,vecy*sf,head_width=.2,head_length=.2,fc='k',ec='k')
    ax1d.annotate('',(phase_model[k2+10,0],phase_model[k2+10,1]),(phase_model[k2,0],phase_model[k2,1]),
                  arrowprops=dict(facecolor='black',shrink=0,
                                  headwidth=15,
                                  frac=1,
                              ))
    #ax1d.arrow(phase_model[k2,0],phase_model[k2,1],vecx*sf,vecy*sf,head_width=.2,head_length=.2,fc='k',ec='k')

    # lines to denote phase angle
    ax1d.plot([0,1],[0,0],ls='dashed',color='gray',lw=2)
    ax1d.plot([0,cos(pi/4)],[0,sin(pi/4)],color='black',lw=2)
    ax1d.text(.2,.03,r'$\theta$')
    #ax1d.text(0,0,)

    ax1d.set_xticks([])
    ax1d.set_yticks([])



    # second tight layout command to fit axis labels
    plt.tight_layout()
    # fill fig 1 canvas

    #plt.savefig('fig1c.png')

    

    return fig


def fi():
    """
    FI curve of ML
    drawn in scribus for now as fi_curves.sla
    """
    pass
    

def phase_diff():
    """
    comparison of phase difference: theory vs experiment
    """
    ## TI numerical phase diff
    fig = plt.figure(4,figsize=(10,5))
    # full model data
    ml_mem1_prc1_t = np.loadtxt("ml_mem1_prc1_vt1_t20k_eps.0025.dat")[:,0]
    ml_mem1_prc1_vt1 = np.loadtxt("ml_mem1_prc1_vt1_t20k_eps.0025.dat")
    ml_mem1_prc1_wt1 = np.loadtxt("ml_mem1_prc1_wt1_t20k_eps.0025.dat")
    ml_mem1_prc1_vt2 = np.loadtxt("ml_mem1_prc1_vt2_t20k_eps.0025.dat")
    ml_mem1_prc1_wt2 = np.loadtxt("ml_mem1_prc1_wt2_t20k_eps.0025.dat")

    # reference model data
    #refT1 = np.loadtxt("ml_mem1_prc1_LCv_full.tab")[2]
    mem1v = np.loadtxt("ml_mem1_prc1_LCv_full.tab")
    mem1w = np.loadtxt("ml_mem1_prc1_LCw_full.tab")
    ml_mem1_prc1_vtref = mem1v[3:]
    ml_mem1_prc1_wtref = mem1w[3:]
    refT1 = mem1v[2] #132.25
    
    
    # poincare section: will be at max v, corresponding w, and v > 0
    max_v_idx = np.argmax(ml_mem1_prc1_vtref) # max v index
    max_v = ml_mem1_prc1_vtref[max_v_idx] # max v value
    max_w = ml_mem1_prc1_wtref[max_v_idx] # corresponding w value
    
    # crossing index of vt1 and vt2
    crossing_idx1 = (ml_mem1_prc1_wt1[:,1][1:]>max_w)*(ml_mem1_prc1_wt1[:,1][:-1]<max_w)*(ml_mem1_prc1_vt1[:,1][1:]>0)
    # remove first crossing time (since v starts on spike)
    crossing_idx1 = crossing_idx1[1:] 
    crossing_idx2 = (ml_mem1_prc1_wt2[:,1][1:]>max_w)*(ml_mem1_prc1_wt2[:,1][:-1]<max_w)*(ml_mem1_prc1_vt2[:,1][1:]>0)

    # match number of crossings
    min_idx = np.amin([sum(crossing_idx1),sum(crossing_idx2)])
    t1_crossing1 = ml_mem1_prc1_t[crossing_idx1][:min_idx]
    t2_crossing1 = ml_mem1_prc1_t[crossing_idx2][:min_idx]

    #prc = np.mod(np.mod(t2_crossing,T)/T - np.mod(t1_crossing,T)/T+.5,1)-.5
    prc1_u = t2_crossing1 - t1_crossing1
    prc1 = np.mod((prc1_u)/refT1+.5,1)-.5    

    ## TII numerical phase diff
    ml_mem2_prc2_t = np.loadtxt("ml_mem2_prc2_vt1_t20k_eps.0025.dat")[:,0]
    ml_mem2_prc2_vt1 = np.loadtxt("ml_mem2_prc2_vt1_t20k_eps.0025.dat")
    ml_mem2_prc2_wt1 = np.loadtxt("ml_mem2_prc2_wt1_t20k_eps.0025.dat")
    ml_mem2_prc2_vt2 = np.loadtxt("ml_mem2_prc2_vt2_t20k_eps.0025.dat")
    ml_mem2_prc2_wt2 = np.loadtxt("ml_mem2_prc2_wt2_t20k_eps.0025.dat")

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
    t1_crossing2 = ml_mem2_prc2_t[crossing_idx1][:min_idx]
    t2_crossing2 = ml_mem2_prc2_t[crossing_idx2][:min_idx]
    print 'total crossing times', sum(crossing_idx1),sum(crossing_idx2)

    #prc = np.mod(np.mod(t2_crossing,T)/T - np.mod(t1_crossing,T)/T+.5,1)-.5
    prc2_u = t2_crossing2 - t1_crossing2
    prc2 = np.mod((prc2_u)/refT2+.5,1)-.5

    
    ml_mem1_prc1_psi_theory = np.loadtxt('ml_phs1ee.dat')
    plt.plot(ml_mem1_prc1_psi_theory[:int(50/.01),0]/.0025,
             ml_mem1_prc1_psi_theory[:int(50/.01),1],lw=5,color='k',ls='--',dashes=(10,4), label='Class 1 Exc. to Class 1 Exc. (Theory)')
    #plt.scatter(t2_crossing1[::7],np.abs(prc1)[::7],s=80,marker='D',facecolor='.9',edgecolor='k',zorder=3, label='Class 1 Exc. to Class 1 Exc. (Numerics)') # [::50] skips every 50th
    plt.plot(t2_crossing1[::7],np.abs(prc1)[::7],markersize=10,marker='D',color='.9',ls='None',zorder=3, label='Class 1 Exc. to Class 1 Exc. (Numerics)') # [::50] skips every 50th


    ml_mem2_prc2_psi_theory = np.loadtxt('ml_phs2ee.dat')
    plt.plot(ml_mem2_prc2_psi_theory[:int(50/.01),0]/.0025,
             ml_mem2_prc2_psi_theory[:int(50/.01),1],lw=5,color='k', label='Class 2 Exc. to Class 2 Exc. (Theory)')
    #plt.scatter(t1_crossing2[::7],np.abs(prc2)[::7],s=80,marker='s',facecolor='.9',edgecolor='k',zorder=3,label='Class 2 Exc. to Class 2 Exc. (Numerics)')
    plt.plot(t1_crossing2[::7],np.abs(prc2)[::7],markersize=10,marker='s',color='.9',ls='None',zorder=3,label='Class 2 Exc. to Class 2 Exc. (Numerics)')

    plt.ylim(-.02,.5)
    plt.xlim(0,20000)
    plt.legend(loc='right',fontsize='15')
    
    plt.ylabel(r'$\psi(t)$')
    plt.xlabel(r'$t$')

    plt.tight_layout()

    return fig

def prc():
    """
    generate PRCs from XPP-exported data.
    """
    prc1 = np.loadtxt('ml_mem1_prc_i43.5.dat')
    prc2 = np.loadtxt('ml_mem2_prc.dat')

    fig = plt.figure(figsize=(10,4)) # width height

    ax1 = plt.subplot2grid((1,2),(0,0))
    ax2 = plt.subplot2grid((1,2),(0,1))
    
    # force x-axis to be on [0,1]
    x1 = np.linspace(0,1,len(prc1[:,0]))
    x2 = np.linspace(0,1,len(prc2[:,0]))

    ax1.set_title(r'\textbf{(a)}',x=-.1,y=1.08)
    ax1.plot([0,x1[-1]],[0,0],ls='dashed',color='gray') # zero line
    ax1.plot(x1,prc1[:,1],lw=3,color='k')
    ax1.fill_between(x1,0,prc1[:,1],alpha=.5,color='gray')

    ax1.set_ylabel(r'$z(\theta)$')
    ax1.set_xlabel(r'$\bm{\theta}$')
    ax1.tick_params(direction='in', pad=10)
    ax1.set_xlim(0,x1[-1])

    ax2.set_title(r'\textbf{(b)}',x=-.1,y=1.08)
    ax2.plot([0,x2[-1]],[0,0],ls='dashed',color='gray') # zero line
    ax2.plot(x2,prc2[:,1],lw=3,color='k')
    ax2.fill_between(x2,0,prc2[:,1],alpha=.5,color='gray')

    ax2.set_xlabel(r'$\bm{\theta}$')
    ax2.tick_params(direction='in', pad=10)
    ax2.set_xlim(0,x2[-1])

    plt.tight_layout()
    return fig

def isochron_fig():
    fig = plt.figure(1,figsize=(6,5)) # width, height


    iapp = 55.# SNIC bifufcation at 40.8
    T, init, err, lc_max = ml_limit_cycle(dy_dt_SNIC, [0,.1], iapp=iapp)
    dt = .01 # time step
    numcycle = 4
    tot = T*numcycle # total integration time
    steps = int(tot/dt)+1 # total steps
    t = np.linspace(0,tot,steps)
    sol = integrate.odeint(ml.ml_rhs, lc_max, t, args=(iapp,))
    #sol = integrate.odeint(lamom, [0,1], t)
    lcsol = sol[:int(T/dt),:]


    ax1a = plt.subplot2grid((1,1),(0,0)) # top left

    # first tight layout command to fit axis tick labels
    #plt.tight_layout()
    #plt.figure(1,figsize=(8,4)) # width, height
    #plt.plot(t,sol[:,0],lw=5);plt.title('Fig1a')
    #plt.savefig('fig1a.png')


    # limit cycle solution over time
    ax1a.plot(lcsol[:,0],lcsol[:,1],lw=3,color='black')
    ax1a.set_xlabel(r'\textbf{V}')
    ax1a.set_ylabel(r'\textbf{w}')

    myLocatora = mticker.MultipleLocator(20)
    ax1a.xaxis.set_major_locator(myLocatora)

    k = int(.92*len(lcsol[:,0]))
    vecx = -(lcsol[k,0] - lcsol[k+10,0])
    vecy = -(lcsol[k,1] - lcsol[k+10,1])
    sf = 1

    #ax1a.annotate('',(lcsol[k+50,0],lcsol[k+50,1]),(lcsol[k,0],lcsol[k,1]),
    #              arrowprops=dict(facecolor='black',shrink=0,
    #                              headwidth=15,
    #                              frac=1,
    #                          ))


    #ax1c.set_title('Title 3')    
    #plt.plot(lcsol[:,0],lcsol[:,1],lw=5);plt.title('Fig1c')

    # add time ticks
    # get spacing
    # divide period T into 'div' even intervals
    div = 20.
    interval = T/div
    d_idx = int(interval/dt)
    
    # for each interval, mark with circle
    k = 0
    color = 0.
    while k < T/dt-d_idx:
        color = k/(1.*T/dt)
        ax1a.scatter(lcsol[k,0],lcsol[k,1],
                     facecolor=str(color),
                     edgecolor='black',
                     s=90,
                     zorder=10)
        k += d_idx
    plt.tight_layout()
    return fig


def hfun_comparison(type):
    """
    compare the different h functions for the 11ee and 11ii cases when beta=0.05 and beta=0.5
    """

    if type=='11ee':
        # load data from xpp
        syn_fast = np.loadtxt('ml_mem1_neg2hodd_exc_i43.5_beta0.25.dat')
        syn_slow = np.loadtxt('ml_mem1_neg2hodd_exc_i43.5_beta0.05.dat')
    elif type=='11ii':
        syn_fast = np.loadtxt('ml_mem1_neg2hodd_inh_i43.5_beta0.25.dat')
        syn_slow = np.loadtxt('ml_mem1_neg2hodd_inh_i43.5_beta0.05.dat')
    elif type=='22ee':
        syn_fast = np.loadtxt('ml_mem2_neg2hodd_exc_beta0.25.dat')
        syn_slow = np.loadtxt('ml_mem2_neg2hodd_exc_beta0.05.dat')
    elif type=='22ii':
        syn_fast = np.loadtxt('ml_mem2_neg2hodd_inh_beta0.25.dat')
        syn_slow = np.loadtxt('ml_mem2_neg2hodd_inh_beta0.05.dat')
        

    
    fig = plt.figure(figsize=(10,6)) # width height

    ax1 = plt.subplot2grid((1,1),(0,0))
    #ax2 = plt.subplot2grid((1,2),(0,1))

    ax1.plot([0,syn_fast[-1,0]],[0,0],ls='dashed',color='gray') # zero line

    ax1.plot(syn_fast[:,0],-2*syn_fast[:,1],lw=5,color='k',label=r'$\beta=0.25$')
    ax1.plot(syn_slow[:,0],-2*syn_slow[:,1],lw=5,color='k',ls='--',label=r'$\beta=0.05$')

    plt.legend(loc='best',fontsize='18')

    #ax1.fill_between(prc1[:,0],0,prc1[:,1],alpha=.5,color='gray')

    ax1.set_xlabel(r'$\bm{\theta}$')
    ax1.set_xlim(0,syn_fast[-1,0])

    plt.tight_layout()
    return fig

def hfun(type):
    """
    show examples of h functions
    """
    if type=='11ee':
        # load data from xpp
        #syn_fast = np.loadtxt('ml_mem1_neg2hodd_exc_i43.5_beta0.25.dat')
        syn_slow = np.loadtxt('ml_mem1_neg2hodd_exc_i43.5_beta0.05.dat')
        syn_slow[:,1] = -2*syn_slow[:,1]
    elif type=='11ii':
        #syn_fast = np.loadtxt('ml_mem1_neg2hodd_inh_i43.5_beta0.25.dat')
        syn_slow = np.loadtxt('ml_mem1_neg2hodd_inh_i43.5_beta0.05.dat')
        syn_slow[:,1] = -2*syn_slow[:,1]
    elif type=='22ee':
        #syn_fast = np.loadtxt('ml_mem2_neg2hodd_exc_beta0.25.dat')
        syn_slow = np.loadtxt('ml_mem2_neg2hodd_exc_beta0.05.dat')
        syn_slow[:,1] = -2*syn_slow[:,1]
    elif type=='22ii':
        #syn_fast = np.loadtxt('ml_mem2_neg2hodd_inh_beta0.25.dat')
        syn_slow = np.loadtxt('ml_mem2_neg2hodd_inh_beta0.05.dat')
        syn_slow[:,1] = -2*syn_slow[:,1]
    elif type=='12ee':
        #syn_fast = np.loadtxt('ml_mem2_neg2hodd_inh_beta0.25.dat')
        syn_slow = np.loadtxt('ml_12ee_hfun_1i43.5.dat')
    
    # force interval to be on [0,2pi]?
    N = len(syn_slow[:,0])
    syn_slow[:,0] = np.linspace(0,2*pi,N)

    # find zeros
    zero_crossings = np.where(np.diff(np.sign(syn_slow[:,1])))[0]

    # discard one of two near-zero crossings
    dx = (syn_slow[1,0] - syn_slow[0,0])


    zero_phase = False
    if np.abs(zero_crossings[-1]-N) < 20:
        zero_crossings = zero_crossings[:-1]
        zero_phase = True

    #print 'zero_crossings for', type,' : ',zero_crossings


    # mark stable/unstable
    stability = []
    k = 0
    for idx in zero_crossings:
        # if prev entry > next entry => stable, -1
        idx_shift_back = np.mod(idx-10,N)
        idx_shift_front = np.mod(idx+10,N)

        if syn_slow[idx_shift_back,1] > syn_slow[idx_shift_front,1]:
            stability.append('stable')
            #print idx,'stable'
        else:
            stability.append('unstable')
        k += 1

    #fig = plt.figure(figsize=(5,4)) # width height

    fig = plt.figure(figsize=(10,6)) # width height
    

    ax1 = plt.subplot2grid((1,1),(0,0))
    
    #ax2 = plt.subplot2grid((1,2),(0,1))

    ax1.plot([0,syn_slow[-1,0]],[0,0],ls='dashed',color='gray') # zero line
    

    #ax1.plot(syn_fast[:,0],-2*syn_fast[:,1],lw=5,color='k',label=r'$\beta=0.25$')
    ax1.plot(syn_slow[:,0],syn_slow[:,1],lw=3,color='k')

    for i in range(len(zero_crossings)):
        idx = zero_crossings[i]
        if stability[i] == 'stable':
            ax1.scatter(syn_slow[idx,0],syn_slow[idx,1],facecolors='black',s=180,zorder=3)

            # draw arrow 
            
            #print dx
            vecx = .05*2*pi
            vecy = 0
            arrow_base1 = 2*vecx/dx
            
            # make arrow width proportional to figure height
            amp = np.amax(syn_slow[:,1])- np.amin(syn_slow[:,1])
            ww = amp/5
            ax1.arrow(syn_slow[idx-arrow_base1,0],0,vecx,vecy,head_width=ww/3.,head_length=vecx/2., width=ww/10.,fc='k',ec='k',zorder=3)
            ax1.arrow(syn_slow[idx+arrow_base1,0],0,-vecx,vecy,head_width=ww/3.,head_length=vecx/2., width=ww/10.,fc='k',ec='k',zorder=3)

            
            if zero_phase == True and idx==0:
                ax1.scatter(syn_slow[-1,0],syn_slow[-1,1],facecolors='black',s=180,zorder=3)

        else:
            ax1.scatter(syn_slow[idx,0],syn_slow[idx,1],facecolors='white',s=180,zorder=3)

            if zero_phase == True and idx==0:
                ax1.scatter(syn_slow[-1,0],syn_slow[-1,1],facecolors='white',s=180,zorder=3)
        
    
    # plot zeros
    

    #plt.legend(loc='best',fontsize='18')

    #ax1.fill_between(prc1[:,0],0,prc1[:,1],alpha=.5,color='gray')

    ax1.set_xlabel(r'$\bm{\theta}$')
    ax1.set_ylabel(r'$\bm{H(\theta)}$')
    ax1.set_xlim(0,syn_slow[-1,0])
    plt.tight_layout()

    return fig

def hfun_combined(type=None):

    if type == None:
        """
        show examples of h functions
        """
        # load data from xpp

        #filenames = ['beta0.05/ml_mem1_neg2hodd_exc_i43.5_beta0.05.dat',
        #             'beta0.05/ml_mem1_neg2hodd_inh_i43.5_beta0.05.dat',
        #             'beta0.05/ml_mem2_neg2hodd_exc_beta0.05.dat',
        #             'beta0.05/ml_mem2_neg2hodd_inh_beta0.05.dat',
        #             'beta0.05/ml_12ei_hfun_1i43.5.dat',
        #             'beta0.05/ml_12ie_hfun_1i43.5.dat',
        #             'beta0.05/ml_12ee_hfun_1i43.5.dat',
        #             'beta0.05/ml_12ii_hfun_2i88.9.dat',
        #             ]
        # all i1=43.5, i2=88.5
        filenames = ['beta0.05/rhs1ee.dat',
                     'beta0.05/rhs1ii.dat',
                     'beta0.05/rhs2ee.dat',
                     'beta0.05/rhs2ii.dat',
                     'beta0.05/rhs1i_2e.dat',
                     'beta0.05/rhs1e_2i.dat',
                     'beta0.05/rhs1e_2e.dat',
                     'beta0.05/rhs1i_2i.dat',
                     ]

        #print filenames
        figure_title = [r'\textbf{(a)}', r'\textbf{(b)}', r'\textbf{(c)}', r'\textbf{(d)}', r'\textbf{(e)}', r'\textbf{(f)}',r'\textbf{(g)}',r'\textbf{(h)}']

        nrow=4;ncol=2;
        fig, axs = plt.subplots(nrows=nrow, ncols=ncol,figsize=(10,14))

        j = 0

        # loop over each subplot
        for ax1 in axs.reshape(-1):
            syn_slow = np.loadtxt(filenames[j])

            # multiply by -2 if function is only the odd part exported from xpp.
            #if (filenames[j] != filenames[-3]) and (filenames[j] != filenames[-4]):
            #    syn_slow[:,1] = -2*syn_slow[:,1]
            #print filenames[j]

            # force interval to be on [0,1] -- easier to eyeball phase locking than [0,2pi]
            N = len(syn_slow[:,0])
            syn_slow[:,0] = np.linspace(0,1,N)

            # find zeros
            zero_crossings = np.where(np.diff(np.sign(syn_slow[:,1])))[0]

            if len(zero_crossings) != 0:
                # if there are zero crossings, determine stability and plot
                # discard one of two near-zero crossings


                dx = (syn_slow[1,0] - syn_slow[0,0])

                zero_phase = False
                if (np.abs(zero_crossings[-1]-N) < 2000):
                    zero_crossings_old = zero_crossings
                    zero_crossings = zero_crossings[:-1]
                    zero_phase = True

                # mark stable/unstable
                stability = []
                k = 0
                for idx in zero_crossings:
                    # if prev entry > next entry => stable, -1
                    idx_shift_back = np.mod(idx-10,N)
                    idx_shift_front = np.mod(idx+10,N)

                    if syn_slow[idx_shift_back,1] > syn_slow[idx_shift_front,1]:
                        stability.append('stable')
                        #print idx,'stable'
                    else:
                        stability.append('unstable')
                    k += 1


                ax1.plot([0,syn_slow[-1,0]],[0,0],ls='dashed',color='gray') # zero line


                #ax1.plot(syn_fast[:,0],-2*syn_fast[:,1],lw=5,color='k',label=r'$\beta=0.25$')
                ax1.plot(syn_slow[:,0],syn_slow[:,1],lw=3,color='k')

                # for each zero crossing and its associated stability, plot arrows
                for i in range(len(zero_crossings)):
                    idx = zero_crossings[i]
                    if stability[i] == 'stable':
                        ax1.scatter(syn_slow[idx,0],syn_slow[idx,1],facecolors='black',s=180,zorder=3)

                        # draw arrow 

                        vecx = 550*dx
                        vecy = 0
                        arrow_base1 = 3*vecx/dx


                        # make arrow width proportional to figure height
                        amp = np.amax(syn_slow[:,1])- np.amin(syn_slow[:,1])
                        ww = amp/5
                        ax1.arrow(syn_slow[idx-arrow_base1,0],0,vecx,vecy,head_width=ww/3.,head_length=vecx/2., width=ww/10.,fc='k',ec='k',zorder=3)
                        ax1.arrow(syn_slow[idx+arrow_base1,0],0,-vecx,vecy,head_width=ww/3.,head_length=vecx/2., width=ww/10.,fc='k',ec='k',zorder=3)


                        if zero_phase == True and (idx>=0 and idx <=10):
                            print 'caught idx0',idx,filenames[j]
                            ax1.scatter(syn_slow[-1,0],syn_slow[-1,1],facecolors='black',s=180,zorder=3)

                    else:
                        ax1.scatter(syn_slow[idx,0],syn_slow[idx,1],facecolors='white',s=180,zorder=3)

                        if zero_phase == True and (idx>=0 and idx <=20):
                            ax1.scatter(syn_slow[-1,0],syn_slow[-1,1],facecolors='white',s=180,zorder=3)


                # plot zeros


                #plt.legend(loc='best',fontsize='18')

                #ax1.fill_between(prc1[:,0],0,prc1[:,1],alpha=.5,color='gray')
                ax1.set_title(figure_title[j], x=-.19,y=1.08)
                
                # hard code this stuff. fixed points too close to zero, or not caught:
                if filenames[j] == 'beta0.05/rhs1ee.dat':
                    ax1.scatter(syn_slow[-1,0],syn_slow[-1,1],facecolors='white',s=180,zorder=3)

                if filenames[j] == 'beta0.05/rhs2ee.dat':
                    print 'text'
                    ax1.scatter(syn_slow[-1,0],syn_slow[-1,1],facecolors='black',s=180,zorder=3)

                if filenames[j] == 'beta0.05/rhs1e_2i.dat':
                    
                    vecx = 550*dx
                    vecy = 0
                    arrow_base1 = 3*vecx/dx
                    
                    
                    # make arrow width proportional to figure height
                    amp = np.amax(syn_slow[:,1])- np.amin(syn_slow[:,1])
                    ww = amp/5
                    idx = zero_crossings_old[-1]
                    ax1.scatter(syn_slow[idx,0],syn_slow[idx,1],facecolors='black',s=180,zorder=3)
                    ax1.arrow(syn_slow[idx-arrow_base1,0],0,vecx,vecy,head_width=ww/3.,head_length=vecx/2., width=ww/10.,fc='k',ec='k',zorder=3)
                    ax1.arrow(syn_slow[np.mod(idx+arrow_base1,N),0],0,-vecx,vecy,head_width=ww/3.,head_length=vecx/2., width=ww/10.,fc='k',ec='k',zorder=3)
                    
            
                if j >= 6:
                    ax1.set_xlabel(r'$\bm{\psi}$')
                else:
                    ax1.xaxis.set_ticklabels([])

                if j%2 == 0:
                    ax1.set_ylabel(r'$\bm{H(\psi)}$')
                ax1.set_xlim(0,syn_slow[-1,0])

                j += 1
            else:
                # if no zero crossing, just plot line

                # if plot is above x-axis, set ylim(below zero,max+pad)
                # if plot below x-axis, set ylim(min-pad,above zero)
                pad = (np.amax(syn_slow[:,1]) - np.amin(syn_slow[:,1]))/20
                if np.sign(syn_slow[0,1])>=0:
                    ax1.set_ylim(-pad,np.amax(syn_slow[:,1])+pad)
                else:
                    ax1.set_ylim(np.amin(syn_slow[:,1])-pad,pad)
                ax1.plot(syn_slow[:,0],syn_slow[:,1],lw=3,color='k')
                ax1.plot([0,syn_slow[-1,0]],[0,0],ls='dashed',color='gray') # zero line

                ax1.set_title(figure_title[j], x=-.19,y=1.08)

                if j >= 6:
                    ax1.set_xlabel(r'$\bm{\psi}$')
                else:
                    ax1.xaxis.set_ticklabels([])

                if j%2 == 0:
                    ax1.set_ylabel(r'$\bm{H(\psi)}$')
                ax1.set_xlim(0,syn_slow[-1,0])
                j += 1


        plt.tight_layout()
        fig.subplots_adjust(hspace=.3)
        return fig



def generate_figure(function, args, filenames, title="", title_pos=(0.5,0.95)):
    # workaround for python bug where forked processes use the same random 
    # filename.
    #tempfile._name_sequence = None;
    fig = function(*args)
    fig.text(title_pos[0], title_pos[1], title, ha='center')
    if type(filenames) == list:
        for name in filenames:
            if name.split('.')[-1] == 'ps':
                fig.savefig(name, orientation='landscape')
            else:
                fig.savefig(name)
    else:
        if name.split('.')[-1] == 'ps':
            fig.savefig(filenames,orientation='landscape')
        else:
            fig.savefig(filenames)




def main():
    
    figures = [
        #(phase_plane,[],['phase_plane.png','phase_plane.eps','phase_plane.pdf']),
        #(prc,[],['prcs.png','prcs.eps','prcs.pdf']),
        #(phase_diff,[],['theory_vs_numerics.png','theory_vs_numerics.eps','theory_vs_numerics.pdf']),
        #(isochron_fig,[],['isochron_fig.png','isochron_fig.eps','isochron_fig.pdf']),
        #(hfun_comparison,['11ee'],['hfun_comparison11ee.png','hfun_comparison11ee.eps','hfun_comparison11ee.pdf']),
        #(hfun_comparison,['11ii'],['hfun_comparison11ii.png','hfun_comparison11ii.eps','hfun_comparison11ii.pdf']),
        #(hfun_comparison,['22ee'],['hfun_comparison22ee.png','hfun_comparison22ee.eps','hfun_comparison22ee.pdf']),
        #(hfun_comparison,['22ii'],['hfun_comparison22ii.png','hfun_comparison22ii.eps','hfun_comparison22ii.pdf']),
        #(hfun,['11ee'],['hfun_11ee.png','hfun_11ee.eps','hfun_11ee.pdf']),
        #(hfun,['11ii'],['hfun_11ii.png','hfun_11ii.eps','hfun_11ii.pdf']),
        #(hfun,['12ee'],['hfun_12ee.png','hfun_12ee.eps','hfun_12ee.pdf']),
        #(hfun,['22ee'],['hfun_12ee.png','hfun_12ee.eps','hfun_12ee.pdf']),
        #(hfun,['22ii'],['hfun_12ii.png','hfun_12ii.eps','hfun_12ii.pdf']),
        #(hfun_combined,[],['hfun_combined.png','hfun_combined.eps','hfun_combined.pdf']),
        #(hfun_combined,[],['hfun_combined_alt1.png','hfun_combined_alt1.eps','hfun_combined_alt1.pdf']),
        (hfun_combined,[],['hfun_combined_alt2.png','hfun_combined_alt2.eps','hfun_combined_alt2.pdf']),
        ]
    for fig in figures:
        generate_figure(*fig)


if __name__ == "__main__":
    main()
