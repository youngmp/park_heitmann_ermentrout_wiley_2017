"""
For ML figures. Will incorporate into generate_figures_ml.py eventually

also consider using toy models Kuramoto, theta, Lamom.

For networks, might consider spacial networks or all to all coupling

Or network of ML oscillators.

"""

from matplotlib import pyplot as plt
from lib_ml import *


def prc_fig(number_of_perts, iapp, dx, dy=0.0):
    fig = plt.figure(figsize=(10,6))
    #width = 1.
    #padding = 0.2*width
    n_phis = np.linspace(0,1,number_of_perts)
    axes = plt.axes()

    #axes = plt.axes((2*padding, 1-(i+1) * width+padding,
    #                 1 - width - 2*padding, width - 1.5*padding))

    n_prc, err = ml_phase_reset(n_phis, iapp=iapp, dx=dx, hmax=1e-2)

    if dx != 0.0:
        n_prc = n_prc/dx

    #print n_prc

    # draw numerical PRC
    axes.plot(n_phis, n_prc, 'bo', markersize=4)
    axes.plot(n_phis, n_prc, 'b--', markersize=4)
    plt.title("Morris-Lecar PRC with iapp="+str(iapp))
    plt.xlabel("Phase")
    plt.ylabel("Phase diff")
    
    T, init, err, lc_max = ml_limit_cycle(dy_dt, [0,.089], iapp=iapp)

    #axes = plt.axes((1-width+padding, 1-(i+1) * width + padding,
    #                 width - 1.5 * padding, width - 1.5 * padding))
    
    #trajectory_fig(dy_dt, init, axes, T, iapp=iapp)

    return fig

def ml_rhs(y,t,iapp):
    ep,iapp,vinit,winit,gl,el,ek,gk,gca,v1,v2,v3,v4,eca,phi = return_params_SNIC(iapp)
    #ep,iapp,vinit,winit,gl,el,ek,gk,gca,v1,v2,v3,v4,eca,phi = return_params(iapp)
    return np.array([
        (-gca*0.5*(1.0 + np.tanh((y[0] - v1)/v2))*(y[0] - eca) -\
         gk*y[1]*(y[0] - ek) - gl*(y[0] - el) + iapp)/ep,
        phi*np.cosh((y[0] - v3)/(2.0*v4))*(0.5*(1.0 + np.tanh((y[0] -\
                                                               v3)/v4)) - y[1])])

def phase_diff_numerical(t,syn):
    """
    plot the combinations of phase differences in the full model
    t: type, str, '11', '12', '21', or '22'
    syn: 'ee', 'ei', 'ie', or 'ii'
    
    """

def main():

    #iapp = 55.# 36.8 for homoclinic
    #T, init, err, lc_max = ml_limit_cycle(dy_dt_SNIC, [0,.1], iapp=iapp)
    #print 'period T,',T
    #print 'init', init
    #T = np.pi*2
    #print 'LC error',err
    #dt = .01 # time step
    #tot = T*4 # total integration time
    #steps = int(tot/dt)+1 # total steps
    #t = np.linspace(0,tot,steps)
    #sol = integrate.odeint(ml_rhs, lc_max, t, args=(iapp,))
    #sol = integrate.odeint(lamom, [0,1], t)
    #lcsol = sol[:int(T/dt),:]
    
    
    # for each of TI, TII:
    # data: v(t), w(t) (run up to 20k but use 10-15k), hfunction, prc
    # for actual, find the limit cycle crossing times for vt1 + correspondnig index.

    ## TI numerical phase diff exc/exc

    plt.figure(55,figsize=(6,3))    
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

    ## TII numerical phase diff exc/exc
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
    max_v = ml_mem2_prc2_vtref[max_v_idx] # max v value
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

    # plot numerical phase diff alone
    #mp.plot(t1_crossing1,np.abs(prc1))    
    #mp.plot(t1_crossing2,np.abs(prc2))

    
    ## TI, TII theory + numerical phase diff comparison

    plt.figure(66,figsize=(6,3))    
    mp.title('phs1ee')
    ml_mem1_prc1_psi_theory = np.loadtxt('ml_phs1ee.dat')
    eps = 0.0025
    t1 = ml_mem1_prc1_psi_theory[:,0]
    dt1 = t1[1]-t1[0] # time step
    y1 = ml_mem1_prc1_psi_theory[:,1]
    mp.plot(ml_mem1_prc1_psi_theory[:int(50/dt1),0]/eps,
            ml_mem1_prc1_psi_theory[:int(50/dt1),1]) # theory
    mp.plot(t2_crossing1,np.abs(prc1)) # numerics

    plt.figure(67,figsize=(6,3))
    mp.title('phs2ee')
    ml_mem2_prc2_psi_theory = np.loadtxt('ml_phs2ee.dat')
    t2 = ml_mem2_prc2_psi_theory[:,0]
    dt2 = t2[1]-t2[0]
    y2 = ml_mem2_prc2_psi_theory[:,1]
    mp.plot(ml_mem2_prc2_psi_theory[:int(50/dt2),0]/eps,
            ml_mem2_prc2_psi_theory[:int(50/dt2),1]) # theory
    mp.plot(t1_crossing2,np.abs(prc2)) # numerics


    # compare TI,TII in all combinations:
    #   I     II
    #I  done  ?
    #II ?     done
    # claim TII with TI sync easily like TII with TII.


    # plot phase diff theory + the RHS of phase models
    suffix = ['1ee','1ei','1ii','2ee','2ei','2ii',
              '12ee','12ei','12ie','12ii']
    grayscale = np.linspace(0,.75,len(suffix)) # 0 black, 1 white
    phs={};rhs={}

    for val in suffix:
        phs[val] = np.loadtxt('ml_phs'+val+'.dat')
        #rhs[val] = np.loadtxt('ml_phs_rhs'+val+'.dat')

    mp.figure()
    mp.title('phase diff theory')
    k = 0

    flag = False
    for val in suffix:
        if val == '12ee':
            flag = True
        if not(flag):
            mp.plot(phs[val][:,0],phs[val][:,1],label=val)#,color=str(grayscale[k]),label=val)
        else:
            mp.plot(phs[val][:,0],phs[val][:,1],label=val,ls='dashed')
        k += 1
    mp.ylim(0,1)
    mp.legend()
    for val in suffix:
        print val, phs[val][-1,1]

    """
    mp.figure()
    mp.title('phase diff rhs')
    k = 0
    for val in suffix:
        mp.plot(rhs[val][:,0],rhs[val][:,1],label=val)#,color=str(grayscale[k]),label=val)
        k += 1
    mp.legend()
    """
    # display canvas

    plt.show()
    
    #fig = prc_fig(100, 36.2, 1e-6)
    #fig.savefig("ml_prc.png")
    

if __name__ == "__main__":
    main()
