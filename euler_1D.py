import matplotlib.pylab as mp
import numpy as np

def ESolve(f,x0,t,threshold=None,reset=None,spike_tracking=False,**args):
    """
    solve given SDE (both IF and non-IF)
    inputs are similar to those used in Odeint:
    x: initial condition (can be array or list)
    t: time vector (initial time, final time, steps implicit)
    f: RHS
    args: tuple of arguments (parameters) for RHS of f
    threshold: max membrane potential before reset
    reset: membrane potential reset
    spike_tracking: (for IF models only) if true, return list of length t of 1s and 0s.
    The entries with 1 denote a spike.

    This function assumes the time array is always divided into bins
    of equal length.    
    """
    # generate default args list if empty
    #print args
    if args == {}:
        args = {'args':[]}

    # h
    dt = t[1]-t[0]

    # create solution vector
    N = len(t)
    sol = np.zeros((N,1))

    # initial condition
    sol[0] = x0

    if spike_tracking:
        spikes = np.zeros(N)


    
    # solve
    time = t[0]

    # input to f (RHS) in loop
    inputs = [sol[0],time]
    for j in range(len(args['args'])):
        inputs.append(args['args'][j])

    for i in range(1,N):
        
        # clean up inputs to RHS
        #print args['args']

        # iterate
        inputs[0]=sol[i-1]
        inputs[1]=time
        sol[i] = sol[i-1]+dt*f(*inputs)

        # catch spikes
        if threshold != None and reset != None:
            if sol[i] >= threshold:
                sol[i] = reset
                if spike_tracking:
                    spikes[i] = 1

        time += dt

    if spike_tracking:
        return sol,spikes
    else:
        return sol
 


def example(x,t,a,b):
    """
    an example right hand side function
    """
    return x*a - t*b

def example2(x,t):
    """
    another example RHS
    """
    return x*(1-x)

def main():
    # an example non-autonomous function
    x0 = .1
    t0=0
    tend=1
    dt=.1
    t = np.linspace(t0,tend,int((tend-t0)/dt))

    # use the same syntax as odeint
    #sol = LSolve(example,x0,t,args=(1,1))
    #sol = ESolve(example2,x0,t)
    sol = ESolve(example,x0,t,args=(1,1))
    
    mp.figure()
    mp.title("Example solution")
    mp.plot(t,sol)

    mp.show()

if __name__ == "__main__":
    main()
