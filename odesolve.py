"""

Willis Sanchez-duPont

odesolve.py

Numerical ODE solver using euler step, midpoint, or RK step.

Notes:


"""
import numpy as np

def odesolve(times, x0, func, stepfunc):
    """
    odesolve

    Numerically integrate a (non)autonomous ODE.

    # TODO: figure out what to put as data type for func and stepfunc docs

    inputs
    -------
    times - (float array) array of times at which we want our solution
    x0 - (float array) initial condition vector
    func - (function(???)) dynamics function that takes x-vectors as input and outpus x_dot (dx/dt = func(x))
    stepfunc - (function?) numerical solver step operation (e.g. euler step x_n=(n+1) = x_n + dt*func(x_n))

    outputs
    --------
    xvals - (list) list of
    """
    deltas = times[1:] - times[:-1] # step sizes
    xvals = np.zeros((len(times), len(x0))) # pre-allocate integration results
    xvals[0,:] = x0
    x_n = x0

    for i in range(1,len(deltas)):
        x_n = stepfunc(times[i-1],deltas[i], x_n, func)
        xvals[i,:] = x_n

    return xvals

def euler_step(t,delta,x,func):
    """
    euler_step

    Perform a step of euler's method.

    inputs
    -------
    t - (float) current time
    delta - (float) step size
    x - (float array) current state vector
    func - (function) dynamics function

    outputs
    --------
    x_n - (float array) state vector after timestep
    """
    return x + delta*func(t,x)

def midpoint_step(t,delta,x,func):
    """
    midpoint_step

    One step of midpoint method integration.

    inputs
    -------
    t - (float) current time
    delta - (float) step size
    x - (float array) current state vector
    func - (function) dynamics function

    outputs
    --------
    x_n - (float array) state vector after timestep
    """
    return x + delta*func(t + delta/2, x + (delta/2)*func(t,x))


def rk4_step(t,delta,x,func):
    """
    rk4

    Single step of 4th-order Runge-Kutta method.

    inputs
    -------
    t - (float) current time
    delta - (float) step size
    x - (float array) current state vector
    func - (function) dynamics function

    outputs
    --------
    x_n - (float array) state vector after timestep
    """
    k1 = func(t,x)
    k2 = func(t + delta/2, x + k1*delta/2)
    k3 = func(t + delta/2, x + k2*delta/2)
    k4 = func(t + delta, x + k3*delta)

    return x + delta*(k1/6 + k2/3 + k3/3 + k4/6)



#
# Functions
#

def exp_decay(t,x):
    return -x

def exp_growth(t,x):
    return x

def linear(t,x):
        return A.dot(x)

# TODO: lorenz
def lorenz(t,x):
    return

#
# Run module as program
#
# TODO: fix strange behavior at last timestep(s?)
def main():
    from matplotlib import pyplot as plt

    times = np.linspace(0,1,2000)
    x0 = np.array([1.])

    # exp exp_decay
    euler_xvals = odesolve(times,x0,exp_decay,euler_step)
    midpoint_xvals = odesolve(times,x0,exp_decay,midpoint_step)
    rk4_xvals = odesolve(times,x0,exp_decay,rk4_step)

    plt.plot(times,np.exp(-times))
    plt.plot(times,euler_xvals)
    plt.plot(times,midpoint_xvals)
    plt.plot(times,rk4_xvals)
    plt.xlabel('time')
    plt.ylabel('x')
    plt.legend(['true','euler','midpoint','rk4'])
    plt.show()

    # exp_growth
    euler_xvals = odesolve(times,x0,exp_growth,euler_step)
    midpoint_xvals = odesolve(times,x0,exp_growth,midpoint_step)
    rk4_xvals = odesolve(times,x0,exp_growth,rk4_step)

    plt.plot(times,np.exp(times))
    plt.plot(times,euler_xvals)
    plt.plot(times,midpoint_xvals)
    plt.plot(times,rk4_xvals)
    plt.xlabel('time')
    plt.ylabel('x')
    plt.legend(['true','euler','midpoint','rk4'])
    plt.show()

    # linear
    x0 = np.random.random((8,2)) - 0.5
    for i in range(x0.shape[0]):
        euler_xvals = odesolve(times,x0[i,:],linear,euler_step)
        midpoint_xvals = odesolve(times,x0[i,:],linear,midpoint_step)
        rk4_xvals = odesolve(times,x0[i,:],linear,rk4_step)

        plt.plot(euler_xvals[:,0],euler_xvals[:,1])
        plt.plot(midpoint_xvals[:,0],midpoint_xvals[:,1])
        plt.plot(rk4_xvals[:,0],rk4_xvals[:,1])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.legend(['euler','midpoint','rk4'])
    plt.show()

if __name__ == '__main__':
    A = np.array([[-1,0],[0,1]])
    main()
