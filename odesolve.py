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

    Numerically integrate a (non)autonomous nonlinear ODE.

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

    for i in range(1,len(deltas)+1):
        x_n = stepfunc(times[i-1],deltas[i-1], x_n, func)
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

def lorenz(sigma,rho,beta,t,x):
    dxdt = sigma*(x[1] - x[0])
    dydt = x[0]*(rho - x[2]) - x[1]
    dzdt = x[0]*x[1] - beta*x[2]
    return np.array([dxdt, dydt, dzdt])

def thomas(b,t,x):
    dxdt = np.sin(x[1]) - b*x[0]
    dydt = np.sin(x[2]) - b*x[1]
    dzdt = np.sin(x[0]) - b*x[2]
    return np.array([dxdt,dydt,dzdt])

#
# Run module as program
#
def main():
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.animation as animation

    times = np.linspace(0,20,2000)
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

    # linear system
    n = 4
    x0 = (np.random.random((n,2)) - 0.5)*4
    for i in range(x0.shape[0]):
        euler_xvals = odesolve(times,x0[i,:],linear,euler_step)
        midpoint_xvals = odesolve(times,x0[i,:],linear,midpoint_step)
        rk4_xvals = odesolve(times,x0[i,:],linear,rk4_step)

        plt.plot(euler_xvals[:,0],euler_xvals[:,1],color='blue')
        plt.plot(midpoint_xvals[:,0],midpoint_xvals[:,1],color='green')
        plt.plot(rk4_xvals[:,0],rk4_xvals[:,1],color='red')
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.legend(['euler','midpoint','rk4'])
    plt.show()

    # # lorenz system
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    #
    # times = np.linspace(0,40,10000)
    # n = 42
    # x0 = (np.random.random((n,3)) - 0.5)*4
    # sigma = 10
    # rho = 28
    # beta = 8./3
    # rk4_all = []
    # for i in range(x0.shape[0]): # find trajectories of each particle
    #     euler_xvals = odesolve(times,x0[i,:],lambda t,x: lorenz(sigma,rho,beta,t,x),euler_step)
    #     midpoint_xvals = odesolve(times,x0[i,:],lambda t,x: lorenz(sigma,rho,beta,t,x),midpoint_step)
    #     rk4_xvals = odesolve(times,x0[i,:],lambda t,x: lorenz(sigma,rho,beta,t,x),rk4_step)
    #
    #     ax.plot(euler_xvals[:,0],euler_xvals[:,1],euler_xvals[:,2],color='blue')
    #     ax.plot(midpoint_xvals[:,0],midpoint_xvals[:,1],midpoint_xvals[:,2],color='green')
    #     ax.plot(rk4_xvals[:,0],rk4_xvals[:,1],rk4_xvals[:,2],color='red')
    #
    #     rk4_all += [rk4_xvals.T]
    #
    # plt.legend(['euler','midpoint','rk4'])
    # plt.show()
    #
    # # lorenz rk4 animation
    # def update(frame,linedata,lines):
    #     for line,data in zip(lines,linedata):
    #         # print('data.shape =',data.shape)
    #         line.set_data(data[0:2,:frame])
    #         line.set_3d_properties(data[2,:frame])
    #     return lines
    #
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # lines = [ax.plot(d[0,0:1],d[1,0:1],d[2,0:1])[0] for d in rk4_all]
    # anim = animation.FuncAnimation(fig,update,len(times),fargs=(rk4_all,lines),interval=1)
    # plt.show()


    # thomas attractor
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')

    tmax = 64
    nstep = 1000
    times = np.linspace(0,tmax,nstep)
    n = 128
    x0 = (np.random.random((n,3)) - 0.5)/2
    b = 0.2
    rk4_all = []
    for i in range(x0.shape[0]): # find trajectories of each particle
        rk4_xvals = odesolve(times,x0[i,:],lambda t,x: thomas(b,t,x),rk4_step)
        # ax.plot(rk4_xvals[:,0],rk4_xvals[:,1],rk4_xvals[:,2],color='blue')
        rk4_all += [rk4_xvals.T]

    # plt.legend(['rk4'])
    # plt.show()

    # lorenz rk4 animation
    tail = 20
    usetail = False
    def update(frame,linedata,lines):
        for line,data in zip(lines,linedata):
            # print('data.shape =',data.shape)
            if frame > tail and usetail:
                line.set_data(data[0:2,frame-tail:frame])
                line.set_3d_properties(data[2,frame-tail:frame])
            else:
                line.set_data(data[0:2,:frame])
                line.set_3d_properties(data[2,:frame])
        return lines

    fig = plt.figure()
    ax = Axes3D(fig)
    lines = [ax.plot(d[0,0:1],d[1,0:1],d[2,0:1])[0] for d in rk4_all]
    anim = animation.FuncAnimation(fig,update,nstep,fargs=(rk4_all,lines),interval=1)
    plt.show()

    # TODO: create strange attractor class that takes a type and parameter list
    # TODO: create ODESolver class to solve and visualize/plot/animate solutions

if __name__ == '__main__':
    # A = np.array([[-1,0],[0,1]]) # saddle
    A = np.array([[-1.5,2],[-1,1]]) # stable spiral
    main()
