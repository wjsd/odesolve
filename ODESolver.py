"""

ODESolver

wsanchezdupont@g.hmc.edu

Class for solving and animating (non)autonomous (non)linear systems of ODEs.

Notes:

"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import odesolve

class ODESolver():
    """
    ODESolver

    A class to solve and animate (non)autnomomous (non)linear systems of ODEs.
    """
    def __init__(self,times,x0,dynamics,stepfunc=odesolve.midpoint_step,solver=odesolve.odesolve,showparticles=True,tail=None,nframes=None,interval=10,blit=False,xlim=None,ylim=None,zlim=None,save=False,fn='ode.mp4'):
        """
        __init__

        Constructor.

        inputs
        -------
        times - (numpy array) times for which to find ODE solution
        x0 - (numpy matrix) matrix of initial conditions (N particles x M dimensions)
        dynamics - (function) differential equation dynamics. Should be a function handle that takes (t,x) as arguments (i.e. x_dot = dynamics(t,x))
        stepfunc - (str) type of step function to use for numerical integration ('euler','midpoint', or 'rk4' | default: 'midpoint')
        solver - (function) function handle for ode solver that takes (times,x0,dynamics,stepfunc) as arguments
        showparticles - (bool) use scatterplot to visualize particle states over time
        tail - (int) if not None, tail is the number of trailing states to draw on each particle trajectory. If None, draw full trajectory (default:None)
        nframes - (int or None) number of frames to animate if not None, else use len(times)
        interval - (int) millisecond delay between each frame of the animation (default: 200)
        blit - (bool) use blitting (NOTE: does not work with showparticles!) TODO: make blitting work with particle scatterplots
        xlim - (list or None) x-axis limits if not None, else use pyplot defaults (default:None)
        ylim - (list or None) y-axis limits if not None, else use pyplot defaults (default:None)
        zlim - (list or None) z-axis limits if not None, else use pyplot defaults (default:None)
        save - (bool) save animation if True
        fn - (str) filename to save animation if save is True
        """
        # solution params
        self.times = times
        self.x0 = x0
        self.dynamics = dynamics
        self.stepfunc=stepfunc
        self.solver = solver
        self.solutions = []

        # animation params
        self.showparticles = showparticles
        self.tail = tail
        self.nframes = nframes
        self.interval = interval
        self.blit = blit
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim

        # save
        self.save = save

        return

    def solve(self):
        """
        solve

        Solve the ODE.
        """
        self.solutions = []
        for i in range(self.x0.shape[0]): # find trajectories of each particle
            solution = self.solver(self.times,self.x0[i,:],self.dynamics,self.stepfunc) # M dims
            self.solutions += [solution.T]

        return self.solutions

    def animate(self):
        """
        animate

        Animate the ODE.
        """
        fig = plt.figure(figsize=(16,16))
        ax = Axes3D(fig)

        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        ax.set_zlim(self.zlim)

        # lines = [ax.plot(d[0,0:1],d[1,0:1],d[2,0:1])[0] for d in rk4_all]
        # anim = animation.FuncAnimation(fig,update,nstep,fargs=(rk4_all,lines),interval=1)
        lines = [ax.plot(ld[0,0:1],ld[1,0:1],ld[2,0:1])[0] for ld in self.solutions]

        if self.showparticles:
            self.scatters = [ax.scatter([],[],[]) for ld in self.solutions]

        self.anim = animation.FuncAnimation(fig,self.update,len(self.times),fargs=(self.solutions,lines),interval=self.interval,blit=self.blit)
        plt.show()

        if self.save:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

            anim.save(fn,writer=Writer)

        return self.anim

    def update(self,frame,linedata,lines):
        """
        update

        Update function for animation frames.
        """
        for line,data in zip(lines,linedata):
            # print('data.shape =',data.shape)
            if self.tail is not None and frame > self.tail:
                line.set_data(data[0:2,frame-self.tail:frame])
                line.set_3d_properties(data[2,frame-self.tail:frame])
            else:
                line.set_data(data[0:2,:frame])
                line.set_3d_properties(data[2,:frame])

        if self.showparticles:
            if frame > 0:
                for i in range(len(self.scatters)):
                    xdata = self.solutions[i][0,frame-1]
                    ydata = self.solutions[i][1,frame-1]
                    zdata = self.solutions[i][2,frame-1]

                    self.scatters[i]._offsets3d = ([xdata],[ydata],[zdata])

        return lines


#
# Run as program
#
if __name__ == "__main__":
    # thomas
    times = np.linspace(0,40,500)
    x0 = (np.random.random((40,3)) - 0.5)*2
    xlim = [-4,4]
    ylim=xlim
    zlim=ylim
    dynamics = lambda t,x:odesolve.thomas(0.2,t,x)
    s = ODESolver(times,x0,dynamics,xlim=xlim,ylim=ylim,zlim=zlim,showparticles=False,blit=True)
    solutions = s.solve()
    a = s.animate()

    # lorenz
    times = np.linspace(0,80,8000)
    x0 = (np.random.random((40,3)) - 0.5)*8
    xlim = [-15,15]
    ylim=xlim
    zlim=ylim
    dynamics = lambda t,x:odesolve.lorenz(10,28,8./3,t,x)
    s = ODESolver(times,x0,dynamics,stepfunc=odesolve.rk4_step,xlim=xlim,ylim=ylim,zlim=zlim,showparticles=True,blit=False)
    solutions = s.solve()
    a = s.animate()
