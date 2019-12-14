"""

ODESolver

wsanchezdupont@g.hmc.edu

Class for solving and animating (non)autonomous (non)linear systems of ODEs.

Notes:

- Can be used to solve any nonautonomous nonlinear ODE, but can only plot in 3D
- If you want to plot a 2D ODE, set dynamics so that z = 0 always. You can also
  manually set your solution data by using the .set_solutions(sol) method.

TODO:

- Add 2D Animation (set view angle down z-axis?

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
    def __init__(self,times,x0,dynamics,stepfunc=odesolve.midpoint_step,solver=odesolve.odesolve,showparticles=True,tail=None,nframes=None,interval=10,blit=False,xlim=None,ylim=None,zlim=None):
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

        return

    def solve(self):
        """
        solve

        Solve the ODE.
        """
        self.solutions = []
        for i in range(self.x0.shape[0]): # find trajectories of each particle
            solution = self.solver(self.times,self.x0[i,:],self.dynamics,self.stepfunc) # T timesteps x M dims
            self.solutions += [solution.T]

        return self.solutions

    def animate(self,fps=60,nframes=None,filename=None,camDist=10,camElev=10,camAzim=0.1):
        """
        animate

        Animate the ODE.

        inputs:
            fps - (int) frames per second
            nframes - (int) number of frames to play
            filename - (str) name to save animation as
            camDist - (float) camera distance
            camElev - (float) camera elevation
            camAzim - (float) camera azimuth
        """
        # make sure that solutions can be plotted
        dims = [s.shape[0] for s in self.solutions]
        for d in dims:
            if d != 3:
                raise Exception('Solutions do not have dimension 3! Cannot animate!')

        fig = plt.figure(figsize=(16,16))
        ax = Axes3D(fig)
        ax.view_init(elev=camElev,azim=camAzim)

        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        ax.set_zlim(self.zlim)

        lines = [ax.plot(ld[0,0:1],ld[1,0:1],ld[2,0:1])[0] for ld in self.solutions]

        if self.showparticles:
            self.scatters = [ax.scatter([],[],[]) for ld in self.solutions]

        frames = nframes if nframes is not None else len(self.times)
        self.anim = animation.FuncAnimation(fig,self.update,frames,fargs=(self.solutions,lines),interval=self.interval,blit=self.blit)
        if filename is not None:
            Writer = animation.FFMpegWriter(fps=fps,codec='libx264', extra_args=['-pix_fmt', 'yuv420p', '-profile:v', 'high', '-tune', 'animation', '-crf', '18'])
            # writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)

            self.anim.save(filename,writer=Writer)
            print("Animation saved as {}".format(filename))
        else:
            plt.show()



    def update(self,frame,linedata,lines):
        """
        update

        Update function for animation frames.

        inputs
        -------
        frame - (int) current frame value
        linedata - (list) list of np arrays describing the data for each line
        lines - (list) list of plot handles
        """
        for line,data in zip(lines,linedata):
            if self.tail is not None and frame > self.tail:
                line.set_data(data[0:2,frame-self.tail:frame])
                line.set_3d_properties(data[2,frame-self.tail:frame])
            else:
                line.set_data(data[0:2,:frame])
                line.set_3d_properties(data[2,:frame])

        # display particles desired
        if self.showparticles:
            if frame > 0:
                for i in range(len(self.scatters)):
                    xdata = self.solutions[i][0,frame-1]
                    ydata = self.solutions[i][1,frame-1]
                    zdata = self.solutions[i][2,frame-1]
                    self.scatters[i]._offsets3d = ([xdata],[ydata],[zdata])

        return lines

    def set_solutions(self,sol):
        """
        set_solutions

        Method to manually set solution arrays. Useful for plotting in 2D or simply animating
        an ODE after having solved it elsewhere. Throws an exception if dim(sol[i]) != 2

        inputs
        -------
        sol - (list) python list of numpy matrices, where each matrix is a D=3 dimensions x N particles
              solution to the ODE.
        """
        dims = [s.shape[0] for s in sol]
        for d in dims:
            if d != 3:
                raise Exception('Solutions do not have dimension 3! Cannot animate!')

        self.solutions = sol

        return


#
# Run as program
#
if __name__ == "__main__":
    # # linear
    # times = np.linspace(0,10,500)
    # x0 = (np.random.random((40,3)) - 0.5)*8
    # xlim = [-15,15]
    # ylim=xlim
    # zlim=ylim
    # A = np.array([[-1.,1.,0.],[-1.,-1.,0.],[0.,0.,1.]])
    # dynamics = lambda t,x: A.dot(x)
    # s = ODESolver(times,x0,dynamics,stepfunc=odesolve.rk4_step,xlim=xlim,ylim=ylim,zlim=zlim)
    # solutions = s.solve()
    # a = s.animate()
    #
    # # thomas
    # times = np.linspace(0,40,500)
    # x0 = (np.random.random((40,3)) - 0.5)*2
    # xlim = [-4,4]
    # ylim=xlim
    # zlim=ylim
    # dynamics = lambda t,x:odesolve.thomas(0.2,t,x)
    # s = ODESolver(times,x0,dynamics,xlim=xlim,ylim=ylim,zlim=zlim,showparticles=False,blit=True)
    # solutions = s.solve()
    # a = s.animate()

    # system setup
    fps = 30
    seconds = 5
    nframes = round(fps*seconds)
    nsolves = nframes
    times = np.linspace(0,seconds,nsolves)
    x0 = (np.random.random((40,3)) - 0.5)*8

    # view settings
    xlim = [-8,8]
    ylim=xlim
    zlim=ylim
    camDist = 2
    camElev = 20
    camAzim = 10

    # solve and animate
    dynamics = lambda t,x:odesolve.thomas(0.2,t,x)
    # dynamics = lambda t,x:odesolve.lorenz(10,28,8./3,t,x)
    s = ODESolver(times,x0,dynamics,stepfunc=odesolve.rk4_step,xlim=xlim,ylim=ylim,zlim=zlim,showparticles=True,blit=False,tail=None)
    solutions = s.solve()
    a = s.animate(fps=fps,nframes=nframes,filename="thomas.mp4")

    print('[ ODESolver.py testing complete ]')
