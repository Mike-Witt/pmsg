#
# t d s e _ p r o f i l e . p y
#
# To get a readable profile in the file profile.out, do:
#
#   ipython --pylab tk tdse_profile.py >& profile.out
#
# Mike Witt - msg2mw@gmail.com / July 2013
#

# First, make sure the cython code has been compiled:
import os
os.system('python csim_build.py build_ext --inplace > csim_build.out 2>&1')

import scipy.integrate
from time import sleep, time
from csim_tdse import csim_tdse_1d
from numpy import exp, sqrt, inf, linspace

#
# demo()
#
#   This implements a simple quantum finite barrier problem.
#
def demo(
    npts=300,
    sim_time=90,
    real_time=10,
    bc=True,
    vh=.5, v0=-2, v1=2, vs=.15):

    # These variables might need to have certain types. Fix them here
    # so we don't have to worry about how they are supplied by the caller.

    npts = int(npts)    # Number of points to use on the x axis
    sim_time = float(sim_time)   # How long to run in simulation time
    real_time = float(real_time) # How long we *want* it to run in real time
    vh = float(vh)  # Height of the potential barrier
    v0 = float(v0)  # Start of the potential barrier
    v1 = float(v1)  # End of the potential barrier
    vs = float(vs)  # How to scale the potential for plotting

    # This establishes the domain of the problem along the x axis
    x0=-70
    xL=70
    x_axis = linspace(x0, xL, npts)

    # This code creates an "initial condition" representing a gaussian
    # wave packet travelling to the right along the x axis.

    ctr=-30 # The center of the wave
    k0=-1.0;
    spread=.05 # A smaller value gives ...
    L = float(xL-x0)
    S = spread*L
    # fnn() is the function we want, but it is "not normalized"
    def fnn(x): return(exp(-((x-ctr)/S)**2 - 1j*k0*x))
    # P() is the "probability wave" which must add up to unity
    # over the entire x axis.
    def P(x): return( (fnn(x)*fnn(x).conjugate()).real )
    # We integrate to find out what it currenty sums to.
    nf = sqrt(scipy.integrate.quad(P, -inf, inf)[0])
    # The defive by that so that it *does* sum to one.
    def f(x): return( exp(-((x-ctr)/S)**2 - 1j*k0*x)/nf )
    # Finally, we create the initial condition, psi_0, which is the
    # discrete version of f().
    psi_0 = []
    for x in x_axis: psi_0 += [ f(x), ]

    # The following constructs the (discrete) potential function.
    # (The potential in these problems is typically called "V")

    V = []  # V is the potential, used in the simulation
    Vs = [] # Vs is a "scaled" version, used  for plotting 
    for x in x_axis:
        if x < v0: value = 0.0
        elif x < v1: value = vh
        else: value = 0.0
        V += [ value, ]
        Vs += [ value*vs, ]

    print('Lengths: x_axis=%s, psi_0=%s, V=%s, Vs=%s'
        %(len(x_axis), len(psi_0), len(V), len(Vs)))

    # Plot height should be set so that the entire waveform fit into
    # the plot for the whole time range of interest.
    plot_height = .25

    # Get a simulator object
    global sim # DEBUG
    sim = csim_tdse_1d()
    # Tell the simulator about all the required parameters
    sim.psi_0 = psi_0
    sim.V = V
    sim.npts = npts
    sim.dx = float(xL - x0) / (npts-1)
    sim.dt = sim_time/10**6
    if bc:
        sim.using_bc = True
        sim.ux0 = 0.0
        sim.uxL = 0.0
    # Set the "dynamic step range" to the plot height
    sim.dsrange = plot_height
    # "dsfrac" is the fraction of dsrange allowed as the maximum change
    sim.dsfrac  = .001
    # Once the parameters are all set up we have to call make_ic()
    sim.make_ic()

    # The simulator is now ready to go. Before doing anything else,
    # we'll set up a plotting window with the initial condition.

    P = sim.u_r*sim.u_r + sim.u_i*sim.u_i
    wave, = plot(x_axis, P, color='black')
    plot(x_axis, Vs, color='black')
    fig = pylab.gcf()
    fig.canvas.set_window_title('Finite Barrier')
    pylab.xlim([x0, xL])
    pylab.ylim([0, plot_height])

    # Comment this out for the profiling version
    #print('Adjust window and hit return ...')
    #c = raw_input()

    #
    # Now we proceed to run the simulation
    #

    frames_per_second = 10
    N = 0
    iter = 0
    tot_run_time = 0.0
    tot_sleep_time = 0.0
    last_sim_t = 0
    frame_time = 1.0/frames_per_second
    run_time = sim_time / (real_time * frames_per_second)
    print('real_time=%.2f, sim_time=%.2f, fps=%s, run_time=%.2f'
        %(real_time, sim_time, frames_per_second, run_time))
    start = time()
    last_display_time = start
    while sim.t < sim_time:
        break_time = 1.0/frames_per_second
        # We want to step for run_time seconds. But if we reach the next
        # display time (at break_time) we'll stop no matter what.
        iter += sim.dynamic_step(run_time, break_time)
        this_run_time = time() - last_display_time
        tot_run_time += this_run_time
        if sim.t - last_sim_t < run_time:
            print('Can\'t keep up')
        if this_run_time < frame_time:
            stime = frame_time - this_run_time
            start_sleep = time()
            sleep(stime)
            act_sleep = time() - start_sleep
            if abs(stime-act_sleep) > .002:
                print('Sleep time: wanted=%s, actual=%s'%(stime, act_sleep)) 
            tot_sleep_time += act_sleep
        last_sim_t = sim.t
        last_display_time = time()
        P = sim.u_r*sim.u_r + sim.u_i*sim.u_i
        wave.set_ydata(P)
        foo = pylab.title('time=%s' %int(sim.t))
        wave.figure.canvas.show()
        wave.figure.show()
        N += 1

    avg_iter = float(iter)/N
    total_sec = time()-start
    print('Number of calls to step = %s' %N)
    print('Total: run_time=%.2f, sleep_time=%.2f'
        %(tot_run_time, tot_sleep_time))
    print('Total time = %.2f seconds' %total_sec)
    print('Avg iterations per display = %s' %avg_iter)

def profile():
    import pstats, cProfile
    cProfile.runctx("demo()", globals(), locals(), "Profile.prof")
    s = pstats.Stats("Profile.prof")
    s.strip_dirs().sort_stats("time").print_stats()

profile()
