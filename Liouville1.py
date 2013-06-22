#
# Liouville1.py
#
# This runs on ubuntu 12.04 circa June 2013. All the "normal python stuff"
# has been installed. For whatever that's worth.
#
# Tweak the bottom few lines to put in the hamiltonian you want, and
# then do: python Liouville1.py
#
# Or: ipython -i Liouville1.py
#
# WARNING: This is hard-coded to use /tmp/Animate. Search for that
# string to change ...
#
# This version is independent of any other code on Vector, and it doesn't
# use cython. This was done so that I could post it on github without the
# other "dependencies." Of course, it runs a LOT slower this way. If you
# want a copy of the other stuff, it's available.
#
# msg2mw (at) gmail (dot) com.
#

import sympy as sy
from sympy.core.cache import clear_cache
import numpy as np
import pylab
import datetime
from numpy.random import random_sample
from IPython.core.display import clear_output
from IPython.core.display import Image
from IPython.core.display import display
import inspect
from time import sleep
from time import time
import os
import sys
import commands

# Some shortcuts:
Sym = sy.Symbol

################################################################

#
# Liouville Simulation
#
os.system('mkdir /tmp/Animate') # Make sure this directory exists

# This is probably a horrible kluge, but I just can't figure out the
# "right" way to determine the amount of free memory!
def getmem():
    foo = commands.getoutput('free -b')
    return(int(foo.split()[16]))

# This is to print, for example, the estimated time left in a run
def make_time_string(t):
    # The time is in seconds. It's probably a float, but make sure.
    t = float(t)
    MINUTE = 60.
    HOUR = 60. * MINUTE
    DAY = 24. * HOUR
    if t > DAY: return( '%.2f days'%(t/DAY) )
    if t > HOUR: return( '%.2f hours'%(t/HOUR) )
    if t > MINUTE: return( '%.2f minutes'%(t/MINUTE) )
    return ( '%.2f seconds'%t )

def animate_points(H, bXs, bPs, iXs, iPs, update_function, xdom, pdom, tdom,
    pointsize=6, fps=5, filename='liouville', title=0):

    npts = len(bXs) # Number of points
    t0 = tdom[0]; t1 = tdom[1]; dt = tdom[2]
    # Clear any old files out of the animation directory
    os.system('cd /tmp/Animate; rm -rf frame*.png')
    
    start_time = time()
    start_time = time()
    start_mem = getmem()
    
    frames_to_do = (t1-t0)/dt
    t = t0 
    frame = 0

    while (t < t1):

        #
        # Plot the points
        #

        pylab.plot(bXs, bPs, color='black',
            linestyle='_', marker='o',  markersize=pointsize)
        pylab.plot(iXs, iPs, color='red',
            linestyle='_', marker='o',  markersize=pointsize)
        pylab.xlim(xdom); pylab.ylim(pdom)
        pylab.plt.axes().set_aspect('equal')
        pylab.plt.axes().set_title(
            'Hamiltonian: $%s$, time: %.2f'%(sy.latex(H), t))

        pylab.savefig('/tmp/Animate/frame%s.png'%str(frame).rjust(6,'0'))
        pylab.close('all')

        #
        # Do memory and time checks
        #

        if getmem() < 1000000000:
            raise Exception(
                '***** Less than a Gig of memory left - Aborting *****')
        if frame < 10:
            print('Frame %s of %s Mem used: %s'
                %(frame, frames_to_do, '{:,}'.format(start_mem - getmem())))
        else:
            # Try to figure out how long we have to go on this particular
            # animation.
            print('Frame %s of %s, mem used: %s'
                %(frame, frames_to_do, '{:,}'.format(start_mem - getmem()))),

            time_last_frame = time()-last_frame_started
            time_left = (frames_to_do - (frame)) * time_last_frame 
            if time_left > 3600.0:
                print('Est %.2f hours left to go' %(time_left/3600.0))
            elif time_left > 60:
                print('Est %.2f minutes left to go' %(time_left/60.0))
            else:
                print('Est %.2f seconds left to go' %time_left)

        sys.stdout.flush()
        last_frame_started = time()
        clear_cache() # Avoid sympy memory leak

        #
        # Update the points for next time
        #

        frame += 1
        t += dt;

        # Update point with cython
        #update_points(bXs, bPs, dt)
        #update_points(iXs, iPs, dt)

        # Update points with python
        update_function(bXs, bPs, dt)
        update_function(iXs, iPs, dt)

    frame_time = time()-start_time
    start_time = time()

    cnvt_cmnd = 'cd /tmp/Animate; '
    cnvt_cmnd += 'avconv '
    cnvt_cmnd += '-y '
    cnvt_cmnd += '-r %s '%fps
    cnvt_cmnd += '-i frame%06d.png '
    cnvt_cmnd += '-r %s '%fps
    cnvt_cmnd += '-b 3000k '
    if not title==0:
        cnvt_cmnd += '-metadata title="%s" '%title
    cnvt_cmnd += '/tmp/Animate/%s.mp4 ' %filename
    cnvt_cmnd += '2> /dev/null'

    print('Convert command:'); 
    print(cnvt_cmnd); sys.stdout.flush()
    os.system(cnvt_cmnd)
    cnvt_time = time()-start_time
    print('%s to create the %s frames'
        %( make_time_string(frame_time), frame ))
    print('%s to run the convert command'
        % make_time_string(cnvt_time) )

# Create an update function based on a Hamiltonian
def ufunc_from_hamiltonian(H):
    print('The hamiltonian is:$\;\;%s$' %sy.latex(H))
    dxdt = sy.diff(H,p)
    dpdt = -sy.diff(H,x)
    print(r'The update equations are:$\;\; \dot{x}=%s,\;\; \dot{p}=%s$'%(dxdt,dpdt))
    def U(X, P, dt):
        for n in range(len(X)):
            old_x = float(X[n])
            old_p = float(P[n])
            X[n] = old_x + dxdt.subs(p, old_p)*dt
            P[n] = old_p + dpdt.subs(x, old_x)*dt
    return(U)

def Liouville(
        Hamiltonian, n_border_pts, n_internal_pts, radius, display_bounds, 
        duration, pointsize=5, filename='Liouville_Test',
        title='Liouville Test', fudge=0, number_frames=100):
    U = ufunc_from_hamiltonian(Hamiltonian) # Update function from Hamiltonian

    # Add the date to the file name
    filename += '-'+datetime.date.today().strftime('%Y.%m.%d')

    # Arrays to hold border and interior x and p values
    bXs = []; bPs = []; iXs = []; iPs = []

    # Make the border points
    R = radius
    angle = 0
    for n in range(n_border_pts):
        bXs += [ R*np.sin(angle), ]
        bPs += [ R*np.cos(angle), ]
        angle += 2*np.pi/n_border_pts
    
    # Generate the inside points randomly. Currently the code will normally
    # provide LESS than the number requested, but to "rejections."
    for n in range(n_internal_pts):
        # Generate random points
        xval = random_sample()*2*R-R
        pval = random_sample()*2*R-R
        # Limit points to a circle of radius R
        if pval**2 + xval**2 - R**2 > -fudge: continue
        iXs += [ xval, ]
        iPs += [ pval, ]
    b = display_bounds
    animate_points(Hamiltonian, bXs, bPs, iXs, iPs, update_function=U,
        xdom=(-b,b), pdom=(-b,b),
        tdom=(0, duration, float(duration)/number_frames),
        pointsize=pointsize, filename=filename, title=title)
    print('Done!')

##################################################################
# THIS WILL RUN BY DEFAULT IF YOU DON'T MAKE ANY CHANGES
#
# Harmonic Oscillator
# Variables x and p need to be declared in advance
x=Sym('x'); p=Sym('p')
Liouville(
    Hamiltonian = p**2/2 + x**2/2, filename='Harmonic_Oscillator',
    title='Harmonic Oscillator',
    n_border_pts=30, n_internal_pts=100,radius=5, display_bounds=6,
    duration=1, number_frames=50)

##################################################################
# This intercepts control^c
"""
try:
    # Sin(p) + Sin(x)
    x=Sym('x'); p=Sym('p')
    Liouville(
        Hamiltonian = sy.sin(p)+sy.sin(x), filename='sinp_sinx',
        title='sin(p) + sin(x)',
        n_border_pts=1000, n_internal_pts=1000, radius=5, display_bounds=10,
        duration=10, number_frames=500)
except Exception:
    sys.exit()
"""
##################################################################
# This is the one that made the video on youtube. It will probably
# take a while to run.
"""
# Sin(p) + Sin(x)
x=Sym('x'); p=Sym('p')
Liouville(
    Hamiltonian = sy.sin(p)+sy.sin(x), filename='sinp_sinx',
    title='sin(p) + sin(x)',
    n_border_pts=1000, n_internal_pts=1000, radius=5, display_bounds=10,
    duration=10, number_frames=500)
"""
##################################################################
