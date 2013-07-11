# cython: profile=True
#
# c s i m _ t d s e . p y x
#
# The TDSE simulation in cython
#

import numpy as np
cimport numpy as cnp
from time import time

# This next import makes any "bool" variable Pythons booleans rather than
# C++ booleans. See:
#   http://wiki.cython.org/FAQ#HowdoIdeclareanobjectoftypebool.3F

from cpython cimport bool

cdef c_derivative(
    cnp.ndarray[cnp.float32_t, ndim=1] f, 
    cnp.ndarray[cnp.float32_t, ndim=1] d,
    int nx, float dx):

    cdef int x
    cdef float over_2dx = 1/(2*dx) # This does make some difference

    d[0] = (f[1] - f[0]) / dx
    for x in range(1, nx-1): d[x] = (f[x+1] - f[x-1]) * over_2dx
    d[nx-1] = (f[nx-1] - f[nx-2]) / dx

class csim_tdse_1d:
    def __init__(self):

        # Note that the caller must initialize everything here that is
        # set equal to "None" before calling make_ic().
        self.m = 1.0 # Mass
        self.hbar = 1.0 # Planck's const
        self.using_bc = False
        self.ux0 = 0.0
        self.uxL = 0.0
        self.npts = None
        self.dx = None
        self.dt = None

        # "dsrange" must be set by the user for dynamic stepping. It
        # could be, for example, just the height of the expected plot.
        # The variables that keep track of the "amount of change" during
        # the simulation will ultimately be viewed as a fraction of
        # this range.
        self.dsrange = None
        # "dsfrac" is the fraction of dsrange allowed as the maximum change.
        self.dsfrac = None

    # make_ic() must be called after all the necessary "self" parameters
    #   have been given values.        
    def make_ic(self):
        V = self.V # Save the V set by the user temporarily
        # The first u is psi_0 ...
        # Use numpy arrays for speed
        self.u_r = np.zeros(self.npts, dtype=np.float32)
        self.u_i = np.zeros(self.npts, dtype=np.float32)
        self.V = np.zeros(self.npts, dtype=np.float32)
        for n in range(self.npts):
            self.u_r[n] = self.psi_0[n].real
            self.u_i[n] = self.psi_0[n].imag
            self.V[n] = V[n]
        self.du_dx_r = np.zeros(self.npts, dtype=np.float32)
        self.du_dx_i = np.zeros(self.npts, dtype=np.float32)
        self.d2u_dx2_r = np.zeros(self.npts, dtype=np.float32)
        self.d2u_dx2_i = np.zeros(self.npts, dtype=np.float32)
        self.t = 0

    # Step the simulation N times. This was the original "step" function.
    #   it might be obsolete at this point.
    def step(self, N):
        cdef int x # Used in x loop below
        cdef float m = self.m
        cdef float hbar = self.hbar
        cdef float hBy2m = hbar/(2*m)
        cdef float vByh # Used in xloop below
        cdef int npts = self.npts
        cdef int nx = self.npts-1 
        cdef float dt = self.dt
        cdef float dx = self.dx
        cdef float ux0 = self.ux0
        cdef float uxL = self.uxL

        # These make a HUGE difference
        cdef cnp.ndarray[cnp.float32_t, ndim=1] V = self.V
        cdef cnp.ndarray[cnp.float32_t, ndim=1] u_r = self.u_r
        cdef cnp.ndarray[cnp.float32_t, ndim=1] u_i = self.u_i
        cdef cnp.ndarray[cnp.float32_t, ndim=1] du_dx_r = self.du_dx_r
        cdef cnp.ndarray[cnp.float32_t, ndim=1] du_dx_i = self.du_dx_i
        cdef cnp.ndarray[cnp.float32_t, ndim=1] d2u_dx2_r = self.d2u_dx2_r
        cdef cnp.ndarray[cnp.float32_t, ndim=1] d2u_dx2_i = self.d2u_dx2_i

        for n in range(N):
            #
            # Calculate the derivatives
            #
            c_derivative(u_r, du_dx_r, npts, dx)
            c_derivative(u_i, du_dx_i, npts, dx)
            # NOTE: We don't need to look at "derivative" BCs
            c_derivative(du_dx_r, d2u_dx2_r, npts, dx)
            c_derivative(du_dx_i, d2u_dx2_i, npts, dx)
            #
            # Updated the probability wave
            #
            for x in range(0, npts):
                vByh  = V[x]/hbar
                du_dt_r = -hBy2m * d2u_dx2_i[x] + vByh * u_i[x]
                du_dt_i =  hBy2m * d2u_dx2_r[x] - vByh * u_r[x]
                u_r[x] = u_r[x] + du_dt_r*dt
                u_i[x] = u_i[x] + du_dt_i*dt
            # Potentially handle Dirichlet (only) boundary conditions here
            if self.using_bc:
                u_r[0] = ux0
                u_i[0] = ux0
                u_r[nx] = uxL
                u_i[nx] = uxL
            # Update the simulation time
            self.t += dt

        self.u_r = u_r
        self.u_i = u_i
        self.du_dx_r = du_dx_r
        self.du_dx_i = du_dx_i
        self.d2u_dx2_r = d2u_dx2_r
        self.d2u_dx2_i = d2u_dx2_i

    # Step the simulation for s seconds from "last." This step function
    #   may also be obsolete.
    def timed_step(self, last, s):
        cdef int x # Used in x loop below
        cdef float m = self.m
        cdef float hbar = self.hbar
        cdef float hBy2m = hbar/(2*m)
        cdef float vByh # Used in xloop below
        cdef int npts = self.npts
        cdef int nx = self.npts-1 
        cdef float dt = self.dt
        cdef float dx = self.dx
        cdef float ux0 = self.ux0
        cdef float uxL = self.uxL
        cdef int N = 0

        # These make a HUGE difference
        cdef cnp.ndarray[cnp.float32_t, ndim=1] V = self.V
        cdef cnp.ndarray[cnp.float32_t, ndim=1] u_r = self.u_r
        cdef cnp.ndarray[cnp.float32_t, ndim=1] u_i = self.u_i
        cdef cnp.ndarray[cnp.float32_t, ndim=1] du_dx_r = self.du_dx_r
        cdef cnp.ndarray[cnp.float32_t, ndim=1] du_dx_i = self.du_dx_i
        cdef cnp.ndarray[cnp.float32_t, ndim=1] d2u_dx2_r = self.d2u_dx2_r
        cdef cnp.ndarray[cnp.float32_t, ndim=1] d2u_dx2_i = self.d2u_dx2_i

        target_time = last + s
        while time() < target_time:
            #
            # Calculate the derivatives
            #
            c_derivative(u_r, du_dx_r, npts, dx)
            c_derivative(u_i, du_dx_i, npts, dx)
            # NOTE: We don't need to look at "derivative" BCs
            c_derivative(du_dx_r, d2u_dx2_r, npts, dx)
            c_derivative(du_dx_i, d2u_dx2_i, npts, dx)
            #
            # Updated the probability wave
            #
            for x in range(0, npts):
                vByh  = V[x]/hbar
                du_dt_r = -hBy2m * d2u_dx2_i[x] + vByh * u_i[x]
                du_dt_i =  hBy2m * d2u_dx2_r[x] - vByh * u_r[x]
                u_r[x] = u_r[x] + du_dt_r*dt
                u_i[x] = u_i[x] + du_dt_i*dt
            # Potentially handle Dirichlet (only) boundary conditions here
            if self.using_bc:
                u_r[0] = ux0
                u_i[0] = ux0
                u_r[nx] = uxL
                u_i[nx] = uxL
            # Update the simulation time
            self.t += dt
            # Count number of iterations
            N += 1

        self.u_r = u_r
        self.u_i = u_i
        self.du_dx_r = du_dx_r
        self.du_dx_i = du_dx_i
        self.d2u_dx2_r = d2u_dx2_r
        self.d2u_dx2_i = d2u_dx2_i
        return(N)

    # Step the simulation for run_time seconds of simulation time, but it
    # we hit break_time seconds of real time abort.
    def new_step(self, run_time, break_time):
        real_time = time()

        cdef int x # Used in x loop below
        cdef float m = self.m
        cdef float hbar = self.hbar
        cdef float hBy2m = hbar/(2*m)
        cdef float vByh # Used in xloop below
        cdef int npts = self.npts
        cdef int nx = self.npts-1 
        cdef float dt = self.dt
        cdef float dx = self.dx
        cdef float ux0 = self.ux0
        cdef float uxL = self.uxL
        cdef int N = 0

        # These make a HUGE difference
        cdef cnp.ndarray[cnp.float32_t, ndim=1] V = self.V
        cdef cnp.ndarray[cnp.float32_t, ndim=1] u_r = self.u_r
        cdef cnp.ndarray[cnp.float32_t, ndim=1] u_i = self.u_i
        cdef cnp.ndarray[cnp.float32_t, ndim=1] du_dx_r = self.du_dx_r
        cdef cnp.ndarray[cnp.float32_t, ndim=1] du_dx_i = self.du_dx_i
        cdef cnp.ndarray[cnp.float32_t, ndim=1] d2u_dx2_r = self.d2u_dx2_r
        cdef cnp.ndarray[cnp.float32_t, ndim=1] d2u_dx2_i = self.d2u_dx2_i

        target_time = self.t + run_time
        while self.t < target_time:
            if time() - real_time > break_time: break
            #
            # Calculate the derivatives
            #
            c_derivative(u_r, du_dx_r, npts, dx)
            c_derivative(u_i, du_dx_i, npts, dx)
            # NOTE: We don't need to look at "derivative" BCs
            c_derivative(du_dx_r, d2u_dx2_r, npts, dx)
            c_derivative(du_dx_i, d2u_dx2_i, npts, dx)
            #
            # Updated the probability wave
            #
            for x in range(0, npts):
                vByh  = V[x]/hbar
                du_dt_r = -hBy2m * d2u_dx2_i[x] + vByh * u_i[x]
                du_dt_i =  hBy2m * d2u_dx2_r[x] - vByh * u_r[x]
                u_r[x] = u_r[x] + du_dt_r*dt
                u_i[x] = u_i[x] + du_dt_i*dt
            # Potentially handle Dirichlet (only) boundary conditions here
            if self.using_bc:
                u_r[0] = ux0
                u_i[0] = ux0
                u_r[nx] = uxL
                u_i[nx] = uxL
            # Update the simulation time
            self.t += dt
            # Count number of iterations
            N += 1

        self.u_r = u_r
        self.u_i = u_i
        self.du_dx_r = du_dx_r
        self.du_dx_i = du_dx_i
        self.d2u_dx2_r = d2u_dx2_r
        self.d2u_dx2_i = d2u_dx2_i
        return(N)

    # dynamic_step() is intended to modify dt as it runs, in order
    # to use the largest time step that appears to be stable.
    #
    # dynamic_step() will run for run_time seconds of simulation time.
    # However, if it hits break_time seconds of real time, it will
    # return even if it hasn't gotten to run_time.
    #
    def dynamic_step(self, run_time, break_time):
        real_time = time()

        cdef int x # Used in x loop below
        cdef float m = self.m
        cdef float hbar = self.hbar
        cdef float hBy2m = hbar/(2*m)
        cdef float over_hbar = 1/hbar
        cdef float vByh # Used in xloop below
        cdef int npts = self.npts
        cdef int nx = self.npts-1 
        cdef float t = self.t
        cdef float dt = self.dt
        cdef float dx = self.dx
        cdef float ux0 = self.ux0
        cdef float uxL = self.uxL
        cdef int N = 0
        cdef float max_point_change
        cdef float dsfrac
        cdef float rdelta
        cdef float idelta
        cdef float temp

        # These make a HUGE difference
        cdef cnp.ndarray[cnp.float32_t, ndim=1] V = self.V
        cdef cnp.ndarray[cnp.float32_t, ndim=1] u_r = self.u_r
        cdef cnp.ndarray[cnp.float32_t, ndim=1] u_i = self.u_i
        cdef cnp.ndarray[cnp.float32_t, ndim=1] du_dx_r = self.du_dx_r
        cdef cnp.ndarray[cnp.float32_t, ndim=1] du_dx_i = self.du_dx_i
        cdef cnp.ndarray[cnp.float32_t, ndim=1] d2u_dx2_r = self.d2u_dx2_r
        cdef cnp.ndarray[cnp.float32_t, ndim=1] d2u_dx2_i = self.d2u_dx2_i

        target_time = t + run_time
        while t < target_time:
            if time() - real_time > break_time: break
            #
            # Calculate the derivatives
            #
            c_derivative(u_r, du_dx_r, npts, dx)
            c_derivative(u_i, du_dx_i, npts, dx)
            # NOTE: We don't need to look at "derivative" BCs
            c_derivative(du_dx_r, d2u_dx2_r, npts, dx)
            c_derivative(du_dx_i, d2u_dx2_i, npts, dx)
            #
            # Updated the probability wave
            #
            max_point_change = 0.0 # Track the largest change
            for x in range(0, npts):
                vByh  = V[x]*over_hbar
                du_dt_r = -hBy2m * d2u_dx2_i[x] + vByh * u_i[x]
                du_dt_i =  hBy2m * d2u_dx2_r[x] - vByh * u_r[x]
                # Track the amount of change
                rdelta = du_dt_r*dt
                idelta = du_dt_i*dt
                temp = abs(rdelta) 
                if temp > max_point_change: max_point_change = temp
                temp = abs(idelta)
                if temp > max_point_change: max_point_change = temp
                u_r[x] = u_r[x] + rdelta
                u_i[x] = u_i[x] + idelta
            # Potentially handle Dirichlet (only) boundary conditions here
            if self.using_bc:
                u_r[0] = ux0
                u_i[0] = ux0
                u_r[nx] = uxL
                u_i[nx] = uxL
            # Update the simulation time
            t += dt
            # Count number of iterations
            N += 1
            # Dynamic stuff ...
            dsfrac = max_point_change / self.dsrange
            # If we should slip over the top of the allowed range, print
            # a warning and cut dt in half
            if dsfrac > self.dsfrac:
                print('Warning: dsfrac=%s, limit=%s'%(dsfrac, self.dsfrac))
                dt = dt/2
            # If we get to 90% of the limit, take 10% off dt
            elif dsfrac > .9*self.dsfrac: dt = .9*dt
            # If we fall below half the limit, add 10% to dt
            elif dsfrac < .5*self.dsfrac: dt = 1.1*dt
            else: pass

        self.t = t
        self.dt = dt
        self.dt = dt
        self.u_r = u_r
        self.u_i = u_i
        self.du_dx_r = du_dx_r
        self.du_dx_i = du_dx_i
        self.d2u_dx2_r = d2u_dx2_r
        self.d2u_dx2_i = d2u_dx2_i
        return(N)

