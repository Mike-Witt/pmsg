pmsg
====

Portland Math &amp; Science Group

This contains various programs and other stuff being discussed by the
Portland Math & Science group. It's all public. You should be able to get
a copy by saying:

	git clone https://github.com/Mike-Witt/pmsg.git

These files contain a graphical demo of Liouville's Theorem in phase space:

    Liouville1.html - A web page that expains what's going on
    Liouville1.py - The python code

These files contain a simple simulation of the time dependend Schroedinger
equations, using the finite difference method:

    csim_build.py - This builds the cython code
    csim_tdse.pyx - Cython code for the simulator
    tdse_demo.py - Python code demonstrating how to use the simulation
    tdse_profile.py - A version the does profiling

See the comments at the top of tdse_demo.py for more information.

