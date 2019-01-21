# -*- coding: utf8 -*-
# This file runs the restricted shape Newton method with a 2D mesh
# and 1 mesh refinement.
#
# Designed for FEniCS 2018.1.

# Import the class for the Newton method
from RestrictedShapeNewtonMethod import *

# Do not bother us with many messages
set_log_level(LogLevel.WARNING)
import logging
logging.getLogger('FFC').setLevel(logging.WARNING)

# Load a 2D mesh
mesh = Mesh("lshape.xml.gz")
mesh.bounding_box_tree().build(mesh)

# Set the rhs for the PDE
rhs = "Constant(2.5)*(x + Constant(0.4) - y**2)**2 + x**2 + y**2 + - Constant(1.0)"

# Generate an instance of shape optimization problem in 3D and solve it
# using the novel restricted Newton method, then refine the mesh, and run again
problem = RestrictedShapeNewtonMethod(mesh = Mesh(mesh), rhs = rhs, JSONinput = "input_newton_2D.json")
problem.run()
refinements = 1
for i in range(refinements):
	problem.mesh = refine(problem.mesh)
	problem.run()

