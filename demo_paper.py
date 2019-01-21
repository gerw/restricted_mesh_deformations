# -*- coding: utf8 -*-
# This file creates the numerical results using the 
#
# * gradient method,
# * projected gradient method,
# * and Newton method (2D and 3D)
# for shape optimization problems as presented in
#
# Etling, Herzog, Loayza, Wachsmuth: First and Second Order Shape Optimization
# based on Restricted Mesh Deformations
#
# Designed for FEniCS 2018.1.

# Import classes implementing the three methods
from ShapeGradientMethod import *
from RestrictedShapeGradientMethod import *
from RestrictedShapeNewtonMethod import *

# Do not bother us with many messages
set_log_level(LogLevel.WARNING)
import logging
logging.getLogger('FFC').setLevel(logging.WARNING)

# Create a 2D mesh
meshlevel = 12
degree = 1
dim = 2
mesh = UnitDiscMesh.create(MPI.comm_world, meshlevel, degree, dim)

# Set the rhs for the PDE
rhs = "Constant(2.5)*(x + Constant(0.4) - y**2)**2 + x**2 + y**2 - Constant(1.0)"

# Generate an instance of the example shape optimization problem and solve it
# using a standard gradient method with line search
ShapeGradientMethod(mesh = Mesh(mesh), rhs = rhs, JSONinput = 'input_paper_gradient_method.json').run()

# Generate an instance of the example shape optimization problem and solve it
# using the novel projected gradient method with line search
RestrictedShapeGradientMethod(mesh = Mesh(mesh), rhs = rhs, JSONinput = 'input_paper_restricted_gradient_method.json').run()

# Generate an instance of the example shape optimization problem and solve it
# using the novel projected Newton method with damping
RestrictedShapeNewtonMethod(mesh = Mesh(mesh), rhs = rhs, JSONinput = 'input_paper_restricted_newton_method.json').run()


# Create a 3D mesh 
meshlevel = 4
mesh = BoxMesh(Point(-0.5,-0.5,-0.5), Point(0.5,0.5,0.5), meshlevel, meshlevel, meshlevel)
mesh = refine(mesh)

# Set the rhs for the PDE
rhs = "Constant(2.5)*(x + Constant(0.4) - y**2)**2 + x**2 + y**2 + z**2 - Constant(1.0)"

# Generate an instance of the example shape optimization problem in 3D and solve it
# using the novel restricted Newton method with damping 
RestrictedShapeNewtonMethod(mesh = Mesh(mesh), rhs = rhs, JSONinput = 'input_paper_restricted_newton_method_3D.json').run()

