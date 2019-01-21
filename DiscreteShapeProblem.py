# -*- coding: utf8 -*-
# FEniCS 2018.1
from dolfin import *
from ufl import replace
from DiscreteShapeProblemUtils import *
import numpy as np
import importlib

# This class implements the state and adjoint problems, as well as the 
# evaluation of the objective, the shape derivative and the (negative) shape gradient w.r.t. 
# the elasticity and L2 inner products. Moreover, mesh movement and restoration methods
# are provided as well as the check of the geometric condition (is_step_valid).
# Finally, a finite difference test for the shape derivative is included.
class DiscreteShapeProblem:
	def __init__(self, mesh, rhs, E = 1.0, nu = 0.4, damping_factor = 0.2):
		
		# Get the mesh and its dimension
		self.mesh = mesh
		self.dim = self.mesh.geometry().dim()

		# Set the right hand side of the pde
		self.rhs = rhs

		# Set elasticity parameters (E,nu) for the gradient computation w.r.t the elasticity inner product
		self.E = E
		self.nu = nu

		# Set the damping factor delta > 0 for the elasticity operator (elasticity inner product)
		self.damping = damping_factor * self.E    # delta (damping term)

		# Distinct between elements for 2D/3D case
		if self.dim == 2:
			self.cell = triangle
		else:
			self.cell = tetrahedron

		# Create the function spaces
		self.V = FunctionSpace(self.mesh, "CG", 1)
		self.U = VectorFunctionSpace(self.mesh, "CG", 1)
		self.DG0 = FunctionSpace(self.mesh, "DG", 0)

		# Create homogeneous Dirichlet boundary conditions for the PDE
		def boundary_D(x, on_boundary):
			return on_boundary
		self.bc = DirichletBC(self.V, Constant(0.0), boundary_D)

		# Create the functions to hold the state and adjoint state. These are used in all the forms.
		# Therefore, compute_state() should be called before compute_obj().
		self.u = Function(self.V)
		self.u.rename("u", "u")
		self.p = Function(self.V)
		self.p.rename("p", "p")

		# Provide some space to store an old state and adjoint state when the mesh is moved.
		# This is neccessary in case the mesh has to be restored.
		self.old_u = Function(self.V)
		self.old_p = Function(self.V)

		# Create a boolean to remember whether [u,p] is valid on the current mesh
		self.has_u = False
		self.has_p = False

	# This function evaluates the rhs of the PDE (given string in self.rhs)
	def f(self, x, y, z = None):
		# Dictionary for allowed operators in eval string (rhs)
		ufl_ops = {}

		# Generate a dictionary with all ufl operators, so that they are allowed in eval
		ops = importlib.import_module("ufl.operators")
		ufl_ops = ops.__dict__
		# Add coordinates to be allowed in eval
		ufl_ops['x'] = x
		ufl_ops['y'] = y
		ufl_ops['z'] = z
		# Add the dolfin Constant() to be allowed in eval
		ufl_ops['Constant'] = Constant

		# If self.rhs is a string, evaluate it
		if (self.rhs) and (isinstance(self.rhs, str)):
			return eval(self.rhs,ufl_ops)
		else:
			raise Exception("The rhs has to be a string.")


	# This function evaluates the Lagrangian
	def lagrangian(self, V, u, p):

		# Compute the deformation gradient (the Jacobian of the transformation)
		F = Identity(self.dim) + grad(V)

		# Some helper routines
		n = FacetNormal(self.mesh)
		dxV = WeightedMeasureWrapper(det(F), dx(self.mesh))
		dsV = WeightedMeasureWrapper(det(F) * sqrt(inner(inv(F).T * n,inv(F).T * n)), ds(self.mesh))
		def gradV(f):
			return inv(F.T) * grad(f)

		# Get the spatial coordinates
		x = SpatialCoordinate(self.mesh)

		# Define the objective
		obj = u * dxV

		# Define the weak form of the PDE
		pde = inner(gradV(u),gradV(p)) * dxV - self.f(*[x[i] + V[i] for i in range(self.dim)]) * p * dxV

		# Build and return the Lagrangian as a UFL form
		return obj + pde


	# This function evaluates the objective from the Lagrangian UFL form
	def compute_objective(self):
		# Compute the state 
		self.compute_state()

		# Prepare a form for the objective if it doesn't exist
		if not hasattr(self, 'obj'):
			Vref = Constant([0.0] * self.dim, self.cell)
			p = Constant(0.0, self.cell)
			self.obj = Form(self.lagrangian(Vref, self.u, p))
		
		# Compute and return the objective
		return assemble(self.obj)


	# This function solves the PDE for the state variable
	def compute_state(self):
		# Prepare PDE solver if it doesn't exist
		if not hasattr(self, 'pde'):
			# Prepare a solver for the PDE
			u = Function(self.V)
			p = Function(self.V)
			Vref = Constant([0.0] * self.dim, self.cell)

			L = self.lagrangian(Vref, u, p)
			pde_form = derivative(L, p, TestFunction(self.V))
			pde_form = replace(pde_form, {u: TrialFunction(self.V)})

			problem = LinearVariationalProblem(lhs(pde_form), rhs(pde_form), self.u, self.bc)
			solver = LinearVariationalSolver(problem)

			self.pde = dict()
			self.pde["solver"] = solver

		# Solve the PDE and store the solution in self.u
		if not self.has_u:
			# Solve the PDE
			self.pde["solver"].solve()
			self.has_u = True


	# This function solves the adjoint PDE 
	def compute_adjoint(self):
		# Compute the state 
		self.compute_state()
		# Prepare a solver for the adjoint PDE if it doesn't exist
		if not hasattr(self, 'adjoint'):
			# Prepare a solver for the adjoint PDE
			u = Function(self.V)
			p = Function(self.V)
			Vref = Constant([0.0] * self.dim, self.cell)

			L = self.lagrangian(Vref, u, p)
			adjoint_form = derivative(L, u, TestFunction(self.V))
			adjoint_form = replace(adjoint_form, {p: TrialFunction(self.V)})
			adjoint_form = replace(adjoint_form, {u: self.u})

			problem = LinearVariationalProblem(lhs(adjoint_form), rhs(adjoint_form), self.p, self.bc)
			solver = LinearVariationalSolver(problem)

			self.adjoint = dict()
			self.adjoint["solver"] = solver

		# Solve the adjoint PDE and store the solution in self.p 
		if not self.has_p:
			# Solve the adjoint PDE
			self.adjoint["solver"].solve()
			self.has_p = True


	# This function stores and returns a form for computing the shape derivative.
	# Before using the form, one has to call compute_adjoint()!
	def get_shape_derivative_form(self):
		# Prepare the form
		if not hasattr(self, 'shape'):
			# Prepare the bilinear form and the linear form of the adjoint PDE
			Vref = Function(self.U)

			L = self.lagrangian(Vref, self.u, self.p)
			V = TestFunction(self.U)
			shape_form = derivative(L, Vref, V)

			Vzero = Constant([0.0] * self.dim, self.cell)
			shape_form = replace(shape_form, {Vref: Vzero})

			self.shape = dict()
			self.shape["form"] = shape_form

		# Return the shape derivative as a UFL form
		return self.shape["form"]


	# This function computes the (negative) shape gradient w.r.t. the elasticity inner product 
	def compute_shape_gradient(self, V):
		# Compute state and the adjoint
		self.compute_state()
		self.compute_adjoint()

		# Prepare the shape gradient solver
		if not hasattr(self, 'shape_grad'):
			def eps(u):
				return sym(grad(u))
			V_ = Function(self.U)

			# Setup the Lame parameters
			lame1 = self.nu * self.E /((1+self.nu)*(1-2*self.nu))  # lambda
			lame2 = self.E / (2*(1+self.nu))                       # mu

			# Setup the elasticity operator
			W = TestFunction(self.U)
			W2 = TrialFunction(self.U)
			elasticity_operator = \
					2*Constant(lame2) * inner(eps(W2), eps(W)) * dx \
					+ Constant(lame1) * div(W2) * div(W) * dx \
					+ Constant(self.damping) * inner(W,W2) * dx

			shape_gradient_problem = LinearVariationalProblem(elasticity_operator, -self.get_shape_derivative_form(), V_)
			shape_gradient_solver = LinearVariationalSolver(shape_gradient_problem)

			self.shape_grad = dict()
			self.shape_grad["solver"] = shape_gradient_solver
			self.shape_grad["V"] = V_
			self.shape_grad["lhs"] = elasticity_operator
		
		# Compute the elasticity shape gradient
		self.shape_grad["solver"].solve()
		V.assign(self.shape_grad["V"])


	# This function computes the (negative) shape gradient w.r.t. the L^2 inner product
	# It is merely useful for visualization of the shape derivative
	def compute_shape_gradient_L2(self, V):
		# Compute state and the adjoint
		self.compute_state()
		self.compute_adjoint()
		# Prepare the shape gradient solver
		if not hasattr(self, 'shape_grad_L2'):
			V_ = Function(self.U)

			# Setup the L2 inner product
			W = TestFunction(self.U)
			W2 = TrialFunction(self.U)
			lhs_L2 = inner(W,W2) * dx

			shape_gradient_L2_problem = LinearVariationalProblem(lhs_L2, -self.get_shape_derivative_form(), V_)
			shape_gradient_L2_solver = LinearVariationalSolver(shape_gradient_L2_problem)

			self.shape_grad_L2 = dict()
			self.shape_grad_L2["solver"] = shape_gradient_L2_solver
			self.shape_grad_L2["V"] = V_
		
		# Compute the L2 shape gradient
		self.shape_grad_L2["solver"].solve()
		V.assign(self.shape_grad_L2["V"])


	# This function checks whether the mesh displacement step alpha * V is geometrically valid, i.e., whether
	#    0.5 <= det(I + alpha DV) <= 2
	#   alpha * || DV ||_F <= 0.3
	# holds in all cells
	def is_step_valid(self, alpha, V):

		# Prepare a solver to evaluate the L2 projection of det(I + alpha DV) onto piecewise constants,
		# as well as for the cell-wise Frobenius norm of DV
		if not hasattr(self, 'validity_solver'):
			alpha_constant = Constant(alpha)
			V_ref = Function(self.U)
			a_det = TestFunction(self.DG0) * TrialFunction(self.DG0) * dx
			L_det = det(Identity(self.dim) + alpha_constant*grad(V_ref)) * TestFunction(self.DG0) * dx

			solution = Function(self.DG0)

			problem = LinearVariationalProblem(a_det, L_det, solution)
			solver = LinearVariationalSolver(problem)

			self.validity_solver = dict()
			self.validity_solver["solver_det"] = solver
			self.validity_solver["solution_det"] = solution
			self.validity_solver["alpha"] = alpha_constant
			self.validity_solver["V"] = V_ref

			# Now check the norm of the deformation gradient
			a_defgrad = TrialFunction(self.DG0) * TestFunction(self.DG0) * dx
			L_defgrad = sqrt(inner(grad(V_ref),grad(V_ref))) * TestFunction(self.DG0) * dx

			normf = Function(self.DG0)

			problem = LinearVariationalProblem(a_defgrad, L_defgrad, normf)
			solver = LinearVariationalSolver(problem)

			self.validity_solver["solver_defgrad"] = solver
			self.validity_solver["solution_defgrad"] = normf
		# End of preparing solver

		# Assign alpha and V to the solver
		self.validity_solver["alpha"].assign(alpha)
		self.validity_solver["V"].assign(V)

		# Compute the determinants on all cells and find the smallest and largest 
		self.validity_solver["solver_det"].solve()
		min_det = self.validity_solver["solution_det"].vector().get_local().min()
		max_det = self.validity_solver["solution_det"].vector().get_local().max()

		# Compute the Frobenius norm on all cells of the deformation gradient
		# and find the largest one, scaled by alpha
		self.validity_solver["solver_defgrad"].solve()
		max_def_grad = alpha * (self.validity_solver["solution_defgrad"].vector().get_local().max())

		# Return the result of the validity check
		return (min_det >= 0.5) and (max_det <= 2.0) and (max_def_grad <= 0.3)


	# This function moves the mesh by a given displacement field and stores the old state and adjoint state.
	# This is neccessary in case the mesh has to be restored.
	def move_mesh(self, displacement):
		self.old_coordinates = self.mesh.coordinates().copy()
		self.old_u.assign(self.u)
		self.old_has_u = self.has_u
		self.old_p.assign(self.p)
		self.old_has_p = self.has_p
		ALE.move(self.mesh, displacement)
		# Invalidate any previously computed state and adjoint state
		self.has_u = False 
		self.has_p = False


	# This function restores the mesh to old coordinates (stored before mesh move)
	# and restores the corresponding state and adjoint state
	def restore_mesh(self):
		self.mesh.coordinates()[:] = self.old_coordinates
		self.u.assign(self.old_u)
		self.has_u = self.old_has_u
		self.p.assign(self.old_p)
		self.has_p = self.old_has_p


	# This function checks the shape derivative. 
	def check_shape_derivative(self, V, dobj, W = None, dot_W = None, dot_u = None, dot_p = None):
		# Helper stuff {{{
		bmesh = BoundaryMesh(self.mesh, 'exterior')
		boundary_dofmap = [bmesh.entity_map(0)[i] for i in range(bmesh.num_vertices())]
		# }}}
		expected_derivative = dobj
		print("====== BEGIN FD ======")
		print("\nCheck correctness of shape derivative\n")
		print("Expected derivative: %e" % expected_derivative)

		self.compute_state()
		u = Function(self.V)
		u.interpolate(self.u)

		self.compute_adjoint()
		p = Function(self.V)
		p.interpolate(self.p)

		obj = self.compute_objective()

		if dot_u != None:
			print("Norm dot_u: %e" % sqrt(assemble(dot_u**2*dx)))

		if dot_p != None:
			print("Norm dot_p: %e" % sqrt(assemble(dot_p**2*dx)))

		if dot_W != None:
			print("Norm dot_W: %e" % sqrt(assemble(inner(dot_W,dot_W)*dx)))

		steps = range(+2, -12, -1)
		# steps = [1e-8]
		# steps = []

		for step_power in steps:
			# Move mesh
			step = - 10**step_power

			V_step = Function(self.U)
			V_step.vector()[:] = step * V.vector()[:]

			self.move_mesh(V_step)

			self.compute_state()
			u_step = Function(self.V)
			u_step.assign(self.u)

			self.compute_adjoint()
			p_step = Function(self.V)
			p_step.assign(self.p)

			obj_step = self.compute_objective()
			numerical_derivative = (obj_step - obj) / step
			string = "step = %e, FD: %+e" % (step, numerical_derivative)

			if dot_W != None:
				# Now, we are going to compute the projected shape gradient
				t = Timer("Block system setup")
				A = PETScMatrix()
				assemble(self.shape_grad["lhs"], tensor = A)
				A = to_scipy(A)

				N = PETScMatrix()
				assemble(inner(TrialFunction(self.U), FacetNormal(self.mesh)) * TestFunction(self.V) * ds, tensor=N)
				N = to_scipy(N)
				N.eliminate_zeros()
				N = N[boundary_dofmap,:]

				Q = scipy.sparse.bmat([[A, None, A], [None, None, -N], [A, -N.T, None]], format='csr')

				# Assemble rhs
				dJ = PETScVector()
				assemble(self.get_shape_derivative_form(), tensor = dJ)
				rhs = np.hstack([-dJ.get_local(), np.zeros([bmesh.num_vertices()+2*self.mesh.num_vertices(),])])
				t.stop()

				# Solve linear system
				t = Timer("Block system solve")
				# For some reason, UMFPack is slower than SuperLU here:
				WFPi = spsolve(Q, rhs, use_umfpack=False)
				t.stop()

				W_step = Function(self.U)
				W_step.vector()[:] = WFPi[0:2*self.mesh.num_vertices()]

			# Move mesh back
			self.restore_mesh()

			if u_step != None and dot_u != None:
				# Check the material derivative
				du = (u_step - u) / Constant(step) - dot_u
				err = sqrt(assemble(du * du * dx))

				string += ",  err_u = %e" % err

			if p_step != None and dot_p != None:
				# Check the material derivative
				dp = (p_step - p) / Constant(step) - dot_p
				err = sqrt(assemble(dp * dp * dx))

				string += ",  err_p = %e" % err

			if dot_W != None:
				# Check the material derivative
				# dW = (W_step - W) / Constant(step) - dot_W
				W_step.vector()[:] -= W
				dW = (W_step) / Constant(step) - dot_W
				err = sqrt(assemble(inner(dW , dW) * dx))

				string += ",  err_W = %e" % err

			# Report
			print(string)
		print("====== END FD ======")
	# End of check_shape_derivative()

# vim: fdm=marker noet

