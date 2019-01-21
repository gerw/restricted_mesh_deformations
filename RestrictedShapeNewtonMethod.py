from SuperMethod import *
from DiscreteShapeProblem import *
from DiscreteShapeProblemUtils import *
from scipy.sparse.linalg import spsolve
import scipy.sparse
import csv

# This class extends the SuperMethod class by methods specific for the 
# restricted shape Newton method.
class RestrictedShapeNewtonMethod(SuperMethod):
	def __init__(self, mesh, rhs, JSONinput = None):

		# Set default parameters 
		self.options = {} 
		
		# Set directory where the output is stored
		self.options["directory"] = "solutions/restricted_newton/"

		# Set export and verbosity options
		self.options["export"] = 1
		self.options["verbosity"] = 1

		# Set default algorithmic parameters 
		self.options["maxiter"] = 100     # maximum number of iterations
		self.options["sigma"] = 0.1       # Armijo slope parameter
		self.options["beta"] = 0.1        # Armijo backtracking parameter
		self.options["alpha0"] = 1e-2     # initial damping parameter for the line search

		# Set absolute tolerance for the elasticity norm of the restricted gradient
		self.options["StoppingTolerance"] = 1e-9 

		# Set rhs function in the state equation
		if isinstance(rhs, str):
			self.options["rhs"] = rhs
		else:
			raise Exception("The rhs for the state equation has to be a string.")
		
		# Verify the mesh
		if type(mesh) == cpp.mesh.Mesh: 
			self.mesh = mesh
		else:
			raise Exception("No valid dolfin mesh was passed to the class.")
		
		# Set elasticity parameters (E,nu) for the gradient computation
		self.options["E"] = 1.0
		self.options["nu"] = 0.4

		# Set the damping factor. Later used as delta > 0, with delta = damping_factor * E, 
		# for the elasticity operator (elasticity inner product)
		self.options["damping_factor"] = 0.2

		# If we have an input file, read it
		if JSONinput:
			if isinstance(JSONinput, str):
				self.load_config(JSONinput)
			else:
				raise Exception("The name of the JSON input file has to be a string.")
		
		# Open output files
		if self.options["export"] > 0:
			self.open_files()

	# End of __init__()

	# This function opens files for export
	def open_files(self):
		directory = self.options["directory"] if self.options["directory"] else "./"
		self.f1 = File(directory + 'solution.pvd')
		self.f2 = File(directory + 'shape_grad_L2.pvd')
		self.f3 = File(directory + 'shape_grad.pvd')
		self.f4 = File(directory + 'normal_force.pvd')
		self.f5 = File(directory + 'shape_grad_elast.pvd')
		self.f6 = open(directory + 'history.txt','w')
		super().open_files()

	# This function closes files for export, called by __del__()
	def close_files(self):
		self.f6.close()
		super().close_files()

	# This function implements the restricted shape Newton method
	def run(self):
		# We do not want to care about DoF maps
		parameters["reorder_dofs_serial"] = False

		# Get some local variables
		maxiter = self.options["maxiter"]
		sigma = self.options["sigma"]
		beta = self.options["beta"]

		# Setup the shape problem 
		dsp = DiscreteShapeProblem(mesh = self.mesh, rhs = self.options["rhs"], E = self.options["E"], nu = self.options["nu"], damping_factor = self.options["damping_factor"])

		# Setup the function space for the shape displacements
		W = Function(dsp.U)
		W.rename("disp", "disp")
		W_step = Function(dsp.U)

		# Create a form for the directional shape derivative (depends on W)
		directional_shape_derivative = Form(action(dsp.get_shape_derivative_form(),W))

		# Create quantities on the boundary mesh
		bmesh = BoundaryMesh(dsp.mesh, 'exterior')
		boundary_dofmap = [bmesh.entity_map(0)[i] for i in range(bmesh.num_vertices())]
		bW = FunctionSpace(bmesh, "CG", 1)

		# Create the normal force as a function on the boundary
		f = Function(bW)
		f.rename("normal_force","normal_force")

		# Write history header to file/bash if export/verbosity
		if self.options["export"] > 0:
			csv.writer(self.f6).writerow(('%4s' % "iter", '%13s' % "objective", '%13s' % "dirderivative", '%13s' % "alpha", '%13s' % "proj_grad_norm"))
		if self.options["verbosity"] > 0:
			csv.writer(sys.stdout).writerow(('%4s' % "iter", '%13s' % "objective", '%13s' % "dirderivative", '%13s' % "alpha", '%13s' % "proj_grad_norm"))

		# Needed to prepare some forms
		dsp.compute_shape_gradient(W)

		# Enter the restricted Newton loop
		for j in range(maxiter):

			# Compute the objective and the (negative) shape gradient w.r.t. the elasticity inner product in \Omega
			obj = dsp.compute_objective()
			dsp.compute_adjoint()
			dsp.compute_shape_gradient_L2(W)

			if self.options["export"] > 0:
				# Output the solution
				self.f1 << dsp.u
				self.f2 << W

			# Compute the restricted shape gradient
			# Start a timer for setting up the block system
			t = Timer("Block system setup")
			E = PETScMatrix()
			assemble(dsp.shape_grad["lhs"], tensor = E)
			E = to_scipy(E)

			N = PETScMatrix()
			assemble(inner(TestFunction(dsp.U), FacetNormal(dsp.mesh)) * TrialFunction(dsp.V) * ds, tensor = N)
			N = to_scipy(N)
			N.eliminate_zeros()
			N = N[:,boundary_dofmap]

			Q = scipy.sparse.bmat([[E, None, E], [None, None, -N.T], [E, -N, None]], format='csr')

			# Assemble rhs
			dJ = PETScVector()
			assemble(dsp.get_shape_derivative_form(), tensor = dJ)
			rhs = np.hstack([-dJ.get_local(), np.zeros([bmesh.num_vertices()+dsp.dim*dsp.mesh.num_vertices(),])])

			# Stop timer (Block system setup)
			t.stop()

			# Start a timer for solving the block system
			t = Timer("Block system solve")

			# Solve the linear system
			# For some reason, UMFPack is slower than SuperLU here:
			V_proj_gradFPi = spsolve(Q, rhs, use_umfpack=False)

			# Stop timer (Block system solve)
			t.stop()

			# Extract the restricted shape gradient (V_proj_grad), the normal force (f), and the Lagrange multiplier (Pi)
			V_proj_grad = V_proj_gradFPi[0:dsp.dim*dsp.mesh.num_vertices()]
			f.vector()[:] = V_proj_gradFPi[dsp.dim*dsp.mesh.num_vertices():dsp.dim*dsp.mesh.num_vertices()+bmesh.num_vertices()]
			Pi = V_proj_gradFPi[dsp.dim*dsp.mesh.num_vertices()+bmesh.num_vertices():]

			# Copy the restricted shape gradient into W
			W.vector()[:] = V_proj_grad

			if self.options["export"] > 0:
				# Output the normal force and the RESTRICTED elasticity shape gradient
				self.f4 << f
				self.f5 << W

			# End of computing the restricted shape gradient

			# Compute the directional shape derivative and the norm of the shape gradient
			d_obj = assemble(directional_shape_derivative) # depends on W
			proj_grad_norm = sqrt(abs(d_obj))

			# Check for convergence
			if proj_grad_norm < self.options["StoppingTolerance"]:
				# Prepare the output string
				output = ('%4d' % j, '%13.4e' % obj, '%13.4e' % d_obj, '%13.4e' % 0, '%13.4e' % proj_grad_norm)
				# Output iteration data to history file and stdout
				if self.options["export"] > 0:
					csv.writer(self.f6).writerow(output)
				if self.options["verbosity"] > 0:
					csv.writer(sys.stdout).writerow(output)
				break

			# Now, we try to compute a Newton step
			# Setup a function space for the unknown h_fun
			VV = FunctionSpace(dsp.mesh, MixedElement([
					FiniteElement("CG", dsp.cell, 1),
					VectorElement("CG", dsp.cell, 1),
					FiniteElement("CG", dsp.cell, 1),
					FiniteElement("CG", dsp.cell, 1),
					VectorElement("CG", dsp.cell, 1),
					FiniteElement("CG", dsp.cell, 1),
					VectorElement("CG", dsp.cell, 1),
				]))
			h_fun = Function(VV)
			[hf, hW, hu, hp, hV_proj_grad, hF, hPi] = split(h_fun)

			# Copy the normal force f into the boundary dofs of F
			F = np.zeros([dsp.mesh.num_vertices(),])
			F[boundary_dofmap] = f.vector()

			# Setup the test and trial functions for the Newton system
			[df, dW, du, dp, dV_proj_grad, dF, dPi] = TestFunctions(VV)
			d_fun = TrialFunction(VV)

			# Some helper routines {{{
			# Deformation gradient associated with hW, this is the Jacobian of the transformation
			FF = Identity(dsp.dim) + grad(hW)
			n = FacetNormal(dsp.mesh)
			nW = (inv(FF).T*n) / (sqrt(inner(inv(FF).T*n,inv(FF).T*n)))

			dxW = WeightedMeasureWrapper(det(FF), dx(dsp.mesh))
			dsW = WeightedMeasureWrapper(det(FF) * sqrt(inner(inv(FF).T * n,inv(FF).T * n)), ds(dsp.mesh))

			# Setup some operators for the formulation of the Newton residual
			def divW(f):
				return tr(inv(FF.T)*grad(f).T)
			def eps(f):
				return sym(grad(f))
			def epsW(f):
				return sym(inv(FF.T)*grad(f).T)

			# Setup the Lame parameters
			lame1 = self.options["nu"] * self.options["E"] /((1+self.options["nu"])*(1-2*self.options["nu"]))  # lambda
			lame2 = self.options["E"] / (2*(1+self.options["nu"]))                                             # mu
			# Set the damping factor delta > 0 for the elasticity operator (elasticity inner product)
			damping = self.options["damping_factor"] * self.options["E"]    # delta (damping term)

			# Set a regularization parameter for interior components of boundary functions
			meps = 1e-26

			# Setup the zero Dirichlet boundary conditions for the state and adjoint PDEs
			def boundary_D(x, on_boundary):
				return on_boundary
			nonlin_bc = [DirichletBC(VV.sub(2), Constant(0.0), boundary_D), DirichletBC(VV.sub(3), Constant(0.0), boundary_D)]

			# Perform a line search
			if j == 0:
				# Set initial damping parameter (at first iteration)
				alpha = self.options["alpha0"]
			else:
				# Increase alpha a little bit
				alpha /= beta

			while True:
				# Setup the iterate for the Newton method
				h_fun.vector()[:] = \
					np.hstack([
						np.zeros([dsp.mesh.num_vertices(),]),
						np.zeros([dsp.dim * dsp.mesh.num_vertices(),]),
						dsp.u.vector(),
						dsp.p.vector(),
						V_proj_grad,
						F,
						Pi
					])

				# Evaluate the Newton residual
				nonlin_res = \
						(hF * df * ds) \
						+ 2*Constant(lame2) * inner(eps(hW), eps(dW)) * dx + Constant(lame1) * div(hW) * div(dW) * dx + Constant(damping) * inner(hW,dW) * dx - hf * inner(dW, n) * ds \
						+ derivative(dsp.lagrangian(hW, hu, hp), hu, du) \
						+ derivative(dsp.lagrangian(hW, hu, hp), hp, dp) \
						+ 2*Constant(lame2) *inner(epsW(hV_proj_grad+hPi), epsW(dV_proj_grad)) * dxW + Constant(lame1) * divW(hV_proj_grad+hPi) * divW(dV_proj_grad) * dxW + Constant(damping) * inner(hV_proj_grad+hPi,dV_proj_grad) * dxW \
						+ derivative(dsp.lagrangian(hW, hu, hp), hW, dV_proj_grad) \
						- dF * inner(hPi,nW) * dsW \
						+ 2*Constant(lame2) * inner(epsW(hV_proj_grad), epsW(dPi)) * dxW + Constant(lame1) * divW(hV_proj_grad) * divW(dPi) * dxW + Constant(damping) * inner(hV_proj_grad,dPi) * dxW \
						- hF * inner(dPi,nW) * dsW \
						+ Constant(meps) * (hF * dF + hf * df) * dx

				# Add the damping term to the residual
				nonlin_res += Constant(-1./alpha) * hf * df * ds

				# Solve the Newton system, where the Newton matrix is created by algorithmic differentiation
				solve(derivative(nonlin_res, h_fun, d_fun) == -nonlin_res, h_fun, nonlin_bc)

				# Extract the displacement field  component W from the solution of the Newton step
				W.vector()[:] = h_fun.vector().get_local()[dsp.mesh.num_vertices():(dsp.dim+1) * dsp.mesh.num_vertices()]

				# Compute the directional shape derivative 
				d_obj = assemble(directional_shape_derivative) # depends on W
				# dsp.check_shape_derivative(W, d_obj, dot_u = hu, dot_p = hp, W = V_proj_grad, dot_W = hV_proj_grad)

				# Prepare the output string
				output = ('%4d' % j, '%13.4e' % obj, '%13.4e' % d_obj, '%13.4e' % alpha, '%13.4e' % proj_grad_norm)

				# Check for validity of step size and whether we have a descent direction
				if dsp.is_step_valid(1, W) and d_obj < 0:
					# Move the mesh by the displacement field W
					W_step.vector()[:] = W.vector()[:]
					dsp.move_mesh(W_step)

					# Compute the objective on the displaced mesh
					obj_step = dsp.compute_objective()

					# Check the Armijo condition
					if obj_step < obj + sigma * d_obj:
						break
					else:
						# Output iteration data to history file and stdout
						if self.options["export"] > 0:
							csv.writer(self.f6).writerow(output + ('%s' % "Armijo condition failed",))
						if self.options["verbosity"] > 0:
							csv.writer(sys.stdout).writerow(output + ('%s' % "Armijo condition failed",))

						# If not, restore the mesh 
						dsp.restore_mesh()
				else:
					if not dsp.is_step_valid(1, W):
						# Output iteration data to history file and stdout
						if self.options["export"] > 0:
							csv.writer(self.f6).writerow(output + ('%s' % "geometry condition failed",))
						if self.options["verbosity"] > 0:
							csv.writer(sys.stdout).writerow(output + ('%s' % "geometry condition failed",))

				# Decrease line search parameter
				alpha *= beta

				# If alpha is too small, something went wrong
				if alpha < 1e-10:
					raise Exception("Damping failed")

			# Output iteration data to history file and stdout
			if self.options["export"] > 0:
				csv.writer(self.f6).writerow(output)
			if self.options["verbosity"] > 0:
				csv.writer(sys.stdout).writerow(output)

	# End of run()

# End of RestrictedShapeNewtonMethod class

# Do a standard run of the restricted shape Newton method
if __name__ == "__main__":
	print("Run standard configuration")
	# Create a 2D mesh
	meshlevel = 12
	degree = 1
	dim = 2
	mesh = UnitDiscMesh.create(MPI.comm_world, meshlevel, degree, dim)
	# Set the rhs for the PDE
	rhs = "Constant(2.5)*(x + Constant(0.4) - y**2)**2 + x**2 + y**2 + - Constant(1.0)"
	problem = RestrictedShapeNewtonMethod(mesh = Mesh(mesh), rhs = rhs)
	problem.run()

# vim: fdm=marker noet
