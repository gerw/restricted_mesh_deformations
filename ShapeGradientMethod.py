from SuperMethod import *
from DiscreteShapeProblem import *
import csv

# This class extends the SuperMethod class by methods specific for the 
# shape gradient method.
class ShapeGradientMethod(SuperMethod):
	def __init__(self, mesh, rhs, JSONinput = None):

		# Set default parameters 
		self.options = {}

		# Set directory where the output is stored
		self.options["directory"] = "solutions/gradient/"

		# Set export and verbosity options
		self.options["export"] = 1
		self.options["verbosity"] = 1

		# Set default algorithmic parameters 
		self.options["maxiter"] = 1000    # maximum number of iterations
		self.options["sigma"] = 0.1       # Armijo line search slope parameter
		self.options["beta"] = 0.5        # Armijo line search backtracking parameter
		self.options["alpha0"] = 1.0      # initial step size 

		# Set absolute tolerance for the elasticity norm of the restricted gradient
		self.options["StoppingTolerance"] = 1e-7 

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
				# Call function to overwrite default variables
				adjusted_options = self.load_config(JSONinput)
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
		self.f6 = open(directory + 'history.txt','w')
		super().open_files()

	# This function closes files for export, called by __del__()
	def close_files(self):
		self.f6.close()
		super().close_files()

	# This function implements the restricted shape Newton method
	def run(self):
		# Get some local variables
		maxiter = self.options["maxiter"]
		sigma = self.options["sigma"]
		beta = self.options["beta"]

		# Setup the shape problem 
		dsp = DiscreteShapeProblem(mesh = self.mesh, rhs = self.options["rhs"], E = self.options["E"], nu = self.options["nu"], damping_factor = self.options["damping_factor"])

		# Setup the function space for the shape displacements
		V = Function(dsp.U)
		V.rename("disp", "disp")
		V_step = Function(dsp.U)
		
		# Create a Form for the directional shape derivative (depends on V)
		directional_shape_derivative = Form(action(dsp.get_shape_derivative_form(),V))

		# Write history header to file/bash if export/verbosity
		if self.options["export"] > 0:
			csv.writer(self.f6).writerow(('%4s' % "iter", '%13s' % "objective", '%13s' % "dirderivative", '%13s' % "alpha", '%13s' % "grad_norm"))
		if self.options["verbosity"] > 0:
			csv.writer(sys.stdout).writerow(('%4s' % "iter", '%13s' % "objective", '%13s' % "dirderivative", '%13s' % "alpha", '%13s' % "grad_norm"))

		# Enter the gradient loop
		for j in range(maxiter):
			
			# Compute the objective
			obj = dsp.compute_objective()

			if self.options["export"] > 0:
				# Output the solution 
				self.f1 << dsp.u
			
			if self.options["export"] > 0:
				# Compute the (negative) shape gradient w.r.t. the L^2 inner product in \Omega (for visualization)
				dsp.compute_shape_gradient_L2(V)
				# Output the L2 shape gradient 
				self.f2 << V
			
			# Compute the (negative) shape gradient w.r.t. the elasticity inner product in \Omega
			dsp.compute_shape_gradient(V)

			if self.options["export"] > 0:
				# Output the elasticity shape gradient 
				self.f3 << V

			# Compute the directional shape derivative and the norm of the shape gradient
			d_obj = assemble(directional_shape_derivative) # depends on V
			grad_norm = sqrt(abs(d_obj))

			# Check for convergence
			if grad_norm < self.options["StoppingTolerance"]:

				# Prepare the output string
				output = ('%4d' % j, '%13.4e' % obj, '%13.4e' % d_obj, '%13.4e' % 0, '%13.4e' % grad_norm)
				# Output iteration data to history file and stdout
				if self.options["export"] > 0:
					csv.writer(self.f6).writerow(output)
				if self.options["verbosity"] > 0:
					csv.writer(sys.stdout).writerow(output)
				break

			# Set the initial step size for the subsequent line search
			if j==0:
				# Set initial step size (at first iteration)
				alpha = self.options["alpha0"]
			else:
				# Increase step size alpha a little bit if the previous step has been accepted
				alpha /= beta

			# Perform a line search
			while True:
				# Prepare the output string 
				output = ('%4d' % j, '%13.4e' % obj, '%13.4e' % d_obj, '%13.4e' % alpha, '%13.4e' % grad_norm)

				# Check for validity of step size
				if dsp.is_step_valid(alpha, V):
					# Move the mesh
					V_step.vector()[:] = alpha * V.vector()[:]
					dsp.move_mesh(V_step)

					# Compute the objective
					obj_step = dsp.compute_objective()

					# Check the Armijo condition
					if obj_step < obj + sigma * alpha * d_obj:
						# Successful: break while loop and go to next iteration
						break
					else:
						# Report Armijo condition failed and restore the mesh
						# Output iteration data to history file and stdout
						if self.options["export"] > 0:
							csv.writer(self.f6).writerow(output + ('%s' % "Armijo condition failed",))
						if self.options["verbosity"] > 0:
							csv.writer(sys.stdout).writerow(output + ('%s' % "Armijo condition failed",))

						# Restore the mesh 
						dsp.restore_mesh()
				else:
					# Report geometric condition violated
					# Output iteration data to history file and stdout
					if self.options["export"] > 0:
						csv.writer(self.f6).writerow(output + ('%s' % "geometry condition failed",))
					if self.options["verbosity"] > 0:
						csv.writer(sys.stdout).writerow(output + ('%s' % "geometry condition failed",))

				# Reduce step size if Armijo or geometric condition failed
				alpha *= beta

				# If alpha is too small, something went wrong
				if alpha < 1e-10:
					raise Exception("Line search failed")

			# Output iteration data to history file and stdout
			if self.options["export"] > 0:
				csv.writer(self.f6).writerow(output)
			if self.options["verbosity"] > 0:
				csv.writer(sys.stdout).writerow(output)
	
	# End of run()

# End of ShapeGradientMethod class

# Do a standard run of the gradient method
if __name__ == "__main__":
	print("Run standard configuration")
	# Create a 2D mesh
	meshlevel = 12
	degree = 1
	dim = 2
	mesh = UnitDiscMesh.create(MPI.comm_world, meshlevel, degree, dim)
	# Set the rhs for the PDE
	rhs = "Constant(2.5)*(x + Constant(0.4) - y**2)**2 + x**2 + y**2 + - Constant(1.0)"
	problem = ShapeGradientMethod(mesh = Mesh(mesh), rhs = rhs)
	problem.run()

# vim: fdm=marker noet
