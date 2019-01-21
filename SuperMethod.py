from dolfin import *
import json
import sys

# The SuperMethod class provides some common functionality related to options and timing 
class SuperMethod(object):
	def __init__(self, mesh, rhs, JSONinput = None):
		self.options = {} # all method options will be stored here

	def get_option_list(self):
		# List of options, which can be set by json input
		option_list = {
			"directory": "directory",
			"export": "export",
			"verbosity": "verbosity",
			"maxiter": "maxiter",
			"sigma": "sigma",
			"beta": "beta",
			"alpha0": "alpha0",
			"StoppingTolerance": "StoppingTolerance",
			"E": "E",
			"nu": "nu",
			"damping": "damping",
		}
		return option_list

	def __del__(self):
		# Print timing results upon destruction
		if self.options["verbosity"] > 1:
			t = timings(TimingClear.keep, [TimingType.wall])
			print("\n" + t.str(True))
		# Close output files
		if self.options["export"] > 1:
			self.close_files()

	def open_files(self):
		pass

	def close_files(self):
		pass

	def load_config(self,config_file):
		# Read input from file
		
		# List of options, which can be set by json input
		option_list = self.get_option_list()

		options = {}
		options_updated = []

		with open(config_file,'r') as infile:
			options = json.load(infile)

		for key, param in option_list.items():
			if param in options:
				self.options[key] = options[param]
				options_updated.append(key)

		if self.options["verbosity"] > 0:
			print("Options from %s:" % config_file)
			print(*options_updated, sep=", ")
			print()

		return options_updated

	def run(self):
		raise Exception("run() is not implemented for %s" %self.method_name)
# End of SuperMethod() class

