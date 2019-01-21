import scipy.sparse

# This module defines some helper functions and classes

# This class implements a wrapper for integration measures
# since FEniCS does not like constructions like f*(y*dx).
class WeightedMeasureWrapper:
	def __init__(self, factor, measure):
		self.factor = factor
		self.measure = measure

	def __rmul__(self, other):
		return (other * self.factor) * self.measure


# Convert PETScMatrix to scipy array
# https://fenicsproject.org/qa/9661/convert-to-a-scipy-friendly-format/
def to_scipy(Q):
	ai, aj, av = Q.mat().getValuesCSR()
	n = Q.size(0)
	m = Q.size(1)
	Q = scipy.sparse.csr_matrix((av, aj, ai), shape = (n,m))
	return Q

