# http://http.developer.nvidia.com/GPUGems3/gpugems3_ch31.html
from numpy import array, linalg, log10
import math
import time

YEARS = 1000
G = 6.6738480e-11

class Star(object):
	def __init__(self, hip, x, y, z, dx, dy, dz, m):
		self.hip = hip
		self.pos = array([x, y, z])
		self.vel = array([dx, dy, dz])
		self.m = m
	
	def _force(self, atom):
		r = atom.pos - self.pos
		rlen = linalg.norm(r)
		return r * (atom.m / (rlen ** 2 + math.e ** 2) ** (3/2))

	def acceleration(self, atoms):
		s = array([0., 0., 0.])
		for a in atoms:
			s += self._force(a)
		return G * s
	
	def __str__(self):
		return '%d\t%.10f\t%.10f\t%.10f' % (self.hip, self.pos[0], self.pos[1], self.pos[2])
	
input_stars = ['7 52.95793746940386 0.020835046586785186 19.313445457524082 1.178977272726114e-07 -1.189581269167245e-13 -1.1476841391322656e-13 0.7580624999008484',
'8 174.01562525792605 0.08288877189333556 84.44668989723645 -4.40340909090907e-07 -4.0753961412492303e-14 1.2083154614704392e-14 1.7910696851949173',
'11 159.152380690833 0.1035863098196834 170.31214772205647 -3.664772727272722e-07 -1.9703957395807868e-14 3.5889985518063045e-15 3.124969403386247',
'14 195.69077494081472 0.1649377317368024 1.2310011794642435 3.082386363636221e-07 9.22781551626482e-14 -1.7439450538431117e-14 2.8955009238198066']

def run():
	stars = list()
	for s in input_stars:
		stars.append(Star(*map(float, s.split())))

	it = 0
	while True:
		i = 0
		if it % 10 == 0:
			print('Ano %.0e' % (it * YEARS))
			print('HIP\tX\t\tY\t\tZ')
		while i < len(stars):
			a = stars[i].acceleration(stars)
			dt = YEARS
			stars[i].pos += stars[i].vel * dt + a * dt ** 2 / 2
			if it % 10 == 0:
				print(stars[i])
			i += 1
		if it % 10 == 0:
			print()
			exit(0)
		it += 1
		
if __name__ == '__main__':
	try:
		import cProfile
		cProfile.run('run()')
	except KeyboardInterrupt:
		pass
