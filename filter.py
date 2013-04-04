#!/usr/bin/python3
# Script written to filter hip_main.dat (Hipparcos main star catalog).
# Author: Paulo Urio
# License: CC BY 3.0
# The purpose of this script is to filter Hipparcos star catalog to
# select and prepare the input data to our Velocity Verlet algorithm
# implementation in CUDA.
# Knowledge sources:
# http://heasarc.gsfc.nasa.gov/W3Browse/all/hipparcos.html
# http://en.wikipedia.org/wiki/Mass%E2%80%93luminosity_relation
import sys
from math import pi, cos, sin, tan, sqrt, log10
from scipy import constants

class Sun:
	MASS = 1.98855e30 # kg
	LUMINOSITY = 1.9e-16 # W
	DISTANCE = 4.848e-6 # parsec
	VMAG = -26.74 # mag
	RADIUS = 6.955 * 10 ** 8 # meters
	TEMPERATURE = 5778 # Kelvin
	(X, Y, Z) = (0, 0, 0) # parsec

cot = lambda x: 1  / tan(x)

class Radius(object):
	# http://www.uni.edu/morgans/astro/course/Notes/section2/spectraltemps.html
	# Absolute Magnitude (mag): Temperature (K)
	TABLE = { -4.5: 54000, -4.0: 45000, -3.9: 43300, -3.8: 40600,
			-3.6: 37800, -3.3: 29200, -2.3: 23000, -1.9: 21000,
			-1.1: 17600, -.4: 15200, .0: 14300, .3: 13500, .7: 12300,
			+1.1: 11400, 1.5: 9600, 1.7: 9330, 1.8: 9040, 2.: 8750,
			2.1: 8480, 2.2: 8310, 2.4: 7920, 3.: 7350, 3.3: 7050,
			3.5: 6850, 3.7: 6700, 4.: 6550, 4.3: 6400, 4.4: 6300,
			4.7: 6050, 4.9: 5930, 5.: 5800, 5.2: 5660, 5.6: 5440,
			6.: 5240, 6.2: 5110, 6.4: 4960, 6.7: 4800, 7.1: 4600,
			7.4: 4400, 8.1: 4000, 8.7: 3750, 9.4: 3700, 10.1: 3600,
			10.7: 3500, 11.2: 3400, 12.3: 3200, 13.4: 3100, 13.9: 2900,
			14.4: 2700}
	
	def __init__(self, star):
		self.star = star
		self.get_temperature()
			
	def _apparent_magnitude(self, absolute_magnitude, distance):
		return absolute_magnitude - 5 * (1 - log10(distance))
	
	def get_temperature(self):
		best_match = None
		best_approx = None
		for mag in self.TABLE:
			vmag = self._apparent_magnitude(mag, self.star.coord.distance())
			approx = abs(vmag - self.star.vmag)
			if not best_match or approx < best_approx:
				best_match = mag
				best_approx = approx
		return self.TABLE[best_match]
	
	def schwarzschild_radius(self):
		# Proportionality constant = 2 * constants.G / constants.c ** 2
		return 1.4851296900185762e-27 * self.star.vmag.mass()

	def __truediv__(self, b):
		return float(self) / b

	def __float__(self):
		# http://en.wikipedia.org/wiki/Stefan%E2%80%93Boltzmann_law
		temperature = (Sun.TEMPERATURE / self.get_temperature()) ** 2
		luminosity = sqrt(self.star.vmag.luminosity() / Sun.LUMINOSITY)
		return Sun.RADIUS * temperature * luminosity

class Declination(object):
	def __init__(self, d, m, s):
		self.degrees = int(d)
		self.arcminutes = int(m)
		self.arcseconds = float(s)
		
	def __float__(self):
		return (self.degrees / 180 + self.arcminutes / 10800 + self.arcseconds / 648000) * pi

	def __rsub__(self, a):
		return a - float(self)

	def __str__(self):
		return str(float(self))

class RightAscension(object):
	def __init__(self, h, m, s):
		self.hours = int(h)
		self.minutes = int(m)
		self.seconds = float(s)
	
	def __float__(self):
		return (self.hours / 12 + self.minutes / 570 + self.seconds / 43200) * pi

	def __rsub__(self, a):
		return a - float(self)

	def __str__(self):
		return str(float(self))


class CelestialCoordinate(object):
	# x = cot(p) * cos(dec) * cos(ra)
	# y = cot(p) * cos(dec) * sin(ra)
	# z = cot(p) * sin(dec)
	def __init__(self, RAhms, DEdms, Parallax):
		self.RA = RightAscension(*RAhms.split())
		self.DE = Declination(*DEdms.split())
		self.Plx = float(Parallax) / 1000 # mas -> arcsec
	
	def x(self):
		return self.distance() * cos(self.DE) * cos(self.RA)

	def y(self):
		return self.distance() * cos(self.DE) * sin(self.RA)

	def z(self):
		return self.distance() * sin(self.DE)
	
	def distance(self):
		return abs(1 / self.Plx)
		
	def __str__(self):
		return '{x} {y} {z}'.format(x=self.x(), y=self.y(), z=self.z())
		
class ProperMotion(object):
	def __init__(self, pmRA, pmDE):
		self.pmRA = float(pmRA) / 1000.0
		self.pmDE = float(pmDE) / 1000.0
	
	def __str__(self):
		return '{RA} {DE}'.format(RA=self.pmRA, DE=self.pmDE)

class VMagnitude(object):
	def __init__(self, Vmag, star):
		self.Vmag = float(Vmag)
		self.star = star
		
	def luminosity(self):
		# http://en.wikipedia.org/wiki/Luminosity#Apparent
		mag = 10 ** ((self.Vmag - Sun.VMAG) / -2.5)
		d = (Sun.DISTANCE / self.star.coord.distance()) ** 2
		return (Sun.LUMINOSITY * mag) / d
		
	def radius(self):
		# http://physics.ucsd.edu/students/courses/winter2008/managed/physics223/documents/Lecture7%13Part3.pdf
		return float(self) ** (.9)
	
	def mass(self):
		a = 3.5
		mag = self.luminosity() / Sun.LUMINOSITY
		return Sun.MASS * mag ** (1 / a)
	
	def __rsub__(self, a):
		return a - float(self)

	def __float__(self):
		return self.Vmag
	
	def __str__(self):
		raise str(self.Vmag)

class Star(object):
	def __init__(self, Catalog, HIP, Proxy, RAhms, DEdms, Vmag, VarFlag, \
		r_Vmag, RAdeg, DEdeg, AstroRef, Plx, pmRA, pmDE, e_RAdeg, \
		e_DEdeg, e_Plx, e_pmRA, e_pmDE, DEDA, PlxRA, PlxDE, pmRARA, \
		pmRADE, pmRAPlx, pmDERA, pmDEDE, pmDEPlx, pmDEpmRA, F1, F2, \
		HIP2, BTmag, e_BTmag, VTmag, e_VTmag, m_BTmag, BV, e_BV, \
		r_BV, VI, e_VI, r_VI, CombMag, Hpmag, e_Hpmag, Hpscat, \
		o_Hpmag, m_Hpmag, Hpmax, HPmin, Period, HvarType, moreVar, \
		morePhoto, CCDM, n_CCDM, Nsys, Ncomp, MultFlag, Source, Qual, \
		m_HIP, theta, rho, e_rho, dHp, e_dHpm, Survey, Chart, Notes, \
		HD, BD, CoD, CPD, VIred, SpType, r_SpType):
		self.id = int(HIP)
		self.coord = CelestialCoordinate(RAhms, DEdms, Plx)
		self.proper_motion = ProperMotion(pmRA, pmDE)
		self.vmag = VMagnitude(Vmag, self)
		self.radius = Radius(self)
		self.info = (('\nID = %(HIP)s\nRaw input\n'
				    '\tRAhms = %(RAhms)s\n'
					'\tDEdms = %(DEdms)s\n'
					'\tVmag = %(Vmag)s mag\n'
					'\tPlx = %(Plx)s mas\n'
					'\tpmRA = %(pmRA)s mas/yr\n'
					'\tpmDE = %(pmDE)s mas/yr\n'
					'Computed data\n' 
					'\tRA = {RA} radians\n' 
					'\tDE = {DE} radians\n'
					'\tParallax = {Plx} arcsec\n'
					'\tX = {X} parsecs\n'
					'\tY = {Y} parsecs\n'
					'\tZ = {Z} parsecs\n'
					'\tProper motion RA = {pmRA} arcsec/yr\n' 
					'\tProper motion DE = {pmDE} arcsec/yr\n'
					'\tDistance = {distance} parsecs\n'
					'\tLuminosity = {luminosity} L☉\n'
					'\tMass = {mass} M☉\n'
					'\tTemperature = {temperature} K☉\n'
					'\tRadius = {radius} R☉\n'
					'\tSchwarzschild radius = {gravradius} m² / kg') 
						% locals()).format(
						RA=self.coord.RA,
						DE=self.coord.DE,
						Plx=self.coord.Plx,
						X=self.coord.x(),
						Y=self.coord.y(),
						Z=self.coord.z(),
						pmRA=self.proper_motion.pmRA,
						pmDE=self.proper_motion.pmDE,
						distance=self.coord.distance(),
						luminosity=self.vmag.luminosity() / Sun.LUMINOSITY,
						mass=float(self.vmag.mass()) / Sun.MASS,
						temperature=self.radius.get_temperature() / Sun.TEMPERATURE,
						radius=self.radius / Sun.RADIUS,
						gravradius=self.radius.schwarzschild_radius())

	def __str__(self):
		return '{id} {position} {motion} {mass}'.format(id=self.id, \
			position=self.coord, 
			motion=self.proper_motion, \
			mass=self.vmag.mass() / Sun.MASS)

def process_line(line):
	try:
		star = Star(*line.split('|'))
		# Filter criteria evaluation
		if star.coord.distance() < 5.:
			print(star.info)
			#print('Output line')
			print(star)
	except ZeroDivisionError:
		# Probably a star with parallax = 0.0
		pass
	except ValueError:
		# A field is empty, so we ignore the record.
		pass
	except TypeError:
		print('Error: invalid input format.')
		exit(1)
	except KeyboardInterrupt:
		exit(0)

if __name__ == '__main__':
	input_data = sys.stdin.readlines()
	for line in input_data:
		process_line(line)

