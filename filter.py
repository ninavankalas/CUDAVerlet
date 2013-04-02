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
from math import pi, cos, sin, tan, sqrt

PARSEC = 3.0856775814671900 * 10 ** 16

class Sun:
	MASS = 1.98855e30
	LUMINOSITY = 1 # 1.9e-16
	DISTANCE = 4.848e-6
	VMAG = -26.74
	(X, Y, Z) = (0, 0, 0)

cot = lambda x: 1 / tan(x)

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
		self.Plx = float(Parallax)
	
	def x(self):
		return cot(self.Plx) * cos(self.DE) * cos(self.RA)

	def y(self):
		return cot(self.Plx) * cos(self.DE) * sin(self.RA)

	def z(self):
		return cot(self.Plx) * sin(self.DE)
	
	def distance(self):
		return sqrt(self.x() ** 2 + self.y() ** 2 + self.z() ** 2)
		
	def __str__(self):
		return '{x} {y} {z}'.format(x=self.x(), y=self.y(), z=self.z())
		
class ProperMotion(object):
	def __init__(self, pmRA, pmDE):
		self.pmRA = float(pmRA)
		self.pmDE = float(pmDE)
	
	def __str__(self):
		return '{RA} {DE}'.format(RA=self.pmRA, DE=self.pmDE)

class Mass(object):
	def __init__(self, Vmag, star):
		self.Vmag = float(Vmag)
		self.star = star
		
	def luminosity(self):
		# http://en.wikipedia.org/wiki/Luminosity#Apparent
		mag = 10 ** ((self.Vmag - Sun.VMAG) / -2.5)
		d = (Sun.DISTANCE / self.star.position.distance()) ** 2
		return (Sun.LUMINOSITY * mag) / d
		
	def radius(self):
		# http://physics.ucsd.edu/students/courses/winter2008/managed/physics223/documents/Lecture7%13Part3.pdf
		return float(self) ** (15 / 19)
	
	def __float__(self):
		a = 3.5
		mag = self.luminosity() / Sun.LUMINOSITY
		return Sun.MASS * mag ** (1 / a)
	
	def __str__(self):
		return str(float(self))

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
		self.position = CelestialCoordinate(RAhms, DEdms, Plx)
		self.proper_motion = ProperMotion(pmRA, pmDE)
		self.mass = Mass(Vmag, self)
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
					'\tX = {X} parsecs\n'
					'\tY = {Y} parsecs\n'
					'\tZ = {Z} parsecs\n'
					'\tDistance = {distance} parsecs\n'
					'\tLuminosity = {luminosity} W\n'
					'\tMass = {mass} kg\n'
					'\tRadius = {radius} meters') % locals()).format(
						RA=self.position.RA,
						DE=self.position.DE,
						X=self.position.x(),
						Y=self.position.y(),
						Z=self.position.z(),
						distance=self.position.distance(),
						luminosity=self.mass.luminosity(),
						mass=self.mass,
						radius=self.mass.radius())


	def __str__(self):
		return '{id} {position} {motion} {mass}'.format(id=self.id, \
			position=self.position, motion=self.proper_motion, \
			mass=self.mass)

def process_line(line):
	try:
		star = Star(*line.split('|'))
		# Filter criteria evaluation
		if (star.id < 5):
			print(star.info)
			print('Output line')
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

input_data = sys.stdin.readlines()
for line in input_data:
	process_line(line)

