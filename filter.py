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
import struct
from math import pi, cos, sin, tan, sqrt, log10
from scipy import constants

HUBBLE_CONSTANT = 70.4e6 # (km/s) / parsec

class Sun:
	MASS = 1.98855e30
	LUMINOSITY = 1.9e-16
	DISTANCE = 4.848e-6
	VMAG = -26.74
	RADIUS = 6.955e8
	SCHWARSZCHILD_RADIUS = 2.9532546450864397e3
	(X, Y, Z) = (0, 0, 0)
	
	def __str__(self):
		return ('\tMass: {} kg\n'
				'\tLuminosity: {} W\n'
				'\tDistance: {} parsec\n'
				'\tVisual Magnitude: {} mag\n'
				'\tRadius: {} meter\n'
				'\tSchwarzschild radius: {} meter\n'
				'\t(X, Y, Z): ({}, {}, {}) parsec').format(self.MASS, 
						self.LUMINOSITY, self.DISTANCE, self.VMAG, self.RADIUS, 
						self.SCHWARSZCHILD_RADIUS,	self.X, self.Y, self.Z)

cot = lambda x: 1  / tan(x)

class Radius(object):
	def __init__(self, star):
		self.star = star
	
	def schwarzschild_radius(self):
		# Proportionality constant = 2 * constants.G / constants.c ** 2
		return 1.4851296900185762e-27 * self.star.vmag.mass()

	def __truediv__(self, b):
		return float(self) / b

	def __float__(self):
		return (self.star.vmag.mass() / Sun.MASS) ** 0.9

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

class InvalidParallaxError(Exception):
	# useful exception when parallax is equal to zero.  Trying to use it
	# would generate division by zero.
	pass

class CelestialCoordinate(object):
	# x = cot(p) * cos(dec) * cos(ra)
	# y = cot(p) * cos(dec) * sin(ra)
	# z = cot(p) * sin(dec)
	def __init__(self, RAhms, DEdms, Parallax):
		self.RA = RightAscension(*RAhms.split())
		self.DE = Declination(*DEdms.split())
		self.Plx = float(Parallax) / 1000 # mas -> arcsec
		if self.Plx == .0:
			raise InvalidParallaxError()
	
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
	def __init__(self, pmRA, pmDE, star):
		self.pmRA = self.arcserc_to_radian(float(pmRA) / 1000.0)
		self.pmDE = self.arcserc_to_radian(float(pmDE) / 1000.0)
		self.star = star

	def arcserc_to_radian(self, arcsec):
		return arcsec * 4.848136811097625e-6
		
	def set_radialvelocity(self, rdV):
		self.rdV = rdV # km/s
		
	def radialvelocity_pcyr(self):
		# Radial Velocity in parsec/yr
		return self.rdV * 1.0227121651072225e-06
		
	def radialvelocity_pc(self):
		# Radial Velocity in parsec
		# https://www.cfa.harvard.edu/~dfabricant/huchra/seminar/lsc/
		return self.rdV / HUBBLE_CONSTANT
		
	def dx(self):
		return self.radialvelocity_pc() * cos(self.pmDE) * cos(self.pmRA)

	def dy(self):
		return self.radialvelocity_pc() * cos(self.pmDE) * sin(self.pmRA)

	def dz(self):
		return self.radialvelocity_pc() * sin(self.pmDE)
	
	def __str__(self):
		return '{dx} {dy} {dz}'.format(dx=self.dx(), dy=self.dy(), dz=self.dz())

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
		return float(self) ** .9
	
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
	def __init__(self, Catalog, HIP, Proxy, RAhms, DEdms, Vmag, VarFlag,
		r_Vmag, RAdeg, DEdeg, AstroRef, Plx, pmRA, pmDE, e_RAdeg,
		e_DEdeg, e_Plx, e_pmRA, e_pmDE, DEDA, PlxRA, PlxDE, pmRARA,
		pmRADE, pmRAPlx, pmDERA, pmDEDE, pmDEPlx, pmDEpmRA, F1, F2,
		HIP2, BTmag, e_BTmag, VTmag, e_VTmag, m_BTmag, BV, e_BV,
		r_BV, VI, e_VI, r_VI, CombMag, Hpmag, e_Hpmag, Hpscat,
		o_Hpmag, m_Hpmag, Hpmax, HPmin, Period, HvarType, moreVar,
		morePhoto, CCDM, n_CCDM, Nsys, Ncomp, MultFlag, Source, Qual,
		m_HIP, theta, rho, e_rho, dHp, e_dHpm, Survey, Chart, Notes,
		HD, BD, CoD, CPD, VIred, SpType, r_SpType):
		self.id = int(HIP)
		self.coord = CelestialCoordinate(RAhms, DEdms, Plx)
		self.proper_motion = ProperMotion(pmRA, pmDE, self)
		self.vmag = VMagnitude(Vmag, self)
		self.radius = Radius(self)
		self.info = ('\nID = %(HIP)s\nRaw input\n'
				    '\tRAhms = %(RAhms)s\n'
					'\tDEdms = %(DEdms)s\n'
					'\tVmag = %(Vmag)s mag\n'
					'\tPlx = %(Plx)s mas\n'
					'\tpmRA = %(pmRA)s mas/yr\n'
					'\tpmDE = %(pmDE)s mas/yr\n' % locals())
		
	def has_enoughdata(self):
		return hasattr(self.proper_motion, 'rdV')
	
	def get_computed_data(self):
		if hasattr(self.proper_motion, 'rdV'):
			rdV = self.proper_motion.rdV
			rdVpcyr = self.proper_motion.radialvelocity_pcyr()
		else:
			rdV, rdVpcyr = '?', '?'
		return self.info + ('Star\n' 
				'\tHIP = {HIP}\n' 
				'\tRA = {RA} radian\n' 
				'\tDE = {DE} radian\n'
				'\tParallax = {Plx} arcsec\n'
				'\tX = {X} parsec\n'
				'\tY = {Y} parsec\n'
				'\tZ = {Z} parsec\n'
				'\tDistance = {distance} parsec\n'
				'\tProper motion RA = {pmRA} radian/yr\n' 
				'\tProper motion DE = {pmDE} radian/yr\n'
				'\tRadial velocity = {radial} km/s ({rdVpcyr} parsec/yr)\n'
				'\tDelta X = {dx} parsec/yr\n'
				'\tDelta Y = {dy} parsec/yr\n'
				'\tDelta Z = {dz} parsec/yr\n'
				'\tLuminosity = {luminosity} L☉\n'
				'\tRadius = {radius} R☉\n'
				'\tSchwarszchild radius = {sradius} Rs☉\n'
				'\tMass = {mass} M☉\n').format(
				HIP=self.id,
				RA=self.coord.RA,
				DE=self.coord.DE,
				Plx=self.coord.Plx,
				X=self.coord.x(),
				Y=self.coord.y(),
				Z=self.coord.z(),
				dx=self.proper_motion.dx(),
				dy=self.proper_motion.dy(),
				dz=self.proper_motion.dz(),
				pmRA=self.proper_motion.pmRA,
				pmDE=self.proper_motion.pmDE,
				radial=rdV,
				rdVpcyr=rdVpcyr,
				distance=self.coord.distance(),
				radius=float(self.radius),
				sradius=self.radius.schwarzschild_radius() / Sun.SCHWARSZCHILD_RADIUS,
				luminosity=self.vmag.luminosity() / Sun.LUMINOSITY,
				mass=float(self.vmag.mass()) / Sun.MASS)

	def __str__(self):
		return '{id} {position} {motion} {mass}'.format(id=self.id,
			position=self.coord, motion=self.proper_motion,
			mass=self.vmag.mass() / Sun.MASS)

def process_line(line):
	try:
		star = Star(*line.split('|'))
		return star
	except InvalidParallaxError:
		pass
	except ValueError:
		# A field is empty, so we ignore the record.
		pass
	except TypeError:
		print('Error: invalid input format.')
		exit(1)

# Merge pulkovo data to the current star instance.
def merge_pulkovo(stars, HIP, HD, rdV, ie_rdV, srcs, ee_rdV, RA, DE, mV):
	# http://www.geocities.ws/orionspiral/
	hip = int(HIP)
	if hip in stars:
		stars[hip].proper_motion.set_radialvelocity(float(rdV))

# Read Pulkovo Radial Velocity data		
def read_pulkovo(stars):
	input_data = open('pcrv.txt', 'r').readlines()
	for line in input_data:
		bline = line.encode()
		merge_pulkovo(stars, *struct.unpack('6s7s7s4s2s6s6s7s5sx', bline))
	return stars

def read_hipparcos():
	stars = dict() 
	input_data = open('hip_main.dat', 'r').readlines()
	for line in input_data:
		star = process_line(line)
		if star:
			stars[star.id] = star
	return stars

if __name__ == '__main__':
	print('Sun information:\n', Sun())
	try:
		stars = read_hipparcos()
		read_pulkovo(stars)
		for s in stars.values():
			if s.has_enoughdata():
				print(s.get_computed_data())
				print(s)
	except KeyboardInterrupt:
		exit(0)
	except FileNotFoundError:
		print('One of the input files were not found.')
		exit(1)
	
