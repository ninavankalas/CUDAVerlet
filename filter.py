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
from math import pi, cos, sin, tan

cot = lambda x: 1 / tan(x)

class Declination(object):
	def __init__(self, d, m, s):
		self.degrees = int(d)
		self.arcminutes = int(m)
		self.arcseconds = float(s)
		
	def __float__(self):
		return (self.degrees / 180 + self.arcminutes / 10800 + self.arcseconds / 648000) * pi

	def __str__(self):
		return str(float(self))

class RightAscension(object):
	def __init__(self, h, m, s):
		self.hours = int(h)
		self.minutes = int(m)
		self.seconds = float(s)
	
	def __float__(self):
		return (self.hours / 12 + self.minutes / 570 + self.seconds / 43200) * pi

	def __str__(self):
		return str(float(self))


class CelestialCoordinate(object):
	# x = cot(p) * sin(pi/4 - dec) * cos(ra)
	# y = cot(p) * sin(pi/4 - dec) * sin(ra)
	# z = cot(p) * cos(pi/4 - dec)
	def __init__(self, RAhms, DEdms, Parallax):
		self.RA = RightAscension(*RAhms.split())
		self.DE = Declination(*DEdms.split())
		self.Plx = float(Parallax)
	
	def x(self):
		return cot(self.Plx) * sin(pi / 4 - float(self.DE)) * cos(self.RA)

	def y(self):
		return cot(self.Plx) * sin(pi / 4 - float(self.DE)) * sin(self.RA)

	def z(self):
		return cot(self.Plx) * sin(pi / 4 - float(self.DE))
	
	def __str__(self):
		return '{x} {y} {z}'.format(x=self.x(), y=self.y(), z=self.z())
		
class ProperMotion(object):
	def __init__(self, pmRA, pmDE):
		self.pmRA = float(pmRA)
		self.pmDE = float(pmDE)
	
	def __str__(self):
		return '{ra} {de}'.format(ra=self.pmRA, de=self.pmDE)

class Mass(object):
	def __init__(self, Vmag):
		self.Vmag = float(Vmag)
	
	def __float__(self):
		sun_mass = 1.98855 * 10 ** 30
		sun_Vmag = 4.83
		return sun_mass * (self.Vmag / sun_Vmag) ** (1 / 3.5)
	
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
		self.mass = Mass(Vmag)

	def __str__(self):
		return '{id} {position} {motion} {mass}'.format(id=self.id,\
			position=self.position, motion=self.proper_motion, \
			mass=self.mass)

def process_line(line):
	try:
		star = Star(*line.split('|'))
		# Filter criteria evaluation
		if (star.id % 10000 == 0):
			print(star)
	except ValueError:
		# A field is empty, so we ignore the record.
		pass
	except TypeError:
		print('Error: invalid input format.')
		exit(1)

input_data = sys.stdin.readlines()
for line in input_data:
	process_line(line)

