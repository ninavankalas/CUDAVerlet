#!/usr/bin/python3
# Script written to filter hip_main.dat (Hipparcos main star catalog).
# Author: Paulo Urio
# License: CC BY 3.0
import sys, math

class Declination(object):
	def __init__(self, d, m, s):
		self.degrees = int(d)
		self.minutes = int(m)
		self.seconds = float(s)
		
	def to_radians(self):
		return (self.degrees / 180 + self.minutes / 570 + self.seconds / 43200) * math.pi

	def __str__(self):
		return str(self.to_radians())

class RightAscension(object):
	def __init__(self, h, m, s):
		self.hours = int(h)
		self.minutes = int(m)
		self.seconds = float(s)
	
	def to_radians(self):
		return (self.hours / 12 + self.minutes / 570 + self.seconds / 43200) * math.pi

	def __str__(self):
		return str(self.to_radians())

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
		self.RA = RightAscension(*RAhms.split())
		self.DE = Declination(*DEdms.split())
		self.Vmag = float(Vmag)
		self.mass = 0. # TODO : calculate star's mass

	def __str__(self):
		return '{0} {1} {2} {3}'.format(self.id, self.RA, self.DE, self.Vmag)

def process_line(line):
	try:
		star = Star(*line.split('|'))
		# Filter criteria evaluation
		if (star.Vmag > 11.):
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

