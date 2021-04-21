#=============================================================================#
# neksuite                                                                    #
#                                                                             #
# A python module for reading and writing nek5000 files                       #
#                                                                             #
# Authors: Jacopo Canton, Nicolo' Fabbiane                                    #
# Contacts: jcanton(at)mech.kth.se, nicolo(at)mech.kth.se                     #
# Last edit: 2015-10-19                                                       #
#=============================================================================#
import struct
import numpy as np
import exadata as exdat


#==============================================================================
def readnek(fname):
	"""
	    readnek
	    A function for reading binary data from the nek5000 binary format

	    input variable:
	    fname : file name
	"""
	#
	try:
		infile = open(fname, 'rb')
	except IOError as e:
		print('I/O error ({0}): {1}'.format(e.errno, e.strerror))
		return -1
	#
	#---------------------------------------------------------------------------
	# READ HEADER
	#---------------------------------------------------------------------------
	#
	# read header
	header = infile.read(132).split()
	#
	# get word size
	wdsz = int(header[1])
	if (wdsz == 4):
		realtype = 'f'
	elif (wdsz == 8):
		realtype = 'd'
	else:
		print('ERROR: could not interpret real type (wdsz = %i)' %(wdsz))
		return -2
	#
	# get polynomial order
	lr1 = [int(header[2]), 
	       int(header[3]),
	       int(header[4])]
	#
	# compute total number of points per element
	npel = lr1[0] * lr1[1] * lr1[2]
	#
	# get number of pysical dimensions
	ndim = 2 + (lr1[2]>1)
	#
	# get number of elements
	nel = int(header[5])
	#
	# get number of elements in the file
	nelf = int(header[6])
	#
	# get current time
	time = float(header[7])
	#
	# get current time step
	istep = int(header[8])
	#
	# get file id
	fid = int(header[9])
	#
	# get tot number of files
	nf = int(header[10])
	#
	# get variables [XUPT]
	vars = header[11].decode('utf-8')
	var = [0 for i in range(5)]
	for v in vars:
		if (v == 'X'):
			var[0] = ndim
		elif (v == 'U'):
			var[1] = ndim
		elif (v == 'P'):
			var[2] = 1
		elif (v == 'T'):
			var[3] = 1
		elif (v == 'S'):
			var[4] = 0 # TODO: need to know how this works
	#
	# compute number of scalar fields
	nfields = sum(var)
	#
	# identify endian encoding
	etagb = infile.read(4)
	etagL = struct.unpack('<f', etagb)[0]; etagL = int(etagL*1e5)/1e5
	etagB = struct.unpack('>f', etagb)[0]; etagB = int(etagB*1e5)/1e5
	if (etagL == 6.54321):
		# print('Reading little-endian file\n')
		emode = '<'
	elif (etagB == 6.54321):
		# print('Reading big-endian file\n')
		emode = '>'
	else:
		print('ERROR: could not interpret endianness')
		return -3
	#
	# read element map for the file
	elmap = infile.read(4*nelf)
	elmap = list(struct.unpack(emode+nelf*'i', elmap))
	#
	#---------------------------------------------------------------------------
	# READ DATA
	#---------------------------------------------------------------------------
	#
	# initialize data structure
	data = exdat.exadata(ndim, nel, lr1, var)
	data.time   = time
	data.istep  = istep
	data.wdsz   = wdsz
	if (emode == '<'):
		data.endian = 'little'
	elif (emode == '>'):
		data.endian = 'big'
	#
	# read geometry
	data.lims.pos[:,0] =  float('inf')
	data.lims.pos[:,1] = -float('inf')
	for iel in elmap:
		for idim in range(var[0]): # if var[0] == 0, geometry is not read
			fi = infile.read(npel*wdsz)
			fi = list(struct.unpack(emode+npel*realtype, fi))
			ip = 0
			for iz in range(lr1[2]):
				for iy in range(lr1[1]):
					data.elem[iel-1].pos[idim,iz,iy,:] = fi[ip:ip+lr1[0]]
					ip += lr1[0]
			data.lims.pos[idim,0] = min([data.lims.pos[idim,0]]+fi)
			data.lims.pos[idim,1] = max([data.lims.pos[idim,1]]+fi)
	#
	# read velocity
	data.lims.vel[:,0] =  float('inf')
	data.lims.vel[:,1] = -float('inf')
	for iel in elmap:
		for idim in range(var[1]): # if var[1] == 0, velocity is not read
			fi = infile.read(npel*wdsz)
			fi = list(struct.unpack(emode+npel*realtype, fi))
			ip = 0
			for iz in range(lr1[2]):
				for iy in range(lr1[1]):
					data.elem[iel-1].vel[idim,iz,iy,:] = fi[ip:ip+lr1[0]]
					ip += lr1[0]
			data.lims.vel[idim,0] = min([data.lims.vel[idim,0]]+fi)
			data.lims.vel[idim,1] = max([data.lims.vel[idim,1]]+fi)
	#
	# read pressure 
	data.lims.pres[:,0] =  float('inf')
	data.lims.pres[:,1] = -float('inf')
	for iel in elmap:
		for ivar in range(var[2]): # if var[2] == 0, pressure is not read
			fi = infile.read(npel*wdsz)
			fi = list(struct.unpack(emode+npel*realtype, fi))
			ip = 0
			for iz in range(lr1[2]):
				for iy in range(lr1[1]):
					data.elem[iel-1].pres[ivar,iz,iy,:] = fi[ip:ip+lr1[0]]
					ip += lr1[0]
			data.lims.pres[ivar,0] = min([data.lims.pres[ivar,0]]+fi)
			data.lims.pres[ivar,1] = max([data.lims.pres[ivar,1]]+fi)
	#
	# read temperature
	data.lims.temp[:,0] =  float('inf')
	data.lims.temp[:,1] = -float('inf')
	for iel in elmap:
		for ivar in range(var[3]): # if var[3] == 0, temperature is not read
			fi = infile.read(npel*wdsz)
			fi = list(struct.unpack(emode+npel*realtype, fi))
			ip = 0
			for iz in range(lr1[2]):
				for iy in range(lr1[1]):
					data.elem[iel-1].temp[ivar,iz,iy,:] = fi[ip:ip+lr1[0]]
					ip += lr1[0]
			data.lims.temp[ivar,0] = min([data.lims.temp[ivar,0]]+fi)
			data.lims.temp[ivar,1] = max([data.lims.temp[ivar,1]]+fi)
	#
	# read scalar fields
	data.lims.scal[:,0] =  float('inf')
	data.lims.scal[:,1] = -float('inf')
	for iel in elmap:
		for ivar in range(var[4]): # if var[4] == 0, scalars are not read
			fi = infile.read(npel*wdsz)
			fi = list(struct.unpack(emode+npel*realtype, fi))
			ip = 0
			for iz in range(lr1[2]):
				for iy in range(lr1[1]):
					data.elem[iel-1].scal[ivar,iz,iy,:] = fi[ip:ip+lr1[0]]
					ip += lr1[0]
			data.lims.scal[ivar,0] = min([data.lims.scal[ivar,0]]+fi)
			data.lims.scal[ivar,1] = max([data.lims.scal[ivar,1]]+fi)
	#
	#
	# close file
	infile.close()
	#
	# output
	return data


#==============================================================================
def writenek(fname, data):
	"""
	    writenek
	    A function for writing binary data in the nek5000 binary format

	    input variable:
	    fname : file name
	    data : exadata data organised as readnek() output
	"""
	#
	try:
		outfile = open(fname, 'wb')
	except IOError as e:
		print('I/O error ({0}): {1}'.format(e.errno, e.strerror))
		return -1
	#
	#---------------------------------------------------------------------------
	# WRITE HEADER
	#---------------------------------------------------------------------------
	#
	# multiple files (not implemented)
	fid = 0
	nf = 1
	nelf = data.nel
	#
	# get fields to be written
	vars = ''
	if (data.var[0] > 0): vars += 'X'
	if (data.var[1] > 0): vars += 'U'
	if (data.var[2] > 0): vars += 'P'
	if (data.var[3] > 0): vars += 'T'
	if (data.var[4] > 0): vars += 'S' # TODO: need to know how this works
	#
	# get word size
	if (data.wdsz == 4):
		realtype = 'f'
	elif (data.wdsz == 8):
		realtype = 'd'
	else:
		print('ERROR: could not interpret real type (wdsz = %i)' %(wdsz))
		return -2
	#
	# get endian
	if (data.endian == 'little'):
		# print('Writing little-endian file\n')
		emode = '<'
	elif (data.endian == 'big'):
		# print('Writing big-endian file\n')
		emode = '>'
	else:
		print('ERROR: could not interpret endianness')
		return -3
	#
	# generate header
	header = '#std %1i %2i %2i %2i %10i %10i %20.13E %9i %6i %6i %s\n' %(
		data.wdsz, data.lr1[0], data.lr1[1], data.lr1[2], data.nel, nelf,
		data.time, data.istep, fid, nf, vars)
	#
	# write header
	header = header.ljust(132)
	outfile.write(header.encode('utf-8'))
	#
	# write tag (to specify endianness)
	etagb = struct.pack(emode+'f', 6.54321)
	outfile.write(etagb)
	#
	# generate and write element map for the file
	elmap = range(data.nel+1)[1:]
	outfile.write(struct.pack(emode+nelf*'i', *elmap))
	#
	#---------------------------------------------------------------------------
	# WRITE DATA
	#---------------------------------------------------------------------------
	#
	# compute total number of points per element
	npel = data.lr1[0] * data.lr1[1] * data.lr1[2]
	#
	# write geometry
	for iel in elmap:
		fo = np.zeros(npel)
		for idim in range(data.var[0]): # if var[0] == 0, geometry is not written
			ip = 0
			for iz in range(data.lr1[2]):
				for iy in range(data.lr1[1]):
					fo[ip:ip+data.lr1[0]] = data.elem[iel-1].pos[idim,iz,iy,:] 
					ip += data.lr1[0]
			outfile.write(struct.pack(emode+npel*realtype, *fo))
	#
	# write velocity
	for iel in elmap:
		fo = np.zeros(npel)
		for idim in range(data.var[1]): # if var[1] == 0, velocity is not written
			ip = 0
			for iz in range(data.lr1[2]):
				for iy in range(data.lr1[1]):
					fo[ip:ip+data.lr1[0]] = data.elem[iel-1].vel[idim,iz,iy,:] 
					ip += data.lr1[0]
			outfile.write(struct.pack(emode+npel*realtype, *fo))
	#
	# write pressure
	for iel in elmap:
		fo = np.zeros(npel)
		for ivar in range(data.var[2]): # if var[2] == 0, pressure is not written
			ip = 0
			for iz in range(data.lr1[2]):
				for iy in range(data.lr1[1]):
					fo[ip:ip+data.lr1[0]] = data.elem[iel-1].pres[ivar,iz,iy,:] 
					ip += data.lr1[0]
			outfile.write(struct.pack(emode+npel*realtype, *fo))
	#
	# write temperature
	for iel in elmap:
		fo = np.zeros(npel)
		for ivar in range(data.var[3]): # if var[3] == 0, temperature is not written
			ip = 0
			for iz in range(data.lr1[2]):
				for iy in range(data.lr1[1]):
					fo[ip:ip+data.lr1[0]] = data.elem[iel-1].temp[ivar,iz,iy,:] 
					ip += data.lr1[0]
			outfile.write(struct.pack(emode+npel*realtype, *fo))
	#
	# write scalars
	for iel in elmap:
		fo = np.zeros(npel)
		for ivar in range(data.var[4]): # if var[4] == 0, scalars are not written
			ip = 0
			for iz in range(data.lr1[2]):
				for iy in range(data.lr1[1]):
					fo[ip:ip+data.lr1[0]] = data.elem[iel-1].scal[ivar,iz,iy,:] 
					ip += data.lr1[0]
			outfile.write(struct.pack(emode+npel*realtype, *fo))
	#
	#
	# write max and min of every field in every element (forced to single precision)
	if (data.ndim==3):
		#
		for iel in elmap:
			for idim in range(data.var[0]):
				outfile.write(struct.pack(emode+'f', np.min(data.elem[iel-1].pos[idim, :,:,:])))
				outfile.write(struct.pack(emode+'f', np.max(data.elem[iel-1].pos[idim, :,:,:])))
			for idim in range(data.var[1]):
				outfile.write(struct.pack(emode+'f', np.min(data.elem[iel-1].vel[idim, :,:,:])))
				outfile.write(struct.pack(emode+'f', np.max(data.elem[iel-1].vel[idim, :,:,:])))
			for idim in range(data.var[2]):
				outfile.write(struct.pack(emode+'f', np.min(data.elem[iel-1].pres[idim, :,:,:])))
				outfile.write(struct.pack(emode+'f', np.max(data.elem[iel-1].pres[idim, :,:,:])))
			for idim in range(data.var[3]):
				outfile.write(struct.pack(emode+'f', np.min(data.elem[iel-1].temp[idim, :,:,:])))
				outfile.write(struct.pack(emode+'f', np.max(data.elem[iel-1].temp[idim, :,:,:])))

	# close file
	outfile.close()
	#
	# output
	return 0


#==============================================================================
def readrea(fname):
	"""
	    readrea
	    A function for reading .rea files for nek5000

	    input variable:
	    fname : file name
	"""
	#
	try:
		infile = open(fname, 'r')
	except IOError as e:
		print('I/O error ({0}): {1}'.format(e.errno, e.strerror))
		#return -1
	#
	#---------------------------------------------------------------------------
	# READ HEADER (2 lines) + ndim + number of parameters
	#---------------------------------------------------------------------------
	#
	infile.readline()
	infile.readline()
	ndim = int(infile.readline().split()[0])
	npar = int(infile.readline().split()[0])
	#
	nface = 2*ndim
	#
	#---------------------------------------------------------------------------
	# READ parameters
	#---------------------------------------------------------------------------
	#
	param = np.zeros((npar,1))
	for ipar in range(npar):
		param[ipar] = float(infile.readline().split()[0])
	#
	#---------------------------------------------------------------------------
	# skip passive scalars
	#---------------------------------------------------------------------------
	#
	npscal = int(infile.readline().split()[0])
	for ipscal in range(npscal):
		infile.readline()
	#
	#---------------------------------------------------------------------------
	# skip logical switches
	#---------------------------------------------------------------------------
	#
	nswitch = int(infile.readline().split()[0])
	for iswitch in range(nswitch):
		infile.readline()
	#
	#---------------------------------------------------------------------------
	# skip XFAC,YFAC,XZERO,YZERO
	#---------------------------------------------------------------------------
	#
	infile.readline()
	#
	#---------------------------------------------------------------------------
	# READ MESH
	#---------------------------------------------------------------------------
	#
	infile.readline()
	nel = int(infile.readline().split()[0])
	#
	# initialize data structure
	lr1 = [2, 2, ndim-1]
	var = [ndim, 0, 0, 0, 0]
	#
	data = exdat.exadata(ndim, nel, lr1, var)
	#
	# read geometry
	data.lims.pos[:,0] =  float('inf')
	data.lims.pos[:,1] = -float('inf')
	for iel in range(nel):
		# skip element number and group
		infile.readline()
		for idim in range(var[0]-1): # if ndim == 3 do this twice
			for jdim in range(var[0]):
				fi = infile.readline().split()
				data.elem[iel].pos[jdim,idim,0,0] = float(fi[0])
				data.elem[iel].pos[jdim,idim,0,1] = float(fi[1])
				data.elem[iel].pos[jdim,idim,1,1] = float(fi[2])
				data.elem[iel].pos[jdim,idim,1,0] = float(fi[3])
	#
	#---------------------------------------------------------------------------
	# CURVED SIDE DATA
	#---------------------------------------------------------------------------
	#
	infile.readline()
	ncurved = int(infile.readline().split()[0])
	data.ncurv = ncurved
	for icurved in range(ncurved):
		line = infile.readline()
		if (nel < 1e3):
			iedge = int(line[0:3])-1
			iel   = int(line[3:6])-1
			data.elem[iel].curv[iedge] = float(line[6:16])
		elif (nel < 1e6):
			iedge = int(line[0:2])-1
			iel   = int(line[2:8])-1
			data.elem[iel].curv[iedge] = float(line[8:18])
		else:
			iedge = int(line[0:2])-1
			iel   = int(line[2:12])-1
			data.elem[iel].curv[iedge] = float(line[12:22])
	#
	#---------------------------------------------------------------------------
	# BOUNDARY CONDITIONS
	#---------------------------------------------------------------------------
	#
	infile.readline()
	infile.readline()
	for iel in range(nel):
		for iface in range(nface):
			line = infile.readline()
			if (nel < 1e3):
				data.elem[iel].bcs[iface][0] = line[1:3].strip()
				data.elem[iel].bcs[iface][1] = int(line[4:7])
				data.elem[iel].bcs[iface][2] = int(line[7:10])
				data.elem[iel].bcs[iface][3] = float(line[10:24])
				data.elem[iel].bcs[iface][4] = float(line[24:38])
				data.elem[iel].bcs[iface][5] = float(line[38:52])
				data.elem[iel].bcs[iface][6] = float(line[52:66])
				data.elem[iel].bcs[iface][7] = float(line[66:80])
			elif (nel < 1e5):
				data.elem[iel].bcs[iface][0] = line[1:3].strip()
				data.elem[iel].bcs[iface][1] = int(line[4:10])
				data.elem[iel].bcs[iface][2] = iface + 1
				data.elem[iel].bcs[iface][3] = float(line[10:24])
				data.elem[iel].bcs[iface][4] = float(line[24:38])
				data.elem[iel].bcs[iface][5] = float(line[38:52])
				data.elem[iel].bcs[iface][6] = float(line[52:66])
				data.elem[iel].bcs[iface][7] = float(line[66:80])
			elif (nel < 1e6):
				data.elem[iel].bcs[iface][0] = line[1:3].strip()
				data.elem[iel].bcs[iface][1] = int(line[4:10])
				data.elem[iel].bcs[iface][2] = iface + 1
				data.elem[iel].bcs[iface][3] = float(line[10:24])
				data.elem[iel].bcs[iface][4] = float(line[24:38])
				data.elem[iel].bcs[iface][5] = float(line[38:52])
				data.elem[iel].bcs[iface][6] = float(line[52:66])
				data.elem[iel].bcs[iface][7] = float(line[66:80])
			else:
				data.elem[iel].bcs[iface][0] = line[1:3].strip()
				data.elem[iel].bcs[iface][1] = int(line[4:15])
				data.elem[iel].bcs[iface][2] = int(line[15:16])
				data.elem[iel].bcs[iface][3] = float(line[16:34])
				data.elem[iel].bcs[iface][4] = float(line[34:52])
				data.elem[iel].bcs[iface][5] = float(line[52:70])
				data.elem[iel].bcs[iface][6] = float(line[70:88])
				data.elem[iel].bcs[iface][7] = float(line[88:106])
	#
	#---------------------------------------------------------------------------
	# FORGET ABOUT WHAT FOLLOWS
	#---------------------------------------------------------------------------	
	#
	#
	# close file
	infile.close()
	#
	# output
	return data


#==============================================================================
def writerea(fname, data):
	"""
	    writerea
	    A function for writing ascii .rea files for nek5000

	    input variables:
	    fname : file name
		 data : exadata data organised as in exadata.py
	"""
	#
	try:
		outfile = open(fname, 'w')
	except IOError as e:
		print('I/O error ({0}): {1}'.format(e.errno, e.strerror))
		#return -1
	#
	#---------------------------------------------------------------------------
	# READ HEADER (2 lines) + ndim + number of parameters
	#---------------------------------------------------------------------------
	#
	outfile.write('****** PARAMETERS ******\n')
	outfile.write('   2.6000     NEKTON VERSION\n')
	outfile.write('   {0:1d} DIMENSIONAL RUN\n'.format(data.ndim))
	outfile.write('         118 PARAMETERS FOLLOW\n')
	outfile.write('   1.00000     P001: DENSITY\n')
	outfile.write('  -1000.00     P002: VISCOSITY\n')
	outfile.write('   0.00000     P003: BETAG\n')
	outfile.write('   0.00000     P004: GTHETA\n')
	outfile.write('   0.00000     P005: PGRADX\n')
	outfile.write('   0.00000     P006: \n')
	outfile.write('   1.00000     P007: RHOCP\n')
	outfile.write('   1.00000     P008: CONDUCT\n')
	outfile.write('   0.00000     P009: \n')
	outfile.write('   0.00000     P010: FINTIME\n')
	outfile.write('   103         P011: NSTEPS\n')
	outfile.write('  -1.00000E-03 P012: DT\n')
	outfile.write('   0.00000     P013: IOCOMM\n')
	outfile.write('   0.00000     P014: IOTIME\n')
	outfile.write('   10          P015: IOSTEP\n')
	outfile.write('   0.00000     P016: PSSOLVER: 0=default\n')
	outfile.write('   1.00000     P017: \n')
	outfile.write('   0.00000     P018: GRID <0 --> # cells on screen\n')
	outfile.write('   0.00000     P019: INTYPE\n')
	outfile.write('   10.0000     P020: NORDER\n')
	outfile.write('   1.00000E-09 P021: DIVERGENCE\n')
	outfile.write('   1.00000E-09 P022: HELMHOLTZ\n')
	outfile.write('   0.00000     P023: NPSCAL\n')
	outfile.write('   1.00000E-02 P024: TOLREL\n')
	outfile.write('   1.00000E-02 P025: TOLABS\n')
	outfile.write('   1.00000     P026: COURANT/NTAU\n')
	outfile.write('   3.00000     P027: TORDER\n')
	outfile.write('   0.00000     P028: TORDER: mesh velocity (0: p28=p27)\n')
	outfile.write('   0.00000     P029: = magnetic visc if > 0, = -1/Rm if < 0\n')
	outfile.write('   0.00000     P030: > 0 ==> properties set in uservp()\n')
	outfile.write('   0.00000     P031: NPERT: #perturbation modes\n')
	outfile.write('   0.00000     P032: #BCs in re2 file, if > 0\n')
	outfile.write('   0.00000     P033: \n')
	outfile.write('   0.00000     P034: \n')
	outfile.write('   0.00000     P035: \n')
	outfile.write('   0.00000     P036: XMAGNET\n')
	outfile.write('   0.00000     P037: NGRIDS\n')
	outfile.write('   0.00000     P038: NORDER2\n')
	outfile.write('   0.00000     P039: NORDER3\n')
	outfile.write('   0.00000     P040: \n')
	outfile.write('   0.00000     P041: 1-->multiplicattive SEMG\n')
	outfile.write('   0.00000     P042: 0=gmres/1=pcg\n')
	outfile.write('   0.00000     P043: 0=semg/1=schwarz\n')
	outfile.write('   0.00000     P044: 0=E-based/1=A-based prec.\n')
	outfile.write('   0.00000     P045: Relaxation factor for DTFS\n')
	outfile.write('   0.00000     P046: reserved\n')
	outfile.write('   0.00000     P047: vnu: mesh material prop.\n')
	outfile.write('   0.00000     P048: \n')
	outfile.write('   0.00000     P049: \n')
	outfile.write('   0.00000     P050: \n')
	outfile.write('   0.00000     P051: \n')
	outfile.write('   0.00000     P052: IOHIS\n')
	outfile.write('   0.00000     P053: \n')
	outfile.write('   0.00000     P054: fixed flow rate dir: |p54|=1,2,3=x,y,z\n')
	outfile.write('   0.00000     P055: vol.flow rate (p54>0) or Ubar (p54<0)\n')
	outfile.write('   0.00000     P056: \n')
	outfile.write('   0.00000     P057: \n')
	outfile.write('   0.00000     P058: \n')
	outfile.write('   0.00000     P059: !=0 --> full Jac. eval. for each el.\n')
	outfile.write('   0.00000     P060: !=0 --> init. velocity to small nonzero\n')
	outfile.write('   0.00000     P061: \n')
	outfile.write('   0.00000     P062: >0 --> force byte_swap for output\n')
	outfile.write('   8.00000     P063: =8 --> force 8-byte output\n')
	outfile.write('   0.00000     P064: =1 --> perturbation restart\n')
	outfile.write('   1.00000     P065: #iofiles (eg, 0 or 64); <0 --> sep. dirs\n')
	outfile.write('   6.00000     P066: output : <0=ascii, else binary\n')
	outfile.write('   6.00000     P067: restart: <0=ascii, else binary\n')
	outfile.write('   0.00000     P068: iastep: freq for avg_all (0=iostep)\n')
	outfile.write('   0.00000     P069:       : freq of srf dump\n')
	outfile.write('   0.00000     P070: \n')
	outfile.write('   0.00000     P071: \n')
	outfile.write('   0.00000     P072: \n')
	outfile.write('   0.00000     P073: \n')
	outfile.write('   0.00000     P074: \n')
	outfile.write('   0.00000     P075: \n')
	outfile.write('   0.00000     P076: \n')
	outfile.write('   0.00000     P077: \n')
	outfile.write('   0.00000     P078: \n')
	outfile.write('   0.00000     P079: \n')
	outfile.write('   0.00000     P080: \n')
	outfile.write('   0.00000     P081: \n')
	outfile.write('   0.00000     P082: \n')
	outfile.write('   0.00000     P083: \n')
	outfile.write('   0.00000     P084: != 0 --> sets initial timestep if p12>0\n')
	outfile.write('   0.00000     P085: dt retio of p84 !=0, for timesteps>0\n')
	outfile.write('   0.00000     P086: reserved\n')
	outfile.write('   0.00000     P087: \n')
	outfile.write('   0.00000     P088: \n')
	outfile.write('   0.00000     P089: \n')
	outfile.write('   0.00000     P090: \n')
	outfile.write('   0.00000     P091: \n')
	outfile.write('   0.00000     P092: \n')
	outfile.write('   20.0000     P093: Number of previous pressure solns saved\n')
	outfile.write('   9.00000     P094: start projecting velocity after p94 step\n')
	outfile.write('   9.00000     P095: start projecting pressure after p95 step\n')
	outfile.write('   0.00000     P096: \n')
	outfile.write('   0.00000     P097: \n')
	outfile.write('   0.00000     P098: \n')
	outfile.write('   3.00000     P099: dealiasing: <0--> off /3--> old /4-->new\n')
	outfile.write('   0.00000     P100: \n')
	outfile.write('   0.00000     P101: Number of additional modes to filter\n')
	outfile.write('   0.00000     P102: Dump out divergence at each time step\n')
	outfile.write('   0.01000     P103: weight of stabilizing filter\n')
	outfile.write('   0.00000     P104: \n')
	outfile.write('   0.00000     P105: \n')
	outfile.write('   0.00000     P106: \n')
	outfile.write('   0.00000     P107: !=0 --> add h2 array in hmholtz eqn\n')
	outfile.write('   0.00000     P108: \n')
	outfile.write('   0.00000     P109: \n')
	outfile.write('   0.00000     P110: \n')
	outfile.write('   0.00000     P111: \n')
	outfile.write('   0.00000     P112: \n')
	outfile.write('   0.00000     P113: \n')
	outfile.write('   0.00000     P114: \n')
	outfile.write('   0.00000     P115: \n')
	outfile.write('   0.00000     P116: \n')
	outfile.write('   0.00000     P117: \n')
	outfile.write('   0.00000     P118: \n')
	outfile.write('      4  Lines of passive scalar data follows2 CONDUCT, 2RHOCP\n')
	outfile.write('   1.00000        1.00000        1.00000        1.00000        1.00000\n')
	outfile.write('   1.00000        1.00000        1.00000        1.00000\n')
	outfile.write('   1.00000        1.00000        1.00000        1.00000        1.00000\n')
	outfile.write('   1.00000        1.00000        1.00000        1.00000\n')
	outfile.write('         13   LOGICAL SWITCHES FOLLOW\n')
	outfile.write(' T      IFFLOW\n')
	outfile.write(' F      IFHEAT\n')
	outfile.write(' T      IFTRAN\n')
	outfile.write(' T F F F F F F F F F F  IFNAV & IFADVC (convection in P.S. fields)\n')
	outfile.write(' F F T T T T T T T T T T  IFTMSH (IF mesh for this field is T mesh)\n')
	outfile.write(' F      IFAXIS\n')
	outfile.write(' F      IFSTRS\n')
	outfile.write(' F      IFSPLIT\n')
	outfile.write(' F      IFMGRID\n')
	outfile.write(' F      IFMODEL\n')
	outfile.write(' F      IFKEPS\n')
	outfile.write(' F      IFMVBD\n')
	outfile.write(' F      IFCHAR\n')
	outfile.write('   2.00000       2.00000      -1.00000      -1.00000     XFAC,YFAC,XZERO,YZERO\n')
	#
	# vertex data
	outfile.write('  ***** MESH DATA *****  6 lines are X,Y,Z;X,Y,Z. Columns corners 1-4;5-8\n')
	outfile.write('  {0:10d} {1:1d} {2:10d} NEL,NDIM,NELV\n'.format(data.nel, data.ndim, data.nel))
	for iel in range(data.nel):
		outfile.write('           ELEMENT {0:10d} [  1a]    GROUP 1\n'.format(iel+1))
		if (data.ndim == 2):
			outfile.write('{0:14.6e}{1:14.6e}{2:14.6e}{3:14.6e}\n'.format(data.elem[iel].pos[0, 0, 0, 0], data.elem[iel].pos[0, 0, 0, -1], data.elem[iel].pos[0, 0, -1, -1], data.elem[iel].pos[0, 0, -1, 0]))
			outfile.write('{0:14.6e}{1:14.6e}{2:14.6e}{3:14.6e}\n'.format(data.elem[iel].pos[1, 0, 0, 0], data.elem[iel].pos[1, 0, 0, -1], data.elem[iel].pos[1, 0, -1, -1], data.elem[iel].pos[1, 0, -1, 0]))
		elif (data.ndim == 3):
			outfile.write('{0:14.6e}{1:14.6e}{2:14.6e}{3:14.6e}\n'.format(data.elem[iel].pos[0, 0, 0, 0], data.elem[iel].pos[0, 0, 0, -1], data.elem[iel].pos[0, 0, -1, -1], data.elem[iel].pos[0, 0, -1, 0]))
			outfile.write('{0:14.6e}{1:14.6e}{2:14.6e}{3:14.6e}\n'.format(data.elem[iel].pos[1, 0, 0, 0], data.elem[iel].pos[1, 0, 0, -1], data.elem[iel].pos[1, 0, -1, -1], data.elem[iel].pos[1, 0, -1, 0]))
			outfile.write('{0:14.6e}{1:14.6e}{2:14.6e}{3:14.6e}\n'.format(data.elem[iel].pos[2, 0, 0, 0], data.elem[iel].pos[2, 0, 0, -1], data.elem[iel].pos[2, 0, -1, -1], data.elem[iel].pos[2, 0, -1, 0]))
			#
			outfile.write('{0:14.6e}{1:14.6e}{2:14.6e}{3:14.6e}\n'.format(data.elem[iel].pos[0, -1, 0, 0], data.elem[iel].pos[0, -1, 0, -1], data.elem[iel].pos[0, -1, -1, -1], data.elem[iel].pos[0, -1, -1, 0]))
			outfile.write('{0:14.6e}{1:14.6e}{2:14.6e}{3:14.6e}\n'.format(data.elem[iel].pos[1, -1, 0, 0], data.elem[iel].pos[1, -1, 0, -1], data.elem[iel].pos[1, -1, -1, -1], data.elem[iel].pos[1, -1, -1, 0]))
			outfile.write('{0:14.6e}{1:14.6e}{2:14.6e}{3:14.6e}\n'.format(data.elem[iel].pos[2, -1, 0, 0], data.elem[iel].pos[2, -1, 0, -1], data.elem[iel].pos[2, -1, -1, -1], data.elem[iel].pos[2, -1, -1, 0]))
	#
	# curved side data
	outfile.write('  ***** CURVED SIDE DATA *****\n')
	outfile.write('  {0:10d} Curved sides follow IEDGE,IEL,CURVE(I),I=1,5, CCURVE\n'.format(data.ncurv))
	for iel in range(data.nel):
		if (data.nel < 1e3):
			#
			for iedge in range(12):
				if (data.elem[iel].curv[iedge] != 0.0):
					outfile.write('{0:3d}{1:3d}{2:10.5f}{3:14.5f}{4:14.5f}{5:14.5f}{6:14.5f}     C\n'.format(
						iedge+1, iel+1, data.elem[iel].curv[iedge][0], 0.0, 0.0, 0.0, 0.0))
		elif (data.nel < 1e6):
			#
			for iedge in range(12):
				if (data.elem[iel].curv[iedge] != 0.0):
					outfile.write('{0:2d}{1:6d}{2:10.5f}{3:14.5f}{4:14.5f}{5:14.5f}{6:14.5f}     C\n'.format(
						iedge+1, iel+1, data.elem[iel].curv[iedge][0], 0., 0., 0., 0.))
		else:
			#
			for iedge in range(12):
				if (data.elem[iel].curv[iedge] != 0.0):
					outfile.write('{0:2d}{1:10d}{2:10.5f}{3:14.5f}{4:14.5f}{5:14.5f}{6:14.5f}     C\n'.format(
						iedge+1, iel+1, data.elem[iel].curv[iedge][0], 0.0, 0.0, 0.0, 0.0))
	#
	# boundary conditions data
	outfile.write('  ***** BOUNDARY CONDITIONS *****\n')
	outfile.write('  ***** FLUID BOUNDARY CONDITIONS *****\n')
	for iel in range(data.nel):
		for iface in range(2*data.ndim):
			if (data.nel < 1e3):
				outfile.write(' {0:2s} {1:3d}{2:3d}{3:14.6e}{4:14.6e}{5:14.6e}{6:14.6e}{7:14.6e}\n'.format(
					data.elem[iel].bcs[iface][0], data.elem[iel].bcs[iface][1], data.elem[iel].bcs[iface][2], data.elem[iel].bcs[iface][3], data.elem[iel].bcs[iface][4], data.elem[iel].bcs[iface][5], data.elem[iel].bcs[iface][6], data.elem[iel].bcs[iface][7]))
			elif (data.nel < 1e5):
				outfile.write(' {0:2s} {1:6d}{2:14.6e}{3:14.6e}{4:14.6e}{5:14.6e}{6:14.6e}\n'.format(
					data.elem[iel].bcs[iface][0], data.elem[iel].bcs[iface][1], data.elem[iel].bcs[iface][3], data.elem[iel].bcs[iface][4], data.elem[iel].bcs[iface][5], data.elem[iel].bcs[iface][6], data.elem[iel].bcs[iface][7]))
			elif (data.nel < 1e6):
				outfile.write(' {0:2s} {1:6d}{2:14.6e}{3:14.6e}{4:14.6e}{5:14.6e}{6:14.6e}\n'.format(
					data.elem[iel].bcs[iface][0], data.elem[iel].bcs[iface][1], data.elem[iel].bcs[iface][3], data.elem[iel].bcs[iface][4], data.elem[iel].bcs[iface][5], data.elem[iel].bcs[iface][6], data.elem[iel].bcs[iface][7]))
			else:
				outfile.write(' {0:2s} {1:11d}{2:1d}{3:18.11e}{4:18.11e}{5:18.11e}{6:18.11e}{7:18.11e}\n'.format(
					data.elem[iel].bcs[iface][0], data.elem[iel].bcs[iface][1], data.elem[iel].bcs[iface][2], data.elem[iel].bcs[iface][3], data.elem[iel].bcs[iface][4], data.elem[iel].bcs[iface][5], data.elem[iel].bcs[iface][6], data.elem[iel].bcs[iface][7]))

	outfile.write('  ***** NO THERMAL BOUNDARY CONDITIONS *****\n')
	outfile.write('    0 PRESOLVE/RESTART OPTIONS  *****\n')
	outfile.write('    7         INITIAL CONDITIONS *****\n')
	outfile.write(' C Default\n')
	outfile.write(' C Default\n')
	outfile.write(' C Default\n')
	outfile.write(' C Default\n')
	outfile.write(' C Default\n')
	outfile.write(' C Default\n')
	outfile.write(' C Default\n')
	outfile.write('  ***** DRIVE FORCE DATA ***** BODY FORCE, FLOW, Q\n')
	outfile.write('    4                 Lines of Drive force data follow\n')
	outfile.write(' C\n')
	outfile.write(' C\n')
	outfile.write(' C\n')
	outfile.write(' C\n')
	outfile.write(' ***** Variable Property Data ***** Overrrides Parameter data.\n')
	outfile.write('    1 Lines follow.\n')
	outfile.write('    0 PACKETS OF DATA FOLLOW\n')
	outfile.write(' ***** HISTORY AND INTEGRAL DATA *****\n')
	outfile.write('    0   POINTS.  Hcode, I,J,H,IEL\n')
	outfile.write(' ***** OUTPUT FIELD SPECIFICATION *****\n')
	outfile.write('    6 SPECIFICATIONS FOLLOW\n')
	outfile.write('    F      COORDINATES\n')
	outfile.write('    T      VELOCITY\n')
	outfile.write('    T      PRESSURE\n')
	outfile.write('    F      TEMPERATURE\n')
	outfile.write('    F      TEMPERATURE GRADIENT\n')
	outfile.write('    0      PASSIVE SCALARS\n')
	outfile.write(' ***** OBJECT SPECIFICATION *****\n')
	outfile.write('        0 Surface Objects\n')
	outfile.write('        0 Volume  Objects\n')
	outfile.write('        0 Edge    Objects\n')
	outfile.write('        0 Point   Objects\n')
	#
	# close file
	outfile.close()
	#
	# output
	return 0
