# =============================================================================#
# neksuite                                                                    #
#                                                                             #
# A python module for reading and writing nek5000 files                       #
#                                                                             #
# Authors: Jacopo Canton, Nicolo' Fabbiane                                    #
# Contacts: jcanton(at)mech.kth.se, nicolo(at)mech.kth.se                     #
# Last edit: 2015-10-19                                                       #
#                                                                             #
# Revised 4/21/21 by Alan Kaptanoglu                                          #
# Files were combined and truncated just to provide the readnek               #
# routine, which is needed for reading data in the von Karman example         #
# Lastly, they were changed to conform to pep8                                #
# =============================================================================#
import struct

import numpy as np


class datalims:
    """
    datalims
    A class containing the extrema of all quantities stored in the mesh
    """

    def __init__(self, var):
        self.pos = np.zeros((3, 2))
        self.vel = np.zeros((3, 2))
        self.pres = np.zeros((var[2], 2))
        self.temp = np.zeros((var[3], 2))
        self.scal = np.zeros((var[4], 2))


class elem:
    """
    elem
    A class containing one nek element/SIMSON flow field
    """

    def __init__(self, var, lr1):
        self.pos = np.zeros((3, lr1[2], lr1[1], lr1[0]))
        self.curv = np.zeros((12, 1))
        self.vel = np.zeros((3, lr1[2], lr1[1], lr1[0]))
        self.pres = np.zeros((var[2], lr1[2], lr1[1], lr1[0]))
        self.temp = np.zeros((var[3], lr1[2], lr1[1], lr1[0]))
        self.scal = np.zeros((var[4], lr1[2], lr1[1], lr1[0]))
        self.bcs = np.zeros((6), dtype="a1, i4, i4, f8, f8, f8, f8, f8")


class exadata:
    """
    data
    A class containing data for reading/writing binary simulation files
    """

    def __init__(self, ndim, nel, lr1, var):
        self.ndim = ndim
        self.nel = nel
        self.ncurv = []
        self.var = var
        self.lr1 = lr1
        self.time = []
        self.istep = []
        self.wdsz = []
        self.endian = []
        self.lims = datalims(var)
        self.elem = [elem(var, lr1) for i in range(nel)]


def readnek(fname):
    """
    readnek
    A function for reading binary data from the nek5000 binary format

    input variable:
    fname : file name
    """
    try:
        infile = open(fname, "rb")
    except IOError as e:
        print("I/O error ({0}): {1}".format(e.errno, e.strerror))
        return -1

        # read header
    header = infile.read(132).split()

    # get word size
    wdsz = int(header[1])
    if wdsz == 4:
        realtype = "f"
    elif wdsz == 8:
        realtype = "d"
    else:
        print("ERROR: could not interpret real type (wdsz = %i)" % (wdsz))
        return -2

        # get polynomial order
    lr1 = [int(header[2]), int(header[3]), int(header[4])]

    # compute total number of points per element
    npel = lr1[0] * lr1[1] * lr1[2]

    # get number of physical dimensions
    ndim = 2 + (lr1[2] > 1)

    # get number of elements
    nel = int(header[5])

    # get number of elements in the file
    nelf = int(header[6])

    # get current time
    time = float(header[7])

    # get current time step
    istep = int(header[8])

    # get variables [XUPT]
    vars = header[11].decode("utf-8")
    var = [0 for i in range(5)]
    for v in vars:
        if v == "X":
            var[0] = ndim
        elif v == "U":
            var[1] = ndim
        elif v == "P":
            var[2] = 1
        elif v == "T":
            var[3] = 1
        elif v == "S":
            var[4] = 0

    # identify endian encoding
    etagb = infile.read(4)
    etagL = struct.unpack("<f", etagb)[0]
    etagL = int(etagL * 1e5) / 1e5
    etagB = struct.unpack(">f", etagb)[0]
    etagB = int(etagB * 1e5) / 1e5
    if etagL == 6.54321:
        emode = "<"
    elif etagB == 6.54321:
        emode = ">"
    else:
        print("ERROR: could not interpret endianness")
        return -3

        # read element map for the file
    elmap = infile.read(4 * nelf)
    elmap = list(struct.unpack(emode + nelf * "i", elmap))

    # initialize data structure
    data = exadata(ndim, nel, lr1, var)
    data.time = time
    data.istep = istep
    data.wdsz = wdsz
    if emode == "<":
        data.endian = "little"
    elif emode == ">":
        data.endian = "big"

        # read geometry
    data.lims.pos[:, 0] = float("inf")
    data.lims.pos[:, 1] = -float("inf")
    for iel in elmap:
        for idim in range(var[0]):  # if var[0] == 0, geometry is not read
            fi = infile.read(npel * wdsz)
            fi = list(struct.unpack(emode + npel * realtype, fi))
            ip = 0
            for iz in range(lr1[2]):
                for iy in range(lr1[1]):
                    data.elem[iel - 1].pos[idim, iz, iy, :] = fi[ip : ip + lr1[0]]
                    ip += lr1[0]
            data.lims.pos[idim, 0] = min([data.lims.pos[idim, 0]] + fi)
            data.lims.pos[idim, 1] = max([data.lims.pos[idim, 1]] + fi)

            # read velocity
    data.lims.vel[:, 0] = float("inf")
    data.lims.vel[:, 1] = -float("inf")
    for iel in elmap:
        for idim in range(var[1]):  # if var[1] == 0, velocity is not read
            fi = infile.read(npel * wdsz)
            fi = list(struct.unpack(emode + npel * realtype, fi))
            ip = 0
            for iz in range(lr1[2]):
                for iy in range(lr1[1]):
                    data.elem[iel - 1].vel[idim, iz, iy, :] = fi[ip : ip + lr1[0]]
                    ip += lr1[0]
            data.lims.vel[idim, 0] = min([data.lims.vel[idim, 0]] + fi)
            data.lims.vel[idim, 1] = max([data.lims.vel[idim, 1]] + fi)

            # read pressure
    data.lims.pres[:, 0] = float("inf")
    data.lims.pres[:, 1] = -float("inf")
    for iel in elmap:
        for ivar in range(var[2]):  # if var[2] == 0, pressure is not read
            fi = infile.read(npel * wdsz)
            fi = list(struct.unpack(emode + npel * realtype, fi))
            ip = 0
            for iz in range(lr1[2]):
                for iy in range(lr1[1]):
                    data.elem[iel - 1].pres[ivar, iz, iy, :] = fi[ip : ip + lr1[0]]
                    ip += lr1[0]
            data.lims.pres[ivar, 0] = min([data.lims.pres[ivar, 0]] + fi)
            data.lims.pres[ivar, 1] = max([data.lims.pres[ivar, 1]] + fi)

            # read temperature
    data.lims.temp[:, 0] = float("inf")
    data.lims.temp[:, 1] = -float("inf")
    for iel in elmap:
        for ivar in range(var[3]):  # if var[3] == 0, temperature is not read
            fi = infile.read(npel * wdsz)
            fi = list(struct.unpack(emode + npel * realtype, fi))
            ip = 0
            for iz in range(lr1[2]):
                for iy in range(lr1[1]):
                    data.elem[iel - 1].temp[ivar, iz, iy, :] = fi[ip : ip + lr1[0]]
                    ip += lr1[0]
            data.lims.temp[ivar, 0] = min([data.lims.temp[ivar, 0]] + fi)
            data.lims.temp[ivar, 1] = max([data.lims.temp[ivar, 1]] + fi)

            # read scalar fields
    data.lims.scal[:, 0] = float("inf")
    data.lims.scal[:, 1] = -float("inf")
    for iel in elmap:
        for ivar in range(var[4]):  # if var[4] == 0, scalars are not read
            fi = infile.read(npel * wdsz)
            fi = list(struct.unpack(emode + npel * realtype, fi))
            ip = 0
            for iz in range(lr1[2]):
                for iy in range(lr1[1]):
                    data.elem[iel - 1].scal[ivar, iz, iy, :] = fi[ip : ip + lr1[0]]
                    ip += lr1[0]
            data.lims.scal[ivar, 0] = min([data.lims.scal[ivar, 0]] + fi)
            data.lims.scal[ivar, 1] = max([data.lims.scal[ivar, 1]] + fi)

            # close file and return
    infile.close()
    return data
