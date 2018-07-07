
import struct
import numpy as np

def readheader(f):
	f.seek(0)
	H0 = struct.Struct('>HBB')
	(zero, datatype, ndims) = H0.unpack(f.read(H0.size))
	if zero != 0:
		return None
	H1 = struct.Struct('>I{:d}I'.format(ndims-1))
	(n, *dimsizes) = H1.unpack(f.read(H1.size))
	pointfmt = {
		8: 'B',
		9: 'b',
		11: 'h',
		12: 'i',
		13: 'f',
		14: 'd'
	}[datatype]
	pointspersample = 1
	for ds in dimsizes:
		pointspersample *= ds
	SS = struct.Struct('>{:d}{:s}'.format(pointspersample, pointfmt))
	return (pointfmt, ndims, n, dimsizes, H0, H1, SS)

def readsample(f, headerdata, i):
	(datatype, ndims, n, dimsizes, H0, H1, SS) = headerdata
	f.seek(H0.size + H1.size + SS.size*i)
	sampledata = SS.unpack(f.read(SS.size))
	sample = np.array(sampledata)
	sample.shape = tuple(dimsizes)
	return sample

