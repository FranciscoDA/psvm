#! /usr/bin/env python3
import idxutils
import sys
from os.path import join

def usage(f):
	print('Usage:', file=f)
	print('{:s} INFILE OUTFILE [SAMPLE_ID_LIST ...]'.format(sys.argv[0]), file=f)

if __name__ == '__main__':
	if len(sys.argv) < 4:
		usage(sys.stderr)
		sys.exit(1)

	ids = [int(i) for i in sys.argv[3:]]
	ids.sort()
	infile = sys.argv[1]
	outfile = sys.argv[2]
	with open(infile, 'rb') as f, open(outfile, 'w') as g:
		hdata = idxutils.readheader(f)
		for sampleid in ids:
			sdata = idxutils.readsample(f, hdata, sampleid)
			g.write(','.join(str(x) for x in sdata.flat) + '\n')

