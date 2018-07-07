#! /usr/bin/env python3
from PIL import Image
import idxutils
import sys
from os.path import join

def usage(f):
	print('Usage:', file=f)
	print('{:s} INFILE OUTDIR [SAMPLE_ID_LIST ...]'.format(sys.argv[0]), file=f)

if __name__ == '__main__':
	if len(sys.argv) < 4:
		usage(sys.stderr)
		sys.exit(1)

	ids = [int(i) for i in sys.argv[3:]]
	ids.sort()
	infile = sys.argv[1]
	outdir = sys.argv[2]
	with open(infile, 'rb') as f:
		hdata = idxutils.readheader(f)
		for sampleid in ids:
			sdata = idxutils.readsample(f, hdata, sampleid)
			h, w = sdata.shape[0], sdata.shape[1]
			im = Image.new('RGB', (w, h))
			for y in range(h):
				for x in range(w):
					if len(sdata.shape) == 3:
						color = sdata[y,x,:]
					else:
						color = tuple([sdata[y,x]]*3)
					im.putpixel((x, y), color)
			im.save(join(outdir, str(sampleid)+'.png'))

