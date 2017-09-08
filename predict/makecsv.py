#!/usr/bin/env python3

import sys
from PIL import Image # install pillow

def get_gray(im, x, y):
	p = im.getpixel((x,y))
	return int(sum(p[0:3])/3)

if __name__ == '__main__':
	files = sys.argv[1:]
	for f in files:
		im = Image.open(f)
		w,h = im.size
		#print(dir(im))
		a = ','.join(str(get_gray(im,x,y)) for y in range(0, h) for x in range(0, w))
		print(a)
