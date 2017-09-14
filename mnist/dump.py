
from PIL import Image

with open('train-images-idx3-ubyte', 'rb') as f:
	f.read(16)
	for z in range(1,100):
		im = Image.new('RGB',(28,28))
		for y in range(0,28):
			for x in range(0,28):
				b = int(f.read(1)[0])
				im.putpixel((x,y), (b,b,b))
		im.save('dump/%s.png' % z, 'png')
