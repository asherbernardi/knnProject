# author: Asher Bernardi
from PIL import Image, ImageOps

def parseCellGrayscales(cell):
	grayscales = ''
	pixels = cell.load()
	for row in range(cell.height):
		for col in range(cell.width):
			# We invert the grayscales so that the black indicates positive
			# direction in the data vector
			# We can take the first RGB integer [0], since it is graysc
			grayscales += str(255 - pixels[col,row][0])
			if row != cell.height - 1 or col != cell.width - 1:
				grayscales += ','
	return grayscales + '\n'

def cropAndParse(image, data_file, target_file):
	coords = (40,3,62,35)
	for row in range(3,741,37):
		for col in range(40,593,37):
			cell = image.crop((col,row,col+32,row+32))
			data_file.write(parseCellGrayscales(cell))
	for i in range(10):
		targets = ''
		for j in range(30):
			targets += str(i) + '\n'
		target_file.write(targets)

csv_data = open('digits.data', 'w+')
csv_target = open('digits.target', 'w+')

digits = Image.open('digits1.png')
cropAndParse(digits, csv_data, csv_target)
digits.close()
digits = Image.open('digits2.png')
cropAndParse(digits, csv_data, csv_target)
digits.close()
digits = Image.open('digits3.png')
cropAndParse(digits, csv_data, csv_target)
digits.close()
digits = Image.open('digits4.png')
cropAndParse(digits, csv_data, csv_target)
digits.close()

# Now we try with 16x16 pictures
def cropAndParse(image, data_file, target_file):
	coords = (40,3,62,35)
	for row in range(3,741,37):
		for col in range(40,593,37):
			cell = image.crop((col,row,col+32,row+32))
			cell = cell.resize((16,16))
			data_file.write(parseCellGrayscales(cell))
	for i in range(10):
		targets = ''
		for j in range(30):
			targets += str(i) + '\n'
		target_file.write(targets)

csv_data = open('digits16.data', 'w+')
csv_target = open('digits16.target', 'w+')

digits = Image.open('digitsline1.png')
cropAndParse(digits, csv_data, csv_target)
digits.close()
digits = Image.open('digitsline2.png')
cropAndParse(digits, csv_data, csv_target)
digits.close()
digits = Image.open('digitsline3.png')
cropAndParse(digits, csv_data, csv_target)
digits.close()
digits = Image.open('digitsline4.png')
cropAndParse(digits, csv_data, csv_target)
digits.close()

csv_data.close()
csv_target.close()