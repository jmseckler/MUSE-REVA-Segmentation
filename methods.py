import glob, os, shutil


flags = {
	"-bt":{"active":False,"name":"Contrasting","inputs":0,"types":[],"value":[],"help":"Perform enhanced contrasting on image before segmentation"},
	"-ep":{"active":False,"name":"Epineurium","inputs":0,"types":[],"value":[],"help":"Segment epinuerium rather than fasciles"},
	"-ct":{"active":False,"name":"Factor","inputs":1,"types":["float"],"value":[],"help":"Sets contrast factor of image preprocessing before segmentation, accepts a float variable. Default: 1.0"},
	"-l":{"active":False,"name":"Lattice","inputs":1,"types":["int"],"value":[],"help":"Sets lattice spacing for points to be selected to be segmented. Default: 50"},
	"-dl":{"active":False,"name":"Dot Lattice Debug","inputs":0,"types":[],"value":[],"help":"Writes the lattice points chosen for each image to a file in /dots/"},
	}


def input_parser(argv):
	n = len(argv)
	
	for i in range(n):
		tag = argv[i]
		if tag in flags:
			flags[tag]['active'] = True
			
			for j in range(flags[tag]['inputs']):
				try:
					if flags[tag]['types'][j] == "int":
						flags[tag]['value'].append(int(argv[i + j + 1]))
					elif flags[tag]['types'][j] == "float":
						flags[tag]['value'].append(float(argv[i + j + 1]))
					elif argv[i + j + 1][0] != '-':
						flags[tag]['value'].append(argv[i + j + 1])
				except:
					pass
	return flags	

def find_file_list(path):
	rawlist = glob.glob(path + '*.png')
	flist = []
	for path in rawlist:
		fname = path.split('.')[1]
		fname = fname.split('_')[-1]
		flist.append(int(fname))
	flist = sorted(flist)
	
	for i in range(len(flist)):
		addon = ''
		if flist[i] < 10:
			addon = '00'
		elif flist[i] < 100:
			addon = '0'

		flist[i] = addon + str(flist[i])
	return flist

def find_mask_path(image_number,basepath):
	addon = ''
	if int(image_number) < 1000:
		addon = '0'
	path = basepath + f"mask_{addon + image_number}.png"
	return path

def replace_directory(directory):
	if os.path.isdir(directory):
		shutil.rmtree(directory)
	os.makedirs(directory)

def make_directory(directory):
	if not os.path.isdir(directory):
		os.makedirs(directory)


