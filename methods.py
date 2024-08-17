import warnings
warnings.filterwarnings("ignore")

import glob, os, shutil, zarr, json


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

def find_file_list(path,override = 1,thousanth=True):
	rawlist = glob.glob(path + '*.png')
	flist = []
	for path in rawlist:
		fname = path.split('.')[override]
		fname = fname.split('_')[-1]
		flist.append(int(fname))
	flist = sorted(flist)
	for i in range(len(flist)):
		addon = ''
		if thousanth:
			if flist[i] < 10:
				addon = '000'
			elif flist[i] < 100:
				addon = '00'
			elif flist[i] < 1000:
				addon = '0'
		else:
			if flist[i] < 10:
				addon = '00'
			elif flist[i] < 100:
				addon = '0'
			

		flist[i] = addon + str(flist[i])
	return flist

def find_mask_path(image_number,basepath,varInt=False):
	addon = ''
	if varInt:
		if int(image_number) < 10:
			addon = '000'
		elif int(image_number) < 100:
			addon = '00'
		elif int(image_number) < 1000:
			addon = '0'
	else:
		if int(image_number) < 1000:
			addon = '0'
	baseNum = addon + str(image_number)
	path = basepath + f"mask_{baseNum}.png"
	return path

def replace_directory(directory):
	if os.path.isdir(directory):
		shutil.rmtree(directory)
	os.makedirs(directory)

def make_directory(directory):
	if not os.path.isdir(directory):
		os.makedirs(directory)

def create_basic_zarr_file(path,fname):
	zarr_path = path + '/' + fname + '.zarr'
	if os.path.isdir(zarr_path):
		shutil.rmtree(zarr_path)
	store = zarr.DirectoryStore(zarr_path, dimension_separator='/')
	root = zarr.group(store=store, overwrite=True)
	data = root.create_group('data')
	return data

def shape_definer(n,x,y,scale):
	zshape = (n,int(x / scale),int(y / scale))
	zchunk = (4,int(x / scale),int(y / scale))
	return zshape, zchunk

def save_data(path,data):
	path = path + 'data.json'
	with open(path, 'w') as json_file:
		json.dump(data, json_file)

def load_data(name):
	path = name + 'data.json'
	with open(path, 'r') as json_file:
		data = json.load(json_file)
	return data

