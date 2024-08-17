import methods as ms
import numpy as np
import cv2 as cv
import sys
from tqdm import tqdm


assert len(sys.argv) > 1, "Requires a specified file name"

BASEPATH = './output/' + sys.argv[1] + '/'
inPath = BASEPATH  + 'segment/'
outPath = BASEPATH  + 'fascicles/'
ms.replace_directory(outPath)

flist = ms.find_file_list(inPath)

fasicles = {}
classNext = 1

def find_all_fasicles_and_make_class_mask(image_number):
	path = inPath + "mask_" + image_number + ".png"
	img = cv.imread(path,cv.IMREAD_GRAYSCALE)
	_, mask = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
	
	means = []
	volumes = []
	class_number = 1
	
	classMask = np.zeros(img.shape)
	
	heirarchy, contours = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
	for h in heirarchy:
		if h.shape[0] > 50:
			means.append(np.mean(h,axis=0)[0].astype('int'))
			tmpMask, tmpVolume = add_class_to_mask(classMask.shape,class_number,h,means[-1])
			classMask = classMask + tmpMask
			volumes.append(tmpVolume)
			class_number += 1
	means = np.array(means)
	volumes = np.array(volumes)
	return means, volumes, classMask


def add_class_to_mask(shape,class_number,bound,mean):
	mask = np.zeros(shape)
	for i in range(bound.shape[0]):
		start_pixel = tuple(bound[i - 1][0])
		end_pixel = tuple(bound[i][0])
		mask[bound[i][0][1]][bound[i][0][0]] = 255
		cv.line(mask, start_pixel, end_pixel, 255, 1)
	mask = mask.astype('uint8')
	pixel = (int(mean[0]),int(mean[1]))
	_, newmask = cv.threshold(mask, 1, 255, cv.THRESH_BINARY)
	h, w = mask.shape[:2]
	mask_fill = np.zeros((h+2, w+2), np.uint8)
	
	
	cv.floodFill(newmask, mask_fill, pixel, 255)
	
	newmask = np.clip(cv.bitwise_not(newmask),0,1)
	newmask = 1 - newmask
	volume = np.sum(newmask)
	newmask = class_number * newmask
	return newmask, volume


def match_fasicles_to_preexisting_classes(i):
	global fasicles, zClass
	fname = flist[i]
	means, volumes, img = find_all_fasicles_and_make_class_mask(fname)
	
	n = means.shape[0]
	decode = {}
	if i == 0:
		for j in range(n):
			create_new_fasicle(i,None,(int(means[j][0]),int(means[j][1])),int(volumes[j]))
		zClass[i] = img
	else:
		for j in range(n)	:
			decode[j+1] = match_class_with_fasicle(img,j+1,i)
		
		for j in range(n):
			if decode[j+1] < 1:
				decode[j+1] = create_new_fasicle(i,None,(int(means[j][0]),int(means[j][1])),int(volumes[j]))
		decode = parse_decoder_and_assign_fasicles(decode,img,volumes,means,i)
		update_zClass_with_new_decoded_data(decode,img,i)
		
		for j in range(n):
			f = decode[j+1]
			fasicles[f]['centroid'].append((int(means[j][0]),int(means[j][1])))
			fasicles[f]['volume'].append(int(volumes[j]))
		
		
	

def create_new_fasicle(i,parent,centroid,volume):
	global fasicles, classNext
	index = classNext
	while index in fasicles:
		classNext += 1
		index = classNext
	fasicles[index] = {'active':True,'start':i,'end':-1,'parents':parent,'children':[],'merge':None,'centroid':[centroid],'volume':[volume]}
	return index

def check_if_two_classes_overlap(newClass,baseClass,new,base):
	mask = (newClass == new)
	nClass = mask * newClass
	nClass = np.clip(nClass,0,1)
	
	mask = (baseClass == base)
	bClass = mask * baseClass
	bClass = np.clip(bClass,0,1)
	
	overlap = nClass * bClass
	overlap = np.sum(overlap)
	return overlap

def match_class_with_fasicle(newClass,new,i):
	matched_class = -1
	matched_overlap = 0
	
	for f in fasicles:
		if fasicles[f]['active']:
			overlap = check_if_two_classes_overlap(newClass,zClass[i-1],new,f)
			if overlap > matched_overlap:
				matched_class = f
				matched_overlap = overlap
	
	return matched_class

def parse_decoder_and_assign_fasicles(decode,img,volumes,means,index):
	fas_list = {}
	for f in fasicles:
		if fasicles[f]['active']:
			fas_list[f] = 0
	for i in decode:
		fas_list[decode[i]] += 1
	
	for f in fas_list:
		if fas_list[f] == 0:
			merge_fasicle(f,img,index,decode)
		elif fas_list[f] > 1:
			decode = split_fasicle(f,decode,volumes, means,index)
	return decode

def merge_fasicle(f,newClass,index,decode):
	global fasicles
	matched_class = -1
	matched_overlap = 0
	
	for i in decode:
		overlap = check_if_two_classes_overlap(zClass[index-1],newClass,f,i)
		if overlap > matched_overlap:
			matched_class = i
			matched_overlap = overlap
	
	fasicles[f]['active'] = False
	fasicles[f]['end'] = index
	if matched_class > 0:
		fasicles[f]['merge'] = matched_class



	
def split_fasicle(f,decode,volumes, means,index):
	splits = []
	for i in decode:
		if decode[i] == f:
			splits.append(i)
	
	primary = splits[0]
	volume = volumes[primary-1]
	for i in splits:
		if volumes[i-1] > volume:
			volume = volumes[i-1]
			primary = i
	for i in splits:
		if i != primary:
			decode[i] = create_new_fasicle(index,f,(int(means[i-1][0]),int(means[i-1][1])),int(volumes[i-1]))
	return decode

def update_zClass_with_new_decoded_data(decode,img,index):
	global zClass
	newClass = np.zeros(img.shape)
	
	for i in decode:
		mask = (img == i)
		nClass = mask * img
		nClass = np.clip(nClass,0,1)
		nClass = decode[i] * nClass
		newClass = newClass + nClass
	
	zClass[index] = newClass

z = len(flist)
path = inPath + "mask_" + flist[0] + ".png"
img = cv.imread(path,cv.IMREAD_GRAYSCALE)

zData = ms.create_basic_zarr_file(outPath,"classMask")
zshape, zchunk = ms.shape_definer(z,img.shape[0],img.shape[1],1)

zClass = zData.zeros('0', shape=zshape, chunks=zchunk, dtype="i2" )


for i in tqdm(range(z)):
	match_fasicles_to_preexisting_classes(i)
#	if i >= 1:
#		break

for f in fasicles:
	print(f,fasicles[f])

ms.save_data(outPath,fasicles)		

