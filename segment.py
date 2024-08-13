from segment_anything import SamPredictor, sam_model_registry
import cv2 as cv
import glob, os, sys
import numpy as np
from tqdm import tqdm
import methods as ms

modelname = 'sam_vit_h_4b8939.pth'

inpath = './input/' + sys.argv[1] + '/'
maskpath = inpath  + 'mask/'

modelpath = './models/' + modelname

BASEPATH = './output/' + sys.argv[1] + '/'
epipath = BASEPATH  + 'epi/'
outpath = BASEPATH  + 'segment/'
debugPath = BASEPATH  +  'dots/'

distance_from_edge_large = 75
distance_from_edge_small = 25 #Not used right now, will be used in next version

elipse_size = 30
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(elipse_size,elipse_size))

contrastFactor = 1.0

lattice_point_spacing = 50

mask_size = 0

flags = {}
flist = []
mlist = []
groundTruth = ''
lattice_points = ''
PATH_TO_DATA = ''

def initial_data_checks():
	global contrastFactor, lattice_point_spacing, flags, flist, mlist, groundTruth, lattice_points,PATH_TO_DATA
	flags = ms.input_parser(sys.argv)
	
	if flags['-ct']['active']:
		contrastFactor = flags['-ct']['value'][0]
	if flags['-l']['active']:
		lattice_point_spacing = flags['-l']['value'][0]
	
	if not os.path.exists(inpath) or not os.path.isdir(inpath):
		print("The input path specified appears to not exist, please check your input...")
		quit()
	
	flist = ms.find_file_list(inpath)
	mlist = ms.find_file_list(maskpath)
	
	path = maskpath + 'image_' + mlist[0] + '.png'
	groundTruth = cv.imread(path,cv.IMREAD_GRAYSCALE)
	lattice_points = calculate_lattice_points(groundTruth)

	if flags['-ep']['active']:
		PATH_TO_DATA = epipath
	else:
		PATH_TO_DATA = outpath
	
	ms.make_directory(BASEPATH)
	ms.make_directory(epipath)
	ms.make_directory(outpath)
	if flags['-dl']['active']:
		ms.make_directory(debugPath)

def calculate_lattice_points(img):
	x = int(img.shape[1] / lattice_point_spacing)
	y = int(img.shape[0] / lattice_point_spacing)
	
	points = []
	for i in range(x):
		if i > 0:
			for j in range(y):
				if y > 0:
					points.append([lattice_point_spacing * i,lattice_point_spacing * j])
	return points

def run_segmentation(m):
	global groundTruth
	if flags['-ep']['active']:
		segment_out_full_nerve(m,groundTruth)
	else:
		groundTruth = segment_out_fasciles(m,groundTruth)

def open_image_and_preprocess(path):
	img = cv.imread(path,cv.IMREAD_GRAYSCALE)
	mean = np.mean(img).astype(int)
	img = img - mean
	img = contrastFactor * img
	img = img + 140
	
	img = np.clip(img,0,255)
	
	if flags['-bt']['active']:
		topHat = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)
		blackHat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)
		img = img + topHat - blackHat
	
	img = np.clip(img,0,255)
	img = img.astype('uint8')
	img = cv.cvtColor(img,cv.COLOR_GRAY2RGB)
	
	return img

def debug(img,points,name):
	image = np.zeros(img.shape)
	for point in points:
		image[point[1]-1][point[0]-1] = 255
		image[point[1]-1][point[0]] = 255
		image[point[1]-1][point[0]+1] = 255
		image[point[1]][point[0]-1] = 255
		image[point[1]][point[0]] = 255
		image[point[1]][point[0]+1] = 255
		image[point[1]+1][point[0]-1] = 255
		image[point[1]+1][point[0]] = 255
		image[point[1]+1][point[0]+1] = 255
	
	cv.imwrite(debugPath + name + '.png',image)
	cv.imwrite(debugPath + 'Mask_' + name + '.png',img)

def find_centroids_and_points_within_segmented_fascicles(MASK,name):
	_, mask = cv.threshold(MASK, 127, 255, cv.THRESH_BINARY)
	
	means = []
	input_points = []
	rect_min = []
	rect_max = []
	
	heirarchy, contours = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
	for h in heirarchy:
		if h.shape[0] > 50:
			means.append(np.mean(h,axis=0)[0].astype('int'))
			rect_min.append(np.amin(h,axis=0)[0])
			rect_max.append(np.amax(h,axis=0)[0])
	
	i = 0
	n = len(lattice_points)
	for h in heirarchy:
		if h.shape[0] > 50:
			for j in range(n):
				distance = cv.pointPolygonTest(h, (lattice_points[j][0], lattice_points[j][1]), True)
				if distance >= distance_from_edge_large:
					input_points.append(lattice_points[j])
			i += 1
	
	if flags['-dl']['active']:
		debug(mask,input_points,name)
	input_points = np.array(input_points)
	means = np.array(means)
	return input_points, means


def segment_out_fasciles(k,MASK):
	global mask_size
	path = inpath + f"image_{k}.png"
	img = open_image_and_preprocess(path)
	
	input_points, input_mean_points = find_centroids_and_points_within_segmented_fascicles(MASK,k)
	n = input_points.shape[0]
	N = input_mean_points.shape[0]
	
	sam = sam_model_registry["vit_h"](checkpoint=modelpath)
	predictor = SamPredictor(sam)
	predictor.set_image(img)
		
	final = np.zeros((img.shape[0],img.shape[1]))
	
	count = 0
	for i in range(N):
		input_point = np.array([input_mean_points[i]])
		masks, _, _ = predictor.predict(input_point,[1])
		mask = np.array(masks[0])
		mask = np.where(mask == False, 255, mask)
		mask = 255 - mask
		
		mask[mask < 200] = 0
		mask[mask > 1] = 255
		
		acount = np.sum(mask)
		acount = acount / 255
		if acount <= 1.05 * mask_size or mask_size == 0:
			final = final + mask
		if acount > count:
				count = acount
	
	for i in range(n):
		x = input_points[i][1]
		y = input_points[i][0]
		if x >= 0 and y >= 0 and x < final.shape[0] and y < final.shape[1] and final[x][y] == 0:
			input_point = np.array([input_points[i]])
			masks, _, _ = predictor.predict(input_point,[1])
			mask = np.array(masks[0])
			mask = np.where(mask == False, 255, mask)
			
			mask = 255 - mask
			mask[mask < 200] = 0
			mask[mask > 1] = 255
			
			acount = np.sum(mask)
			acount = acount / 255
			if acount > acount:
				count = acount
			if acount <= 1.05 * mask_size or mask_size == 0:
				final = final + mask
				if acount > count:
					count = acount
		elif x >= 0 and y >= 0 and x < final.shape[0] and y < final.shape[1] and final[x][y] > 0:
			pass
		
	if count > mask_size:
		mask_size = count
	
	final[final < 200] = 0
	final[final > 1] = 255
	
	if int(k) < 1000:
		I = '0' + k
	else:
		I = k
	
	cv.imwrite(outpath + f"mask_{I}.png",final)
	final = final.astype('uint8')
	return final

def segment_out_full_nerve(k,MASK):
	path = inpath + f"image_{k}.png"
	img = open_image_and_preprocess(path)
	
	input_points, input_mean_points = find_centroids_and_points_within_segmented_fascicles(MASK,k)
	inputs = np.concatenate((input_points, input_mean_points), axis=0)
	labels = np.ones(inputs.shape[0])
	
	
	sam = sam_model_registry["vit_h"](checkpoint=modelpath)
	predictor = SamPredictor(sam)
	predictor.set_image(img)
	masks, _, _ = predictor.predict(inputs,labels)
	
	cleaned_mask = format_masks(masks)
	cleaned_mask.astype('uint8')
	new_inputs = []
	
	_, cleaned_mask_bi = cv.threshold(cleaned_mask, 200, 255, cv.THRESH_BINARY)	
	heirarchy, contours = cv.findContours(cleaned_mask_bi, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
	length = 0
	
	for h in heirarchy:
		if h.shape[0] > length:
			HEIRARCHY = h
			length = h.shape[0]
	
	for j in range(len(lattice_points)):
		distance = cv.pointPolygonTest(HEIRARCHY, (lattice_points[j][0], lattice_points[j][1]), True)
		if distance >= 80:
			new_inputs.append(lattice_points[j])
	
	new_inputs = np.array(new_inputs)
	labels = np.ones(new_inputs.shape[0])
	
	masks, _, _ = predictor.predict(new_inputs,labels)
	cleaned_mask = format_masks(masks)
	
	path = find_mask_path(k,False)
	cv.imwrite(path,cleaned_mask)
	return cleaned_mask


initial_data_checks()
for m in tqdm(flist):
	path = ms.find_mask_path(m,PATH_TO_DATA)
	if os.path.exists(path):
		groundTruth= cv.imread(path,cv.IMREAD_GRAYSCALE)
	else:
		path = maskpath + f"image_{m}.png"
		if os.path.exists(path):
			groundTruth = cv.imread(path,cv.IMREAD_GRAYSCALE)
		run_segmentation(m)




