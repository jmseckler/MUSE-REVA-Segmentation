from segment_anything import SamPredictor, sam_model_registry
import cv2 as cv
import glob, os
import numpy as np
from tqdm import tqdm


inpath = './data/img/'
maskpath = './data/mask/'
modelpath = './models/sam_vit_h_4b8939.pth'
outpath = './data/segment/'
debugPath = './data/dots/'

contrastFactor = 1.0

elipse_size = 30
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(elipse_size,elipse_size))

lattice_point_spacing = 50

mask_size = 0


def find_file_list(masks=False):
	if masks:
		rawlist = glob.glob(maskpath + '*.png')
	else:
		rawlist = glob.glob(inpath + '*.png')
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
	
	

def find_centroids_of_segmented_fascicles(mask):
	input_points = []
	heirarchy, contours = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
	
	means = []
	stds = []
	astds = []
	angle = np.sqrt(2) / 2
	
	
	for h in heirarchy:
		if h.shape[0] > 50:
			means.append(np.mean(h,axis=0)[0].astype('int'))
			stds.append(np.std(0.7 * h,axis=0)[0].astype('int'))
			tmp = 0.7 * angle * np.std(h,axis=0)[0]
			astds.append(tmp.astype('int'))

	n = len(means)
	for i in range(n):
		mean = means[i]
		std = stds[i]
		astd = astds[i]
		input_points.append(np.array([mean[0] + std[0],mean[1]]))
		input_points.append(np.array([mean[0] - std[0],mean[1]]))
		input_points.append(np.array([mean[0],mean[1] + std[1]]))
		input_points.append(np.array([mean[0],mean[1] - std[1]]))
		input_points.append(np.array([mean[0] + astd[0],mean[1] + astd[1]]))
		input_points.append(np.array([mean[0] - astd[0],mean[1] - astd[1]]))
		input_points.append(np.array([mean[0] + astd[0],mean[1] - astd[1]]))
		input_points.append(np.array([mean[0] - astd[0],mean[1] + astd[1]]))
		
	input_points = np.array(input_points)
	means = np.array(means)
	return input_points, means

def find_centroids_and_points_within_segmented_fascicles(MASK,name):
	rawmask = np.zeros(MASK[0].shape)
	
	n = len(MASK)
	for i in range(n):
		rawmask = rawmask + MASK[i]
	rawmask = rawmask / n
	rawmask = rawmask.astype('uint8')
	
	_, mask = cv.threshold(rawmask, 127, 255, cv.THRESH_BINARY)
	
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
				if distance >= 50:
#					print(distance)
					input_points.append(lattice_points[j])
			i += 1
	
#	debug(mask,input_points,name)
	input_points = np.array(input_points)
	means = np.array(means)
	return input_points, means

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


	
def open_image_and_preprocess(path):
	img = cv.imread(path,cv.IMREAD_GRAYSCALE)
	mean = np.mean(img).astype(int)
	img = img - mean
	img = contrastFactor * img
	img = img + 140
	
	img = np.clip(img,0,255)
	topHat = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)
	blackHat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)
	img = img + topHat - blackHat
	
	img = np.clip(img,0,255)
	img = img.astype('uint8')
	img = cv.cvtColor(img,cv.COLOR_GRAY2RGB)
	
	return img

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
#	print(img.shape)
#	print(input_points)
	
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
#		else:
#			print(count,input_point)
	
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
#			else:
#				print(count,input_point)
		elif x >= 0 and y >= 0 and x < final.shape[0] and y < final.shape[1] and final[x][y] > 0:
			pass
		
	if count > mask_size:
		mask_size = count

	final[final < 200] = 0
	final[final > 1] = 255

	
#	print(count,mask_size)
	if int(k) < 1000:
		I = '0' + k
	else:
		I = k
	
	cv.imwrite(outpath + f"mask_{I}.png",final)
	final = final.astype('uint8')
	return final

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
	
	

flist = find_file_list()
mlist = find_file_list(True)

path = maskpath + 'image_' + mlist[0] + '.png'
groundTruth = [cv.imread(path,cv.IMREAD_GRAYSCALE)]

lattice_points = calculate_lattice_points(groundTruth[0])


for m in tqdm(flist):
	addon = ''
	if int(m) < 1000:
		addon = '0'
	path = outpath + f"mask_{addon + m}.png"
	if os.path.exists(path):
		groundTruth= [cv.imread(path,cv.IMREAD_GRAYSCALE)]
	else:
		path = maskpath + f"image_{m}.png"
		if os.path.exists(path):
			groundTruth.append(cv.imread(path,cv.IMREAD_GRAYSCALE))
		groundTruth.append(segment_out_fasciles(m,groundTruth))
	
	while len(groundTruth) > 3:
		groundTruth.pop(0)
	

#put in the kuda to speed this up 

