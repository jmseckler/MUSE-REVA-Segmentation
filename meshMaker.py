import trimesh
import sys, glob, zarr, os
import numpy as np
import methods as ms
from skimage import measure
from tqdm import tqdm
import cv2 as cv
import dask.array as da
import vtkplotlib as vpl
from stl.mesh import Mesh

inPath = "/media/james/T9/segmentation/"
outPath = "/media/james/T9/mesh/"
fname = sys.argv[1]

voxelSize = [3,1,1]
scaleFactor = 0.2
splitNerve = 2000

dataPath = inPath + fname + '/'
savePath = outPath + fname + '/'

indvMeshPath = savePath + 'tmp/'
ms.make_directory(savePath)
ms.replace_directory(indvMeshPath)

def load_in_images_and_produce_zarrs(path):
	flist = ms.find_file_list(path,0)
	z = int(flist[-1]) + 1
	zimg = ms.create_basic_zarr_file(savePath,"binary")
	img = cv.imread(path + 'mask_' + flist[0] + '.png',cv.IMREAD_GRAYSCALE)
	new_size = (int(img.shape[1] * scaleFactor), int(img.shape[0] * scaleFactor))
	img = cv.resize(img, new_size, interpolation = cv.INTER_AREA)
	
	x = img.shape[0]
	y = img.shape[1]
	
	zshape, zchunk = ms.shape_definer(z,x,y,1)
	full = zimg.zeros('0', shape=zshape, chunks=zchunk, dtype="i2" )
	
	binary = load_image_and_return_binary(path + 'mask_' + flist[0] + '.png',new_size)
	for i in tqdm(range(z)):
		imgPath = ms.find_mask_path(i,path,True)
		if os.path.isfile(imgPath):
			binary = load_image_and_return_binary(imgPath,new_size)
		full[i] = binary
	return full

def display_filelist(path):
	flist = ms.find_file_list(path,0)
	print(flist)


def load_image_and_return_binary_(path,new_size):
	img = cv.imread(path,cv.IMREAD_GRAYSCALE)
	img = cv.resize(img, new_size, interpolation = cv.INTER_AREA)
	
	mask = np.zeros(img.shape)
	heirarchy, contours = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
	for h in heirarchy:
		for coord in h:
			mask[coord[0][1]][coord[0][0]] = 255
	binary_array = np.where(mask > 0, 1, 0)
	binary_array = np.array(binary_array)
	return binary_array

def load_image_and_return_binary(path,new_size):
	img = cv.imread(path,cv.IMREAD_GRAYSCALE)
	img = cv.resize(img, new_size, interpolation = cv.INTER_AREA)
	
	binary_array = np.where(img > 0, 1, 0)
	binary_array = np.array(binary_array)
	return binary_array

def split_up_zarr_array_into_bitesized_chunks(zArray):
	z = zArray.shape[0] - 1
	n = int(z/splitNerve) + 1
	for i in tqdm(range(n)):
		arr = np.array(zArray[splitNerve*i:splitNerve*(i+1)])
		mesh = arr_to_3d_mesh(arr)
		mesh.export(indvMeshPath + f'mesh_{i}.stl')
	return n
		

def arr_to_3d_mesh(array):
	verts, faces, normals, values = measure.marching_cubes(array, level=0.5)
	mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
	mesh.apply_scale(voxelSize)
	return mesh


def combine_all_meshes(n):
	mesh = None
	for i in tqdm(range(n)):
		path = indvMeshPath + f'mesh_{i}.stl'
		mesh = merge_stl_files(mesh, path)
	mesh.save(savePath + 'fascicles.stl')
	return mesh
	

def merge_stl_files(baseMesh, file2):
	if baseMesh is None:
		baseMesh =  Mesh.from_file(file2)
		return baseMesh
	mesh2 = Mesh.from_file(file2)
	combined_data = np.concatenate([baseMesh.data, mesh2.data])
	combined_mesh = Mesh(combined_data)
	return combined_mesh

binaryPath = savePath + 'binary.zarr'

if os.path.isdir(binaryPath):
	zAll = zarr.open(binaryPath, mode='r')
	zArray = zAll['data']['0']
else:
	zArray = load_in_images_and_produce_zarrs(dataPath)

chunks = split_up_zarr_array_into_bitesized_chunks(zArray)
mesh = combine_all_meshes(chunks)

#mesh = Mesh.from_file('./output_mesh.stl')

vpl.mesh_plot(mesh)
vpl.show()























