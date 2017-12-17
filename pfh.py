import numpy as np
import scipy.spatial.distance as ssd

import geom


def collect_neighbors(points, r_neigh, n=6):
	'''
	collect neighborhood for each point, and calculate surface normal
	
	for each points return neighbors that contain at least n points
	'''

	neighbors = list()
	normals = list()

	D = ssd.cdist(points.T, points.T, 'euclidean')
	for i in range(points.shape[1]):
		r = r_neigh
		idx = np.argwhere(D[i,:] < r)[:, 0]
		while len(idx) < n:
			r *= 2
			idx = np.argwhere(D[i,:] < r)[:,0]

		p_n = list()
		for j in idx:
			if j == i:
				continue
			p_n.append(points[:,j])
		p_n.append(points[:,i]) # append query point to the end

		p_ns = get_normal(p_n)
		neighbors.append(p_n)
		normals.append(p_ns)

	return neighbors, normals

def get_normal(p):
	'''
	given some points p, return surface normal of those points
	'''
	n = len(p)
	P = np.concatenate(p, axis=1)
	mp = np.mean(P, axis=1)
	P = (P - mp) / np.sqrt(n-1)

	U,S,V = np.linalg.svd(P)
	i = np.argmin(S)

	return U[:,i]


def get_p2feature(pn, ns):
	'''
	compute angular between each pair of points inside a neighborhood
	
	pn: points in neighbor
	ns: surface normal for each points

	'''

	features = list()

	n = len(pn)
	# when compute feature, only uses points in neighbor(exclude the query point)
	for i in range(n-1):
		for j in range(i+1, n-1):
			p1 = pn[i]
			p2 = pn[j]
			ns1 = ns[i]
			ns2 = ns[j]

			# distance
			d_vec = p2 - p1
			d = np.linalg.norm(d_vec)

			# src frame
			u = ns1
			v = np.cross(u, d_vec/d, axis=0)
			v = v/np.linalg.norm(v)
			w = np.cross(u, v, axis=0)
			w = w/np.linalg.norm(w)

			assert(np.abs(np.linalg.norm(v) - 1) < 1e-6)
			assert(np.abs(np.linalg.norm(w) - 1) < 1e-6)

			# angular feature
			a = np.arccos(np.dot(v.T, ns2))
			phi = np.arccos(np.dot(u.T, d_vec/d))
			theta = np.arctan(np.dot(w.T,ns2), np.dot(u.T,ns2))

			features.append([a, phi, theta, d])

	return features


def get_feature(points, bmin, bmax, nbins, r_neigh, a, b):
	'''
	TODO: implement in multi-thread way

	return: 
		h: pfh feature
		idx: corresponding index
	'''

	histograms =list() 

	p_neighbors, p_normals = collect_neighbors(points, r_neigh)
	idx = geom.extract_keypoint(p_neighbors, points, a, b)


	for i in idx:
		pn = p_neighbors[i]

		features = get_p2feature(pn, p_normals)
		h = get_histogram(features, bmin, bmax, nbins)
		
		histograms.append(h)

	return histograms, idx


def get_histogram(features, bmin, bmax, nbins):
	'''
	from raw feature to histogram

	features: raw feature -> list([alpha, phi, theta], ...)
	bmin: min value for each dimensions
	bmax: max values
	nbins: number of bins per dimension
	'''

	nf = 3;

	h = np.zeros((1, nbins**nf))
	binsteps = [(bmax[i]-bmin[i])/(nbins-1) for i in range(nf)]
	
	# rounding 
	for f in features:
		bins = list()
		for i in range(nf):
			# rounding
			if f[i] < bmin[i]:
				b = 0
			elif f[i] > bmax[i]:
				b = nbins - 1
			else:
				b = int((f[i] - bmin[i]) / binsteps[i])
			bins.append(b)

		# binidx = nbins*(nbins*(nbins*bins[0]+bins[1])+bins[2]) + bins[3]
		binidx = nbins*(nbins*bins[0]+bins[1])+bins[2]
		h[0, binidx] += 1

	# # bilinear interpolation
	# for f in features:
	# 	binidices = list()
	# 	for i in range(4):
	# 		if f[i] < bmin:
	# 			b = 0
	# 			if len(binidices) == 0:
	# 				binidices.append([b, 1])
	# 			else:
	# 				binidices = [binidices[i][0]*nbins + b for i in range(len(binidices))]
	# 		elif f[i] > bmax:
	# 			b = nbins - 1
	# 			if len(binidices) == 0:
	# 				binidices.append([b, 1])
	# 			else:
	# 				binidices = [binidices[i][0]*nbins + b for i in range(len(binidices))]
	# 		elif:
	# 			b = int((f[i] - bmin[i]) / binsteps[i])
	# 			s1 = (f[i] - b * binsteps[i] + bmin[i]) * 1.0 / binsteps[i]
	# 			s2 = ((b+1) * binsteps[i] + bmin[i] - f[i]) * 1.0 / binsteps[i]

	# 			if len(binidices) == 0:
	# 				binidices.append([b, s2])
	# 				binidices.append([b+1, s1])
	# 			else:
	# 				tmp1 = list()
	# 				tmp2 = list()
	# 				for ii in range(len(binidices)):
	# 					tmp1.append([binidices[ii][0]*nbins+b, binidices[ii][1]*s2])
	# 					tmp2.append([binidices[ii][1]*nbins+b+1, binidices[ii][1]*s1])
	# 				binidices = [tmp1, tmp2]

	# 	for binidx in binidices:
	# 		h[binidx[0]] += binidx[1]
	h = h * 1.0 / np.sum(h)
	return h 


