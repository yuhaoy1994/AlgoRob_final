import utils
import numpy as np
import matplotlib.pyplot as plt

from pfh import *
from icp import *
from fit import *
from geom import *

p_src = utils.load_pc('template_alignment/object_template_0.csv')
p_tgt = utils.load_pc('template_alignment/object_template_2.csv')

r_neigh = 0.01
nbin = 5
pi = np.pi
bmin = [0 + pi/2/nbin,
        0 + pi/2/nbin,
        -pi/2 + pi/2/nbin,
        0 + r_neigh/2/nbin]

bmax = [pi - pi/2/nbin,
		pi - pi/2/nbin,
		pi/2 - pi/2/nbin,
		r_neigh - r_neigh/2/nbin]


T, ptrans, key_idx = icp_pfh(p_src, p_tgt, bmin, bmax, nbin, r_neigh=r_neigh)

print T

fig = utils.view_pc([ptrans], linewidths=3, marker='^')
# utils.view_pc([p_src], fig, color='g')
utils.view_pc([p_tgt], fig, color='r')



# fig1 = utils.view_pc([[p_src[i] for i in key_idx]], linewidths=3, marker='^', color='r')
# utils.view_pc([p_src], fig1, color='g')

raw_input(" ")