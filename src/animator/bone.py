import numpy as np

def deg2rad(deg):
    return deg/180*np.pi

# Transformation between two coordinate system
def t(x, y, theta):
    return np.array([[np.cos(theta),-np.sin(theta),x],
                     [np.sin(theta),np.cos(theta),y],
                     [0,0,1]])

# Transformation from a line/bone
# theta wrt +x axis, l length of line
def t_line(l, theta):
    return t(l*np.cos(theta), l*np.sin(theta), theta)

# Convert list of params to list of frames
# Eech param describes the bones pose
# Each frame has all the bones coordinates
# params2bones is the function that map a single frame parameters to bones coordinates
def params2bones_with_params2bones(params, params2bones):
    bones_frames = []

    for p in params:
        bones_frames.append(params2bones(p))

    return bones_frames
