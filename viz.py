import os
import sys
import pickle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import numpy as np

def visualise3D(verts,assignment,angle=0,view=0,figsize=20):
# Takes as input a list of coords, where each coord is a list that goes [x,y,z].
    if type(verts[0]) == str:
        verts = [[x.split(' ')[1],x.split(' ')[2],x.split(' ')[3].split('\n')[0]] for x in verts]
        xs = [float(x[0]) for x in verts]
        ys = [float(x[1]) for x in verts]
        zs = [float(x[2]) for x in verts]
    elif type(verts)==list:
        xs = [x[0] for x in verts]
        ys = [x[1] for x in verts]
        zs = [x[2] for x in verts]
    else:
        xs = verts[:,0]
        ys = verts[:,1]
        zs = verts[:,2]

    fig = plt.figure(figsize=(figsize,figsize))
    ax = fig.add_subplot(111, projection='3d')
    #angle+=45
    ax.view_init(view, angle)

    ax.scatter(xs, ys, zs, c=assignment, marker='+')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # plt.show()
    plt.savefig('clusters.png')

def read_mesh_input(datadir="data", dataname="mesh"):
    train_file = os.path.join(datadir, dataname, 'Ytrn.npz')
    '''this edges file is constant and therefore it can be used to create
    a graph structure once. Make copies of this graph object and set the
    node features as the point coordinates for different input graphs.
    '''
    train_np_file = np.load(train_file)
    train_points  = train_np_file[train_np_file.files[0]][:10,:,:]
    shape = train_points.shape
    print(shape)
    num_points = shape[0] * shape[1]
    assignment = np.random.randint(2, size=num_points)
    visualise3D(train_points.reshape(num_points, shape[2]), assignment)
    # print(train_points[:,:,])

def plot_assignments(fpath):
    f = open(fpath, 'rb')
    x, assignment = pickle.load(f)
    f.close()

    shape = x.shape
    print(shape)
    print(assignment.shape)
    num_points = shape[0] * shape[1]
    visualise3D(x.reshape(num_points, shape[2]), assignment.reshape(num_points))

if __name__ == "__main__":
    # read_mesh_input()
    plot_assignments(sys.argv[1])
