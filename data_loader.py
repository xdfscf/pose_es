import os

import numpy
import numpy as np
import tensorflow as tf
from PIL import ImageEnhance, Image
import torchvision.transforms.functional as TF
import random
from typing import Sequence

from utils.load_llff import load_llff_data
class llff_data():
    def __init__(self, datadir, factor=8, spherify=True, llffhold=8, no_ndc=True):
        self.datadir=datadir
        self.factor=factor
        self.spherify=spherify
        self.llffhold=llffhold
        self.no_ndc=no_ndc
        self.images=None
        self.poses=None
        self.bds=None
        self.render_poses=None
        self.i_test=None
        self.hwf=None

    def load_data(self):

        self.images, self.poses, self.bds, self.render_poses, self.i_test = load_llff_data(self.datadir, self.factor,
                                                                          recenter=True, bd_factor=.75,
                                                                          spherify=self.spherify)
        self.hwf = self.poses[0, :3, -1]
        self.poses = self.poses[:, :3, :4]
        print('Loaded llff', self.images.shape,
                      self.render_poses.shape, self.hwf, self.datadir)
        if not isinstance(self.i_test, list):
                self.i_test = [self.i_test]

        if self.llffhold > 0:
                print('Auto LLFF holdout,', self.llffhold)
                self.i_test = np.arange(self.images.shape[0])[::self.llffhold]

        self.i_val = self.i_test
        self.i_train = np.array([i for i in np.arange(int(self.images.shape[0])) if
                                    (i not in self.i_test and i not in self.i_val)])

        print('DEFINING BOUNDS')
        if self.no_ndc:
            self.near = tf.reduce_min(self.bds) * .9
            self.far = tf.reduce_max(self.bds) * 1.
        else:
            self.near = 0.
            self.far = 1.
        print('NEAR FAR', self.near, self.far)

def rotation_angles(matrix, order):
    """
    input
        matrix = 3x3 rotation matrix (numpy array)
        oreder(str) = rotation order of x, y, z : e.g, rotation XZY -- 'xzy'
    output
        theta1, theta2, theta3 = rotation angles in rotation order
    """
    r11, r12, r13 = matrix[0]
    r21, r22, r23 = matrix[1]
    r31, r32, r33 = matrix[2]

    if order == 'xzx':
        theta1 = np.arctan(r31 / r21)
        theta2 = np.arctan(r21 / (r11 * np.cos(theta1)))
        theta3 = np.arctan(-r13 / r12)

    elif order == 'xyx':
        theta1 = np.arctan(-r21 / r31)
        theta2 = np.arctan(-r31 / (r11 *np.cos(theta1)))
        theta3 = np.arctan(r12 / r13)

    elif order == 'yxy':
        theta1 = np.arctan(r12 / r32)
        theta2 = np.arctan(r32 / (r22 *np.cos(theta1)))
        theta3 = np.arctan(-r21 / r23)

    elif order == 'yzy':
        theta1 = np.arctan(-r32 / r12)
        theta2 = np.arctan(-r12 / (r22 *np.cos(theta1)))
        theta3 = np.arctan(r23 / r21)

    elif order == 'zyz':
        theta1 = np.arctan(r23 / r13)
        theta2 = np.arctan(r13 / (r33 *np.cos(theta1)))
        theta3 = np.arctan(-r32 / r31)

    elif order == 'zxz':
        theta1 = np.arctan(-r13 / r23)
        theta2 = np.arctan(-r23 / (r33 *np.cos(theta1)))
        theta3 = np.arctan(r31 / r32)

    elif order == 'xzy':
        theta1 = np.arctan(r32 / r22)
        theta2 = np.arctan(-r12 * np.cos(theta1) / r22)
        theta3 = np.arctan(r13 / r11)

    elif order == 'xyz':
        theta1 = np.arctan(-r23 / r33)
        theta2 = np.arctan(r13 * np.cos(theta1) / r33)
        theta3 = np.arctan(-r12 / r11)

    elif order == 'yxz':
        theta1 = np.arctan(r13 / r33)
        theta2 = np.arctan(-r23 * np.cos(theta1) / r33)
        theta3 = np.arctan(r21 / r22)

    elif order == 'yzx':
        theta1 = np.arctan(-r31 / r11)
        theta2 = np.arctan(r21 * np.cos(theta1) / r11)
        theta3 = np.arctan(-r23 / r22)

    elif order == 'zyx':
        theta1 = np.arctan(r21 / r11)
        theta2 = np.arctan(-r31 * np.cos(theta1) / r11)
        theta3 = np.arctan(r32 / r33)

    elif order == 'zxy':
        theta1 = np.arctan(-r12 / r22)
        theta2 = np.arctan(r32 * np.cos(theta1) / r22)
        theta3 = np.arctan(-r31 / r33)

    theta1 = theta1 * 180 / np.pi
    theta2 = theta2 * 180 / np.pi
    theta3 = theta3 * 180 / np.pi

    return (theta1, theta2, theta3)

def rotation_matrix_from_coordinate(matrix1, matrix2):
    m1_r=np.linalg.inv(matrix1)
    rotation_matrix=np.dot(m1_r,matrix2)
    return rotation_matrix

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    if any(v): #if not all zeros then
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

    else:
        return np.eye(3)

def collect_pose_trasformation(pose1, pose2):
    pose1 = pose1.T
    pose2 = pose2.T
    transformation=[]
    translation=[]
    pose1_axi=pose1[:3]
    pose2_axi=pose2[:3]

    '''
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.use('TkAgg')
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    angle=30
    c1 = np.cos(angle * np.pi / 180)
    s1 = np.sin(angle * np.pi / 180)
    R=numpy.asarray([[c1,-s1, 0],
                     [s1, c1, 0],
                     [0,  0,  1]])

    
    ax.quiver(0, 0, 0, R.dot(pose1_axi[0].T)[0], R.dot(pose1_axi[0].T)[1], R.dot(pose1_axi[0].T)[2], arrow_length_ratio=0.3)
    ax.quiver(0, 0, 0, R.dot(pose1_axi[1].T)[0], R.dot(pose1_axi[1].T)[1], R.dot(pose1_axi[1].T)[2], arrow_length_ratio=0.3)
   
    
    ax.quiver(0, 0, 0, pose1_axi[0][0], pose1_axi[0][1], pose1_axi[0][2], arrow_length_ratio=0.3)
    ax.quiver(0, 0, 0, pose1_axi[1][0], pose1_axi[1][1], pose1_axi[1][2], arrow_length_ratio=0.3)
   
    ax.quiver(0, 0, 0, pose1_axi[2][0], pose1_axi[2][1], pose1_axi[2][2], arrow_length_ratio=0.3)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_zlim(0, 5)

    plt.show()
   
    for i, j in zip(pose1_axi, pose2_axi):
        rotates=rotation_matrix_from_vectors(i,j)
        transformation.append(rotates)
    '''

    transformation.append(rotation_matrix_from_coordinate(pose1_axi,pose2_axi))
    translation.append(pose2[-1]-pose1[-1])
    return transformation, translation

def data_transform(pose_axi, image_name):
    pose_axi=pose_axi.T
    angles = [-30, -15, 15, 30]
    transform_poses=[]
    new_names=[]
    for angle in angles:

        c1 = np.cos(angle * np.pi / 180)
        s1 = np.sin(angle * np.pi / 180)
        R = numpy.asarray([[c1, -s1, 0],
                           [s1, c1, 0],
                           [0, 0, 1]])
        transform_poses.append(R)
        image = Image.open(image_name)
        image = TF.rotate(image, -angle)
        new_name="./transform_image/"+str(angle)+"_"+image_name.split('\\')[-1]
        print(image_name)
        print(new_name)
        new_names.append(new_name)
        image.save(new_name)
    translate=np.zeros((4, 3))
    return transform_poses, translate, new_names

def normalize_translation(translation):
    translation=np.asarray(translation)
    max=np.max(translation)
    min=np.min(translation)
    if abs(min)>max:
        max=abs(min)
    translation=np.divide( translation, max)
    return translation

def normalize_rotation(transformation):
    transformation=np.asarray(transformation)
    print(np.max(transformation))
    print(np.min(transformation))
    return transformation

def prepare_data(basedir):
    basedir="./nerf_llff_data/fern"
    datas=llff_data(basedir, factor=8)
    datas.load_data()
    poses=datas.poses
    images=datas.images


    image_names= [os.path.join(basedir, 'images_8', f) for f in sorted(os.listdir(os.path.join(basedir, 'images_8'))) \
                if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]

    dataset=[]

    transformation=[]
    translation=[]
    for i in range(len(poses)):
        for j in range(i, len(poses)):
            dataset.append([image_names[i],image_names[j]])
            t1,t2=collect_pose_trasformation(poses[i], poses[j])
            transformation.append(t1)
            translation.append(t2)

    for i in range(len(poses)):
        t1s, t2s, names=data_transform(poses[i], image_names[i])
        for name, t1, t2 in zip(names, t1s, t2s):
            dataset.append([image_names[i], name])
            t1=[np.asarray(t1)]
            transformation.append(t1)
            t2=[np.asarray(t2)]
            translation.append(t2)

    translation=normalize_translation(translation)
    transformation=normalize_rotation(transformation)
    return dataset, translation, transformation






