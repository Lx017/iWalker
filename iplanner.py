# ======================================================================
# Copyright (c) 2023 Fan Yang
# Robotic Systems Lab, ETH Zurich
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ======================================================================

import sys
import numpy as np
import argparse
import torch
import rospy
import torchvision.transforms as transforms
from cv_bridge import CvBridge
import cv2
bridge = CvBridge()
from sensor_msgs.msg import Image
import datetime
import pypose as pp
from Hlip import *
planner_path = "iplanner"
sys.path.append(planner_path)

from snp import SharedNp
import os 
from lxutil import *
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import PointStamped
import std_msgs
import tf.transformations as tft
from sensor_msgs.msg import Image, PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
MAX_DEPTH = 10
TRAINING = False
RECORDING = False
LR = 1e-6
IFSIM = True
recording_dir = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
if RECORDING:
    os.makedirs("recordings/"+recording_dir)
START_RECORDING = False
FPS = 10
N_STEP = 150 #20
tar_step_len = 0.01
step_width = 0.02
node_name = "iplanner_node"
rospy.init_node(node_name, anonymous=False)
markerPub = rospy.Publisher('/iplanner_out', MarkerArray)
collisionMapPub = rospy.Publisher('/collision_map', Image)
GLOBAL_TARGET_POINT = None
listener = tf.TransformListener()
SN = SharedNp("shm1.json")
def Targetcallback(data):
    global GLOBAL_TARGET_POINT,START_RECORDING
    START_RECORDING = True
    GLOBAL_TARGET_POINT = data
    # X = GLOBAL_TARGET_POINT.point.x
    # Y = GLOBAL_TARGET_POINT.point.y
    # GLOBAL_TARGET_POINT.point.x = -Y
    # GLOBAL_TARGET_POINT.point.y = X
    print("clicked point at ", GLOBAL_TARGET_POINT.point.x, GLOBAL_TARGET_POINT.point.y, GLOBAL_TARGET_POINT.point.z)
cps = rospy.Subscriber("/clicked_point", PointStamped, Targetcallback)
import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class CubicSplineTorch:
    # Reference: https://stackoverflow.com/questions/61616810/how-to-do-cubic-spline-interpolation-and-integration-in-pytorch
    def __init__(self):
        return None

    def h_poly(self, t):
        alpha = torch.arange(4, device=t.device, dtype=t.dtype)
        tt = t[:, None, :]**alpha[None, :, None]
        A = torch.tensor([
            [1, 0, -3, 2],
            [0, 1, -2, 1],
            [0, 0, 3, -2],
            [0, 0, -1, 1]
            ], dtype=t.dtype, device=t.device)
        return A @ tt

    def interp(self, x, y, xs):
        m = (y[:, 1:, :] - y[:, :-1, :]) / torch.unsqueeze(x[:, 1:] - x[:, :-1], 2)
        m = torch.cat([m[:, None, 0], (m[:, 1:] + m[:, :-1]) / 2, m[:, None, -1]], 1)
        idxs = torch.searchsorted(x[0, 1:], xs[0, :])
        dx = x[:, idxs + 1] - x[:, idxs]
        hh = self.h_poly((xs - x[:, idxs]) / dx)
        hh = torch.transpose(hh, 1, 2)
        out = hh[:, :, 0:1] * y[:, idxs, :]
        out = out + hh[:, :, 1:2] * m[:, idxs] * dx[:,:,None]
        out = out + hh[:, :, 2:3] * y[:, idxs + 1, :]
        out = out + hh[:, :, 3:4] * m[:, idxs + 1] * dx[:,:,None]
        return out
    
cs_interp = CubicSplineTorch()
def CostofTraj(waypoints, goal, gamma=2.0, delta=5.0):
    goal = goal.cuda()
    batch_size, num_p, _ = waypoints.shape

    # Goal Cost
    gloss = torch.norm(goal[:, :3] - waypoints[:, -1, :], dim=1)
    gloss = torch.mean(torch.log(gloss + 1.0))
    # gloss = torch.mean(gloss)
    
    # Motion Loss
    step=1.0/(num_p-1)
    preds = waypoints#goal[:, None, 0:3]

    points_preds = torch.cat((torch.zeros(batch_size, 1, 3, device=preds.device, requires_grad=preds.requires_grad), preds), axis=1)
    num_p = num_p + 1
    xs = torch.arange(0, num_p-1+step, step, device=preds.device)
    xs = xs.repeat(batch_size, 1)
    x  = torch.arange(num_p, device=preds.device, dtype=preds.dtype)
    x  = x.repeat(batch_size, 1)
    desired_wp = cs_interp.interp(x, points_preds, xs)
    
    desired_ds = torch.norm(desired_wp[:, 1:num_p-1, :] - desired_wp[:, 0:num_p-2, :], dim=2)
    wp_ds = torch.norm(waypoints[:, 1:, :] - waypoints[:, :-1, :], dim=2)
    mloss = torch.abs(desired_ds - wp_ds)
    mloss = torch.sum(mloss, axis=1)
    mloss = torch.mean(mloss)
    
    return gamma*mloss + delta*gloss

class iStepper(nn.Module):
    def __init__(self, sample_N):
        super(iStepper, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Linear(2*sample_N, 256)#input [xy,dirxy,perpxy]
        self.conv2 = nn.Linear(256, 512)
        self.fc1 = nn.Linear(512, 2*sample_N)

    def forward(self, x, ifLeft = 0):
        N = x.shape[1]
        centers = x[:,:,0:2]
        dir_vecs = x[:,:,2:4]
        perp_vecs = x[:,:,4:6]

        x = self.relu(self.conv1(centers.flatten(1)))
        x = self.relu(self.conv2(x))
        xy = self.fc1(x).reshape(-1, N, 2)
        dx = xy[:,:,0:1]
        dy = xy[:,:,1:2]

        signs = torch.arange(N) + ifLeft
        signs = (signs % 2) * 2 - 1
        signs = signs[None, :, None].float().cuda()
        out = centers + dx*dir_vecs + dy*perp_vecs*signs
        return out,dx,dy

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
class PerceptNet(nn.Module):

    def __init__(self, layers, block=BasicBlock, groups=1, 
                 width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
                 
        super(PerceptNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)
    
class iPlannerNode:
    def __init__(self, args):
        super(iPlannerNode, self).__init__()

        self.iplanner_algo = IPlannerAlgo(args)

        self.is_goal_init = False
        self.ready_for_planning = False
        self.is_goal_processed = False
        self.is_smartjoy = False

        # fear reaction
        self.fear_buffter = 0
        self.is_fear_reaction = False
        self.iStepper = iStepper(N_STEP).cuda()
        self.iStepper.load_state_dict(torch.load("step.pth"))
        self.stepNNoptimizer = torch.optim.Adam(self.iStepper.parameters(), lr=1e-3)
        self.stepMSE = nn.MSELoss()
        if TRAINING:
            loggingfile = open("log/"+datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")+".txt", "w")
            import time
            from unicycle import Unicycle_MPC
            path_optimizer = torch.optim.Adam(self.iplanner_algo.net.parameters(), lr=LR)#1e-7
            criterion = nn.MSELoss()
            rec_path = 'recordings'
            self.iplanner_algo.net.train()
            depth_viz = rospy.Publisher("/camera/depth/image_rect_raw", Image, queue_size=1)
            EPOCH = 30

            height = 0.8
            TSSP = 0.24
            TDSP = 0.0
            vdes = 0.2
            step_size = 0.3

            Ax, Bx, _ = global_hlip_initialization(height, TSSP, TDSP)
            print("Matrix A:\n", Ax)
            print("Matrix B:\n", Bx)
            Ay, By, _ = global_hlip_initialization(height, TSSP, TDSP)

            A = np.block([[Ax, np.zeros((3,3))],[np.zeros((3,3)), Ay]])
            B = np.block([[Bx, np.zeros((3,1))],[np.zeros((3,1)), By]])

            N = N_STEP

            MPC = OSQPMPC(nx=6,nu=2,N=N,polish=True)

            F = np.concatenate([A, B], axis=1)
            C = np.array([[0, 0, 0, 0, 0, 0,1,0],
                        [0, 0, 0, 0, 0, 0,0,1]])

            MPC.init_dyn(F, C)
            MPC.set_lower(np.array([-2,-2]))
            MPC.set_upper(np.array([2,2]))

            step_cost = 3
            state_cost = np.array([0,0,0,0,0,0,step_cost,step_cost])
            M = np.array([[1,-1,0,0,0,0,0,0],
                        [0,0,0,1,-1,0,0,0]])
            tar_cost = np.array([10,10])
            MPC.set_cost(state_cost=state_cost, M=M, tar_cost=tar_cost)

            UN = 100
            UMPC = Unicycle_MPC(UN)

            #K = np.array([426.821533203125, 0.0, 429.42242431640625, 0.0, 426.821533203125, 243.98818969726562, 0.0, 0.0, 1.0]).reshape(3,3)
            K_inv = 1/80 #80 is 160/2, 160 is recorded depth image width
            #K_inv = 0.04444#for sim

            point_cloud_pub = rospy.Publisher('/ptsc', PointCloud2, queue_size=10)
            ROT1 = np.array([[0, 0, 1], 
                            [0, 1, 0], 
                            [-1, 0, 0]])
            ROT2 = np.array([[1,0,0],
                            [0,0,1],
                            [0,-1,0]])
            ROT = ROT2@ROT1

            D_THRESH = 10#5 meters
            W_THRESH = 10

            def point2map(map,pts,color,ifAdd=False):
                # pts = pts[pts[:,0]<D_THRESH]
                # pts = pts[pts[:,1]<W_THRESH/2]
                # pts = pts[pts[:,1]>-W_THRESH/2]
                points_idx = ((pts[:,:2]/np.array([D_THRESH,W_THRESH])*np.array([map.shape[0],map.shape[1]]))).astype(int)
                points_idx[:,1] = 50 - points_idx[:,1]
                points_idx[:,0] = 99 - points_idx[:,0]
                points_idx[:,0] = np.clip(points_idx[:,0],1,map.shape[0]-2)
                points_idx[:,1] = np.clip(points_idx[:,1],1,map.shape[1]-2)
                # points_idx = points_idx[points_idx[:,0]<map.shape[0]]
                # points_idx = points_idx[points_idx[:,1]<map.shape[1]]
                # points_idx = points_idx[points_idx[:,0]>0]
                # points_idx = points_idx[points_idx[:,1]>0]
                if ifAdd:
                    np.add.at(map, (points_idx[:,0], points_idx[:,1]), color)
                    #map[points_idx[:,0],points_idx[:,1]] += color
                # else:
                #     map[points_idx[:,0],points_idx[:,1]] = color
                return points_idx


            TEST_DIR = ["test1","test2","2","2ndStair-uneven","2ndLeft2"]
            ONLY_DIR = ["sim1"]
            files = ((dir, file) for dir in os.listdir(rec_path) for file in os.listdir(os.path.join(rec_path, dir)))
            train_files = []
            for f in files:
                dir = f[0]
                if len(ONLY_DIR) > 0 and dir not in ONLY_DIR:
                    continue
                # if dir in TEST_DIR:
                #     continue
                file_name = f[1]
                data = torch.load(os.path.join(rec_path, dir, file_name))
                train_files.append({"dir":dir, "file":file_name, "data":data})
            random.shuffle(train_files)
            #TEST_DIR = []
            DEBUG = False
            TESTING = False
            ifFlip = False
            for epoch in range(EPOCH):
                records = {}
                for dir in os.listdir(rec_path):
                    records[dir] = {"path_losses":[], "step_losses":[], "col_risks":[], "smoothnesses":[], "dis_vars":[]}


                for file in train_files:
                    # if not file.endswith('.pt'):
                    #     continue
                    # if random.random() > 0.5:
                    #     continue
                    dir = file["dir"]
                    data = file["data"]
                    ifTesting = dir in TEST_DIR
                    #ifTesting = ifTesting or epoch == 0 #test first epoch
                    # if TESTING and dir not in TEST_DIR:
                    #     continue
                    # if not TESTING and dir in TEST_DIR:
                    #     continue
                    #data = torch.load(os.path.join(rec_path, dir, file))
                    preds = data['preds'].detach()
                    goal = data['goal'].detach()
                    if goal[0,0] < 0:
                        goal[0,0]*=-1
                    depth = data['input']
                    depth[depth==0.8] = 0
                    if not ifTesting:
                        if ifFlip:
                            preds[:,:,1] *= -1
                            goal[:,1] *= -1
                            depth = np.flip(depth, axis=1).copy()
                        # random_scale = random.random()*1 + 0.5
                        # preds = preds * random_scale
                        # goal = goal * random_scale
                        # depth = depth * random_scale
                    else:
                        print("Testing on ", dir)
                    ifFlip = not ifFlip
                    depth_viz.publish(bridge.cv2_to_imgmsg(depth, "passthrough"))
                    preds_1, _, __, img_process = self.iplanner_algo.plan(depth, goal)
                    preds_1[:,:,2]*=0
                    #cv2.imshow("process", img_process.detach().cpu().numpy()[0].transpose(1,2,0))

                    
                    mks = MarkerArray()
                    pred_pts_np = preds_1[0].detach().cpu().numpy()
                    pred_pts_np = np.vstack([np.zeros(3),pred_pts_np, goal])


                    mk = generate_3DPointMarker(5, preds[0],color=[0, 1, 1,1],scale = 0.1)
                    mk.header.frame_id = "camera"
                    mks.markers.append(mk)

                    #pts_torch = torch.cat([torch.zeros(1,3).cuda(), preds_1[0], goal.cuda()], dim=0)
                    pts_torch = torch.cat([torch.zeros(1,3).cuda(), preds_1[0,:-1], goal.cuda()], dim=0)

                    dis = (pts_torch[1:]-pts_torch[:-1])[:-1,0:2].detach().cpu().numpy()
                    dis = np.linalg.norm(dis,axis=1)
                    dis/=dis.mean()
                    dis_var = np.var(dis)

                    way_points = pp.chspline(pts_torch, 0.003)
                    vecs = way_points[1:] - way_points[:-1]
                    lens = torch.linalg.norm(vecs,axis=1)
                    total_len = lens.sum()
                    interval = (total_len / UN)*0.99
                    intp_sample_torch= [way_points[0]]
                    for p in way_points[1:]:
                        vec = p - intp_sample_torch[-1]
                        dis = torch.linalg.norm(vec)
                        if dis > interval:
                            intp_sample_torch.append(p)
                    while len(intp_sample_torch) < UN:
                        intp_sample_torch.append(way_points[-1])
                    intp_torch = pp.chspline(pts_torch, 5/UN)[:UN]
                    intp_sample_torch = torch.stack(intp_sample_torch)
                    intp_sample_torch = intp_sample_torch[:UN]
                    pts_torch_prev = torch.cat([torch.zeros(1,3).cuda(), preds[0,:-1], goal.cuda()], dim=0)
                    intp_torch_prev = pp.chspline(pts_torch_prev, 5/UN)[:UN]
                    U_tar_traj = np.array(intp_sample_torch.detach().cpu())[:,0:2]
                    smoothness = []
                    last_vec = U_tar_traj[1] - U_tar_traj[0]
                    last_vec/=np.linalg.norm(last_vec)
                    for i in range(1,len(U_tar_traj)):
                        vec = U_tar_traj[i] - U_tar_traj[i-1]
                        vec = vec / np.linalg.norm(vec)
                        sim = np.dot(vec,last_vec)
                        if not np.isnan(sim):
                            smoothness.append(np.dot(vec,last_vec))
                        last_vec = vec
                    smoothness = np.array(smoothness).var()
                    records[dir]["dis_vars"].append(dis_var)
                    records[dir]["smoothnesses"].append(smoothness)
                    int_dis = U_tar_traj[:-1] - U_tar_traj[1:]
                    int_dis = np.linalg.norm(int_dis,axis=1)

                    PC = True

                    TURN_COST = 100
                    ACC_COST = 5000
                    COM_x, COM_y, path_MPC_L = UMPC.solve(U_tar_traj,0,TURN_COST,ACC_COST,int_dis.mean(), iter=100)
                    U_pts = np.vstack([COM_x,COM_y,np.zeros(UN+1)]).T
                    gt = torch.tensor(U_pts).cuda().float()

                    path_MPC_loss = criterion(intp_torch, gt[:UN])
                    goal_loss = criterion(preds_1[-1], goal.cuda())
                    path_optimizer.zero_grad()

                    if PC and TRAINING:
                        depth = depth
                        height, width = depth.shape
                        # Create a grid of pixel coordinates
                        i, j = np.meshgrid(np.arange(width), np.arange(height), indexing='xy')#+np.random.rand((width*height*2)).reshape(2,height,width)

                        i_flat = (i.flatten() - width/2) * K_inv
                        j_flat = (j.flatten() - height/2) * K_inv
                        
                        # Reshape the coordinates to get the 3D points
                        X = i_flat.reshape((height, width)).astype(np.float32)
                        Y = j_flat.reshape((height, width)).astype(np.float32)
                        Z = np.ones_like(X)

                        header = std_msgs.msg.Header()
                        header.stamp = rospy.Time.now()
                        header.frame_id = "camera"  # Change as appropriate for your setup

                        fields = [
                            PointField('x', 0, PointField.FLOAT32, 1),
                            PointField('y', 4, PointField.FLOAT32, 1),
                            PointField('z', 8, PointField.FLOAT32, 1),
                        ]

                        points = (np.stack((X, Y, Z), axis=-1)*depth[:,:,None]).reshape(-1, 3)
                        Y_thresh = 0.2
                        points = points[points[:,2]>0.8]
                        points = points[points[:,1]<Y_thresh]
                        points = points[points[:,1]>-Y_thresh]
                        points = ROT @ points.T
                        points = points.T
                        points*=0.5 #rviz scale


                        #points = ROTC @ points.T
                        #points = points.T
                        #points += np.random.randn(points.size).reshape(points.shape)*0.05
                        point_cloud_msg = pc2.create_cloud(header, fields, points)
                        point_cloud_pub.publish(point_cloud_msg)


                        map = np.zeros((100,100,3))
                        point2map(map,points,0.1,True)
                        map[map>2] = 2
                        map[map<0.2] = 0
                        map = cv2.GaussianBlur(map, (7, 7), 0)
                        idx = point2map(map,pred_pts_np,[1,0,0],False)[1:-1]
                        pred_xy = pred_pts_np[:,:2]
                        direction = pred_xy[1:] - pred_xy[:-1]
                        direction = direction / np.linalg.norm(direction,axis=1)[:,None]
                        perpendic = np.array([-direction[:,1],direction[:,0]]).T
                        coll_risk = 0
                        coll_grads_path = []
                        try:
                            for i in range(len(idx)):
                                x = idx[i][0]
                                y = idx[i][1]
                                x_grad = (map[x+1,y,2] - map[x-1,y,2])/2
                                y_grad = (map[x,y+1,2] - map[x,y-1,2])/2
                                grad = np.array([x_grad,y_grad])
                                grad = perpendic[i] * np.dot(perpendic[i],grad)
                                grad = np.array([grad[0],grad[1],0])
                                coll_risk += map[x,y,2]
                                coll_grads_path.append(grad)
                            coll_grads_path = np.array(coll_grads_path)
                            idx = 10
                            for p,g in zip(pred_pts_np[1:],coll_grads_path):
                                idx+=1
                                mk = generate_ArrowMarker(idx,p,g*10,scale=0.5)
                                mk.header.frame_id = "camera"
                                mks.markers.append(mk)
                            collisionMapPub.publish(bridge.cv2_to_imgmsg((cv2.resize(map, (500,500))*255).astype(np.uint8)))
                        except Exception as e:
                            if DEBUG:
                                print("Error in cost map:", e)
                            continue

                    if len(coll_grads_path) ==5:
                        col_target = pred_pts_np[1:6] + coll_grads_path
                        col_loss = criterion(preds_1, torch.tensor([col_target]).cuda().float())
                        #path_loss = path_loss# + col_loss*2
                    # path_loss.backward()
                    # path_optimizer.step()

                    mk = generate_3DPointMarker(4, U_pts,color=[1, 0, 1,1],scale = 0.01)
                    mk.header.frame_id = "camera"
                    mks.markers.append(mk)
                    HLIP = True
                    if HLIP:
                        x = 0
                        last_x = 0
                        last_y = 0
                        sampled = []
                        dirs = []
                        perps = []
                        centers = []
                        centers = []
                        idx = 0
                        left_start = 0
                        while len(sampled)<N_STEP:
                            if idx >= len(way_points):
                                sampled.append(sampled[-1])
                                centers.append(centers[-1])
                                dirs.append(dirs[-1])
                                perps.append(perps[-1])
                                continue
                            x = way_points[idx,0].item()
                            y = way_points[idx,1].item()
                            dx = x - last_x
                            dy = y - last_y
                            dis = np.sqrt(dx**2 + dy**2)
                            org_loc = np.array([x,y])
                            if dis > tar_step_len:
                                centers.append(way_points[idx,0:2])
                                step_idx = len(sampled)
                                vec = np.array([dx,dy])
                                vec = vec / np.linalg.norm(vec)
                                dirs.append(vec)
                                perp = np.array([-vec[1],vec[0]])
                                perps.append(perp)
                                perpendic = perp * (1 if (step_idx+left_start) % 2 == 0 else -1)
                                offseted = org_loc + perpendic*step_width
                                sampled.append(offseted)
                                last_x = x
                                last_y = y
                            idx+=1

                        if TRAINING and len(sampled) != N_STEP:
                            continue

                        step_traj = np.array(sampled)
                        centers = torch.stack(centers)
                        dir_vecs = torch.tensor(dirs).cuda().float()
                        perp_vecs = torch.tensor(perps).cuda().float()

                        stepx = torch.concatenate([centers,dir_vecs,perp_vecs],axis=1)
                        stepx = stepx.unsqueeze(0)

                        app_out,app_dx, app_y = self.iStepper(stepx)
                        #step_lens = app_dx[:,:-1]-app_dx[:,1:]
                        step_len_loss = (app_dx-0)**2
                        step_len_loss = step_len_loss.mean()
                        step_width_loss = (app_y - step_width)**2
                        step_width_loss = step_width_loss.mean()


                        nm = point2map(map,app_out[0].detach().cpu().numpy(),[1,0,0],True)
                        coll_grads_step = []
                        coll_risk = 0
                        for i in range(len(nm)):
                            x = nm[i][0]
                            y = nm[i][1]
                            if x+1>=map.shape[0] or x-1<0 or y+1>=map.shape[1] or y-1<0:
                                coll_grads_step.append(np.array([0,0]))
                                continue
                            x_grad = (map[x+1,y,2] - map[x-1,y,2])/2
                            y_grad = (map[x,y+1,2] - map[x,y-1,2])/2
                            grad = np.array([x_grad,y_grad])
                            prep = perp_vecs[i].detach().cpu().numpy()
                            grad = prep * np.dot(prep,grad)
                            grad = np.array([grad[0],grad[1]])
                            coll_risk += map[x,y,2]
                            coll_grads_step.append(grad)
                        coll_grads_step = torch.tensor(coll_grads_step).cuda().float()
                        col_target = (app_out[0] + coll_grads_step).detach()
                        col_loss = criterion(app_out[0], col_target)
                        #MPC.set_tar_traj(step_traj)
                        MPC.set_tar_traj(app_out.detach().cpu().numpy()[0])
                        x0 = np.array([0,0,0,0,0,0])
                        solved, cost = MPC.solve(x0,step_cost=True)

                        step_x = solved[1:,0:1] - solved[1:,1:2]
                        step_y = solved[1:,3:4] - solved[1:,4:5]
                        solved = np.hstack([step_x, step_y, np.zeros((N,1))])

                        out_gt = torch.tensor(solved).float().cuda()[None,:,:2]
                        step_MPC_loss = self.stepMSE(app_out, out_gt)
                        ORG_LOSS = False
                        if not ORG_LOSS:
                            step_loss = step_len_loss*10 + step_width_loss*10 + step_MPC_loss*5 + col_loss*10
                            total_loss = path_MPC_loss*50 + step_loss
                        else:
                            total_loss = CostofTraj(preds_1, goal, gamma=2.0, delta=5.0) + col_loss*0.2

                        path_dyn_loss = path_MPC_loss.item()
                        step_loss = step_MPC_loss.item()
                        #print(dyn_loss)
                        if path_dyn_loss < 99:
                            #path_losses.append(path_dyn_loss)
                            records[dir]["path_losses"].append(path_dyn_loss)
                            records[dir]["step_losses"].append(step_loss)
                            records[dir]["col_risks"].append(coll_risk)
                            if not ifTesting:
                                self.stepNNoptimizer.zero_grad()
                                total_loss.backward()
                                #path_loss.backward()
                                path_optimizer.step()
                                if not ORG_LOSS:
                                    self.stepNNoptimizer.step()
                        else:
                            if DEBUG:
                                print("discarding dir: ", dir, " file: ", file, " due to high dynamic loss: ", dyn_loss)
                            pass
                        
                        
                        mk = generate_3DPointMarker(0, way_points.detach().cpu().numpy(),color=[0, 1, 0,1],scale = 0.01)
                        mk.header.frame_id = "camera"
                        mks.markers.append(mk)
                        mk = generate_3DPointMarker(1, solved,color=[1, 0, 0,1],scale =0.02)
                        mk.header.frame_id = "camera"
                        mks.markers.append(mk)
                        mk = generate_3DPointMarker(8, app_out[0],color=[0, 1, 0,1],scale =0.02)
                        mk.header.frame_id = "camera"
                        mks.markers.append(mk)
                    else:
                        
                        mk = generate_3DPointMarker(0, U_tar_traj,color=[0, 1, 0,1],scale = 0.01)
                        mk.header.frame_id = "camera"
                        mks.markers.append(mk)
                        mk = generate_3DPointMarker(6, intp_torch_prev.detach().cpu().numpy() ,color=[0, 1, 1,1],scale = 0.01)
                        mk.header.frame_id = "camera"
                        mks.markers.append(mk)
                    



                    mk = generate_3DPointMarker(2, pred_pts_np,color=[0, 0, 1,1],scale = 0.1)
                    mk.header.frame_id = "camera"
                    mks.markers.append(mk)

                    # mk = generate_marker(np.array([[0, 0, 0]]), marker_id=1)
                    # mks.markers.append(mk)
                    markerPub.publish(mks)
                    #time.sleep(0.3)
                
                for dir in records:
                    path_losses = records[dir]["path_losses"]
                    step_losses = records[dir]["step_losses"]
                    col_risks = records[dir]["col_risks"]
                    dis_vars = records[dir]["dis_vars"]
                    smoothnesses = records[dir]["smoothnesses"]

                    path_losses = np.array(path_losses).mean()*1000
                    path_losses = str(path_losses)[0:6]
                    col_risks = np.array(col_risks).mean()
                    col_risks = str(col_risks)[0:6]
                    dis_vars = np.array(dis_vars).mean()
                    dis_vars = str(dis_vars)[0:6]
                    smoothnesses = np.array(smoothnesses).mean()
                    smoothnesses = str(smoothnesses)[0:6]
                    ifTesting = dir in TEST_DIR
                    dir = dir+" "*(20-len(dir))
                    print("dir ", dir, "testing: ", ifTesting, " dyn loss: ", path_losses, " col risk: ", col_risks, " smoothness: ", smoothnesses, " dis var: ", dis_vars)
                    loggingfile.write("{name:"+dir+", testing: "+str(ifTesting)+", dynLoss: "+path_losses+", colRisk: "+col_risks+", smoothness: "+smoothnesses+", disVar: "+dis_vars+"}\n")

                loggingfile.write("Epoch "+str(epoch)+" done at "+datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")+"\n")
                loggingfile.flush()

            res = input("save checkpoint?")
            if res == "y":
                torch.save(self.iplanner_algo.net.state_dict(), "path.pth")
                torch.save(self.iStepper.state_dict(), "step.pth")
        else:
            self.sub = rospy.Subscriber("/camera/depth/image_rect_raw", Image, self.imageCallback)
            self.last_plan_time = rospy.Time.now()
            rospy.spin()

    def imageCallback(self, msg):
        # rospy.loginfo("Received image %s: %d"%(msg.header.frame_id, msg.header.seq))
        if rospy.Time.now() - self.last_plan_time < rospy.Duration(1.0/FPS):
            return
        self.last_plan_time = rospy.Time.now()

        if GLOBAL_TARGET_POINT is None:
            self.goal_rb = torch.tensor([2, 0, 0], dtype=torch.float32)[None, ...]
        else:
            GLOBAL_TARGET_POINT.header.stamp = rospy.Time(0)
            transformed = listener.transformPoint("camera", GLOBAL_TARGET_POINT).point
            transformed = np.array([transformed.x, transformed.y, transformed.z])
            # if IFSIM:
            #     transformed = -
            #print("Transformed point at ", transformed)
            self.goal_rb = torch.tensor([transformed[0], transformed[1], 0], dtype=torch.float32)[None, ...]
        #self.goal_rb = torch.tensor([2, 0, 0], dtype=torch.float32)[None, ...]
        depth = np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.width).copy()
        if depth.dtype == np.uint16:            
            depth = depth / 1000.0
        
        #cv2.imshow("depth", 1.0 - depth/5)
        depth[~np.isfinite(depth)] = 0
        # DEBUG - Visual Image
        depth[depth > MAX_DEPTH] = 0.0
        
        cur_image = depth.astype(np.float32)
        self.preds, self.waypoints, fear_output, img_process = self.iplanner_algo.plan(cur_image, self.goal_rb)


        HLIP = True
        left_start = SN["ifLeft"].copy()
        mks = MarkerArray()
        if HLIP:
            pts = self.preds[0].detach().cpu().numpy()
            pts = np.vstack([np.zeros(3),pts, self.goal_rb])
            way_points = pp.chspline(torch.tensor(pts).float(), 0.01)
            last_x = 0
            last_y = 0
            sampled = []
            traj = []
            idx = 0
            while len(sampled)<N_STEP:
                if idx >= len(way_points):
                    sampled.append(way_points[-1,:2].numpy())
                    continue
                x = way_points[idx,0]
                y = way_points[idx,1]
                dx = x - last_x
                dy = y - last_y
                dis = np.sqrt(dx**2 + dy**2)
                org_loc = np.array([x,y])
                traj.append(org_loc)
                if dis > tar_step_len:
                    step_idx = len(sampled)
                    vec = np.array([dx,dy])
                    vec = vec / np.linalg.norm(vec)
                    perpendic = np.array([-vec[1],vec[0]]) * (1 if (step_idx+left_start) % 2 == 0 else -1)
                    offseted = org_loc + perpendic*step_width
                    sampled.append(offseted)
                    last_x = x
                    last_y = y
                idx+=1

            step_traj = np.array(sampled)

            MPC.set_tar_traj(step_traj)
            x0 = np.array([0,0,0,0,0,0])
            solved, cost = MPC.solve(x0,step_cost=True)

            step_x = solved[1:,0:1] - solved[1:,1:2]
            step_y = solved[1:,3:4] - solved[1:,4:5]
            solved = np.hstack([step_x, step_y, np.zeros((N,1))])
            print("first 3 steps: ", solved[:3,:2])
            mk = generate_3DPointMarker(1, solved,color=[1, 1, 0,1],scale=0.03)
            mks.markers.append(mk)
            mk.header.frame_id = "camera"

        if RECORDING:
            data = {"preds": self.preds, "goal": self.goal_rb, "input": cur_image}
            date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            torch.save(data, f"recordings/{recording_dir}/{date}.pt")
        #self.preds*=-1
        print(SN["ifLeft"])
        # check goal less than coverage range
        pts = self.preds[0].detach().cpu().numpy()
        pts = np.vstack([pts, self.goal_rb])
        mk = generate_3DPointMarker(0, pts,color=[0, 1, 0,1])
        mks.markers.append(mk)
        mk.header.frame_id = "camera"
        # mk = generate_marker(np.array([[0, 0, 0]]), marker_id=1)
        # mks.markers.append(mk)
        markerPub.publish(mks)
        cv2.waitKey(1)


class PlannerNet(nn.Module):
    def __init__(self, encoder_channel=16, k=5):
        super().__init__()
        self.encoder = PerceptNet(layers=[2, 2, 2, 2])
        self.decoder = Decoder(512, encoder_channel, k)

    def forward(self, x, goal):
        x = self.encoder(x)
        x, c = self.decoder(x, goal)
        return x, c


class Decoder(nn.Module):
    def __init__(self, in_channels, goal_channels, k=5):
        super().__init__()
        self.k = k
        self.relu    = nn.ReLU(inplace=True)
        self.fg      = nn.Linear(3, goal_channels)
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d((in_channels + goal_channels), 512, kernel_size=5, stride=1, padding=1)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0);

        self.fc1   = nn.Linear(256 * 128, 1024) 
        self.fc2   = nn.Linear(1024, 512)
        self.fc3   = nn.Linear(512,  k*3)
        
        self.frc1 = nn.Linear(1024, 128)
        self.frc2 = nn.Linear(128, 1)

    def forward(self, x, goal):
        # compute goal encoding
        goal = self.fg(goal[:, 0:3])
        goal = goal[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
        # cat x with goal in channel dim
        x = torch.cat((x, goal), dim=1)
        # compute x
        x = self.relu(self.conv1(x))   # size = (N, 512, x.H/32, x.W/32)
        x = self.relu(self.conv2(x))   # size = (N, 512, x.H/60, x.W/60)
        x = torch.flatten(x, 1)

        f = self.relu(self.fc1(x))

        x = self.relu(self.fc2(f))
        x = self.fc3(x)
        x = x.reshape(-1, self.k, 3)

        c = self.relu(self.frc1(f))
        c = self.sigmoid(self.frc2(c))

        return x, c


class CubicSplineTorch:
    # Reference: https://stackoverflow.com/questions/61616810/how-to-do-cubic-spline-interpolation-and-integration-in-pytorch
    def __init__(self):
        return None

    def h_poly(self, t):
        alpha = torch.arange(4, device=t.device, dtype=t.dtype)
        tt = t[:, None, :]**alpha[None, :, None]
        A = torch.tensor([
            [1, 0, -3, 2],
            [0, 1, -2, 1],
            [0, 0, 3, -2],
            [0, 0, -1, 1]
            ], dtype=t.dtype, device=t.device)
        return A @ tt

    def interp(self, x, y, xs):
        m = (y[:, 1:, :] - y[:, :-1, :]) / torch.unsqueeze(x[:, 1:] - x[:, :-1], 2)
        m = torch.cat([m[:, None, 0], (m[:, 1:] + m[:, :-1]) / 2, m[:, None, -1]], 1)
        idxs = torch.searchsorted(x[0, 1:], xs[0, :])
        dx = x[:, idxs + 1] - x[:, idxs]
        hh = self.h_poly((xs - x[:, idxs]) / dx)
        hh = torch.transpose(hh, 1, 2)
        out = hh[:, :, 0:1] * y[:, idxs, :]
        out = out + hh[:, :, 1:2] * m[:, idxs] * dx[:,:,None]
        out = out + hh[:, :, 2:3] * y[:, idxs + 1, :]
        out = out + hh[:, :, 3:4] * m[:, idxs + 1] * dx[:,:,None]
        return out

class TrajOpt:
    def __init__(self):
        self.cs_interp = CubicSplineTorch()
        return None

    def TrajGeneratorFromPFreeRot(self, preds, step): 
        # Points is in se3
        batch_size, num_p, dims = preds.shape
        points_preds = torch.cat((torch.zeros(batch_size, 1, dims, device=preds.device, requires_grad=preds.requires_grad), preds), axis=1)
        num_p = num_p + 1
        xs = torch.arange(0, num_p-1+step, step, device=preds.device)
        xs = xs.repeat(batch_size, 1)
        x  = torch.arange(num_p, device=preds.device, dtype=preds.dtype)
        x  = x.repeat(batch_size, 1)
        waypoints = self.cs_interp.interp(x, points_preds, xs)
        return waypoints  # R3

import PIL.Image
import math
class IPlannerAlgo:
    def __init__(self, args):
        super(IPlannerAlgo, self).__init__()
        self.config(args)

        self.depth_transform = transforms.Compose([
            transforms.Resize(tuple(self.crop_size)),
            transforms.ToTensor()])

        #net, _ = torch.load("iplanner/models/plannernet.pt", map_location=torch.device("cpu"))
        net = PlannerNet()
        net.load_state_dict(torch.load("my2.pt"))
        self.net = net.cuda() if torch.cuda.is_available() else net

        self.traj_generate = TrajOpt()
        return None

    def config(self, args):
        self.crop_size  = args.crop_size
        self.sensor_offset_x = args.sensor_offset_x
        self.sensor_offset_y = args.sensor_offset_y
        self.is_traj_shift = False
        if math.hypot(self.sensor_offset_x, self.sensor_offset_y) > 1e-1:
            self.is_traj_shift = True
        return None


    def plan(self, image, goal_robot_frame):
        img = PIL.Image.fromarray(image)
        img = self.depth_transform(img).expand(1, 3, -1, -1)
        if torch.cuda.is_available():
            img = img.cuda()
            goal_robot_frame = goal_robot_frame.cuda()
        keypoints, fear = self.net(img, goal_robot_frame)
        if self.is_traj_shift:
            batch_size, _, dims = keypoints.shape
            keypoints = torch.cat((torch.zeros(batch_size, 1, dims, device=keypoints.device, requires_grad=False), keypoints), axis=1)
            keypoints[..., 0] += self.sensor_offset_x
            keypoints[..., 1] += self.sensor_offset_y
        traj = self.traj_generate.TrajGeneratorFromPFreeRot(keypoints , step=0.1)
        
        return keypoints, traj, fear, img


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Path Planner Configuration')

    parser.add_argument('--crop_size', type=int, nargs=2, default=[360, 640], help='Dimensions to crop the image to.')
    parser.add_argument('--sensor_offset_x', type=float, default=0.0, help='Sensor offset on the X-axis.')
    parser.add_argument('--sensor_offset_y', type=float, default=0.0, help='Sensor offset on the Y-axis.')

    args = parser.parse_args()

    node = iPlannerNode(args)
