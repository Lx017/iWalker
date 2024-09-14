from Startups.memory_manager import *
from Play.Walking.low_level import lowMain
from Play.Walking.high_level import highMain
from Play.Walking.top_level import topMain
import numpy as np
import multiprocessing as mp
import time
from Hlip import *

low_process=mp.Process(target=lowMain)
high_process=mp.Process(target=highMain)
top_process=mp.Process(target=topMain)



"""
Script for communication with Mujoco
"""

import time
from Settings.Bruce import *
import Startups.memory_manager as MM
import Library.ROBOT_MODEL.BRUCE_kinematics as kin
import Library.ROBOT_MODEL.BRUCE_dynamics as dyn
from Settings.BRUCE_macros import *
from lxutil import *
import cv2
bot=BRUCE()

import mujoco
import mujocoViewer
from robot import *
from snp import SharedNp

from sensor_msgs.msg import Image, PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
bridge = CvBridge()
import std_msgs
from std_msgs.msg import String
import rospy
import tf.transformations as tft

contactForceSMArr = shmArr("forceSM", (4,3), np.float32, reset=True)
contactForceSMArr[0,0]=1234
dt = 0.001
SN = SharedNp("shm1.json")
SN["boffset"][0] = 0
SN["boffset"][1] = 0

rospy.init_node("mujoco")
point_cloud_pub = rospy.Publisher('/ptsc', PointCloud2, queue_size=10)
depth_pub = rospy.Publisher('/camera/depth/image_rect_raw', Image, queue_size=10)
marker_pub = rospy.Publisher('/muj_marker', Marker, queue_size=10)
traj_pub = rospy.Publisher('/muj_traj', Marker, queue_size=10)
sol_step_pub = rospy.Publisher('/muj_step_solved', PointStamped, queue_size=10)
mea_step_pub = rospy.Publisher('/muj_step_mea', PointStamped, queue_size=10)
TFbr = tf.TransformBroadcaster()
listener = tf.TransformListener()
planned_traj = np.zeros((6,3))
def plannerCallback(data):
    traj = [[p.x, p.y, p.z] for p in data.markers[0].points] #camera frame
    planned_traj[1:] = traj
iplannerS = rospy.Subscriber("/iplanner_out", MarkerArray, plannerCallback)
rate = rospy.Rate(10000)

ROTZ = RotMatZ(90)
ROT_OFFSET = np.eye(3)
ROOT_OFFSET = np.array([0.,0.,0.])
YAW_OFFSET = 0
ROT_d = RotMatZ(0.5)
ROOT_MEA = np.zeros(3)
ROTC = ROTZ @ np.array([[1,  0,  0],
                    [0,  0,  1],
                    [0,  -1,  0.]]) 


bot.setModuleState(MODULE_IFSIM,1)

def land_callback(id):
    ROOT_OFFSET[:] = data.qpos[0:3].copy()
    traj = -planned_traj
    if traj[0,0] >= traj[1,0]:
        traj[1,0] = traj[0,0] + 0.01
    if traj[1,0] >= traj[2,0]:
        traj[2,0] = traj[1,0] + 0.01
    if traj[2,0] >= traj[3,0]:
        traj[3,0] = traj[2,0] + 0.01
    if traj[3,0] >= traj[4,0]:
        traj[4,0] = traj[3,0] + 0.01
    if traj[4,0] >= traj[5,0]:
        traj[5,0] = traj[4,0] + 0.01
    spline = CubicSpline(traj[:,0], traj[:,1])

    

    cur_step_ps = PointStamped()
    cur_step_ps.header.stamp = rospy.Time.now()
    cur_step_ps.header.frame_id = "camera"
    cur_step_ps.point.x = -SN["footLoc"][0]
    cur_step_ps.point.y = -SN["footLoc"][1]
    cur_step_ps.point.z = 0


    sol_step_pub.publish(cur_step_ps)


    x = 0
    last_x = 0
    last_y = spline(last_x)
    tar_step_len = 0.03
    sampled = []
    traj = []
    step_width = 0.03
    N_STEP = 50
    while len(sampled)<N_STEP:
        y = spline(x)
        dx = x - last_x
        dy = y - last_y
        dis = np.sqrt(dx**2 + dy**2)
        org_loc = np.array([x,y])
        traj.append(org_loc)
        if dis > tar_step_len:
            step_idx = len(sampled)
            vec = np.array([dx,dy])
            vec = vec / np.linalg.norm(vec)
            perpendic = np.array([-vec[1],vec[0]]) * (1 if (step_idx+mujSim.step_idx) % 2 == 0 else -1)
            offseted = org_loc + perpendic*step_width
            sampled.append(offseted)
            last_x = x
            last_y = y
        x+=0.001

    step_traj = np.array(sampled)

    MPC.set_tar_traj(step_traj)
    x0 = np.array([-cur_step_ps.point.x,0,0,-cur_step_ps.point.y,0,0])
    solved, cost = MPC.solve(x0,step_cost=True)

    step_x = solved[1:,0:1] - solved[1:,1:2]
    step_y = solved[1:,3:4] - solved[1:,4:5]
    STEP_TAR[:] = np.array([step_x[4,0], step_y[4,0]])

    marker = generate_3DPointMarker(0, np.hstack((-step_x,-step_y, np.zeros(N_STEP)[:,None])), color=(1.0, 1.0, 0.0, 1.0), scale=0.1)
    for i in range(len(marker.points)):
        ps = PointStamped()
        ps.header.frame_id = "camera"
        ps.point.x = marker.points[i].x
        ps.point.y = marker.points[i].y
        ps.point.z = 0
        transformed = listener.transformPoint("map", ps)
        marker.points[i].x = transformed.point.x
        marker.points[i].y = transformed.point.y
        marker.points[i].z = 0

    traj_pub.publish(marker)

    org_traj = np.array(traj)

    tar_pt = [step_traj[10,0], step_traj[10,1], 0.]

    (trans, rot) = listener.lookupTransform("map", "camera", rospy.Time(0))
    trans = np.array(trans)
    rotation_matrix = tft.quaternion_matrix(rot)[:3, :3]
    transformed = (rotation_matrix @ -np.array(tar_pt) + trans) * np.array([-1, -1, 1])
    ROOT_TAR[:] = [transformed[0], transformed[1]]


    cur_step_ps = PointStamped()
    cur_step_ps.header.stamp = rospy.Time.now()
    cur_step_ps.header.frame_id = "map"
    cur_step_ps.point.x = -data.xpos[2:][4+5*id][0]
    cur_step_ps.point.y = -data.xpos[2:][4+5*id][1]
    cur_step_ps.point.z = 0

    mea_step_pub.publish(cur_step_ps)
class MujocoSimulator:
    def __init__(self):
        # robot info
        self.num_legs = 2
        self.num_joints_per_leg = 5
        self.num_arms = 2
        self.num_joints_per_arms = 3
        self.num_joints = self.num_legs * self.num_joints_per_leg + self.num_arms * self.num_joints_per_arms
        self.num_contact_sensors = 4
        
        self.leg_p_gains = [265, 150,  80,  80,    30]
        self.leg_i_gains = [  0,   0,   0,   0,     0]
        self.leg_d_gains = [ 1., 2.3, 0.8, 0.8, 0.003]

        self.arm_p_gains = [ 1.6,  1.6,  1.6]
        self.arm_i_gains = [   0,    0,    0]
        self.arm_d_gains = [0.03, 0.03, 0.03]

        self.p_gains = self.leg_p_gains * 2 + self.arm_p_gains * 2  # the joint order matches the robot"s sdf file
        self.i_gains = self.leg_i_gains * 2 + self.arm_i_gains * 2
        self.d_gains = self.leg_d_gains * 2 + self.arm_d_gains * 2
        
        # simulator info
        self.simulator = None
        self.simulation_frequency = 1000  # Hz
        self.simulation_modes = {"torque": 0, "position": 2}
        self.simulation_mode = self.simulation_modes["position"]
        self.last_foot_contacts = np.zeros(self.num_contact_sensors)
        self.step_idx = 0
        
    def update_sensor_info(self,bot:BRUCE,data:mujoco.MjData):
        """
        Get sensor info and write it to shared memory.
        """
        q = data.qpos[7:].copy()
        dq = data.qvel[6:].copy()
        # data.qpos[7:] = q
        # data.qvel[6:] = dq
        self.q_leg  = np.array([q[0]-PI_2, q[1]+PI_2, q[2], q[3], q[4],
                                q[5]-PI_2, q[6]+PI_2, q[7], q[8], q[9]])
        self.q_arm  = q[10:16]
        self.dq_leg = dq[0:10]
        self.dq_arm = dq[10:16]

        baseSE3_data = list(data.qpos[0:7])
        baseSE3_data.append(baseSE3_data.pop(3))
        baseSE3 = pp.SE3(to_tensor(baseSE3_data))

        self.rot_mat = ROT_OFFSET @ baseSE3.matrix()[:3,:3].numpy().astype(np.float64)
        self.accel   = ROT_OFFSET @ data.qacc[0:3].copy()

        IMU_data = data.sensordata[0:3].copy()

        self.omega   = self.rot_mat.T @ data.qvel[3:6]
        self.foot_contacts = np.array([0,0,0,0])

        if(data.xpos[2:][4][2]<0.028):
            self.foot_contacts[0:2] = 1
        if(data.xpos[2:][9][2]<0.028):
            self.foot_contacts[2:4] = 1
        
        if self.last_foot_contacts[0] == 0 and self.foot_contacts[0] == 1:
            try:
                land_callback(0)
            except:
                pass
            self.step_idx = 0
        
        if self.last_foot_contacts[2] == 0 and self.foot_contacts[2] == 1:
            try:
                land_callback(1)
            except:
                pass
            self.step_idx = 1

        self.last_foot_contacts = self.foot_contacts.copy()

        bot.setState(LEG_STATE_POS,self.q_leg)
        bot.setState(LEG_STATE_VEL,self.dq_leg)
        bot.setState(ARM_STATE_POS,self.q_arm)
        bot.setState(ARM_STATE_VEL,self.dq_arm)
    
    def calculate_robot_model(self,bot:BRUCE,data:mujoco.MjData):
        """
        Calculate kinematics & dynamics and write it to shared memory.
        """
        R_jntPos1, R_jntPos2, R_jntPos3, R_jntPos4, R_jntPos5 = self.q_leg[0], self.q_leg[1], self.q_leg[2], self.q_leg[3], self.q_leg[4]
        R_jntVel1, R_jntVel2, R_jntVel3, R_jntVel4, R_jntVel5 = self.dq_leg[0], self.dq_leg[1], self.dq_leg[2], self.dq_leg[3], self.dq_leg[4]
        L_jntPos1, L_jntPos2, L_jntPos3, L_jntPos4, L_jntPos5 = self.q_leg[5], self.q_leg[6], self.q_leg[7], self.q_leg[8], self.q_leg[9]
        L_jntVel1, L_jntVel2, L_jntVel3, L_jntVel4, L_jntVel5 = self.dq_leg[5], self.dq_leg[6], self.dq_leg[7], self.dq_leg[8], self.dq_leg[9]

        bodyRot = self.rot_mat
        bodyOmg_b = self.omega

        bodyPos = ROT_OFFSET @ (data.qpos[0:3].copy()- ROOT_OFFSET) 
        bodyVel = ROT_OFFSET @ data.qvel[0:3].copy()

        bodyAcc = bodyRot @ self.accel
        bodyVel_b = bodyRot.T @ bodyVel
        yaw_angle = np.arctan2(bodyRot[1, 0], bodyRot[0, 0])

        # compute leg forward kinematics
        R_ToeBodyPos    , R_ToeBodyVel  , R_Jac_bodyToeVel  , R_dJdt_bodyToeVel, \
        R_HeelBodyPos   , R_HeelBodyVel , R_Jac_bodyHeelVel , R_dJdt_bodyHeelVel, \
        R_AnkleBodyPos  , R_AnkleBodyVel, R_Jac_bodyAnkleVel, R_dJdt_bodyAnkleVel, \
        R_FootBodyPos   , R_FootBodyVel , R_bodyFootRot     , R_Jac_bodyFoot   , R_dJdt_bodyFoot, \
        L_ToeBodyPos    , L_ToeBodyVel  , L_Jac_bodyToeVel  , L_dJdt_bodyToeVel, \
        L_HeelBodyPos   , L_HeelBodyVel , L_Jac_bodyHeelVel , L_dJdt_bodyHeelVel, \
        L_AnkleBodyPos  , L_AnkleBodyVel, L_Jac_bodyAnkleVel, L_dJdt_bodyAnkleVel, \
        L_FootBodyPos   , L_FootBodyVel , L_bodyFootRot     , L_Jac_bodyFoot   , L_dJdt_bodyFoot = kin.legFK(R_jntPos1 , R_jntPos2    , R_jntPos3    , R_jntPos4    , R_jntPos5,
                                                                                                            L_jntPos1  , L_jntPos2    , L_jntPos3    , L_jntPos4    , L_jntPos5,
                                                                                                            R_jntVel1 , R_jntVel2   , R_jntVel3   , R_jntVel4   , R_jntVel5,
                                                                                                            L_jntVel1 , L_jntVel2   , L_jntVel3   , L_jntVel4   , L_jntVel5)

        L_Vel=np.array([L_jntVel1,L_jntVel2,L_jntVel3,L_jntVel4,L_jntVel5])
        R_Vel=np.array([R_jntVel1,R_jntVel2,R_jntVel3,R_jntVel4,R_jntVel5])
        # compute robot forward kinematics
        R_toePos    , R_toeVel  , R_Jac_wToeVel  , R_dJdq_toeVel, \
        R_heelPos   , R_heelVel , R_Jac_wHeelVel , R_dJdq_heelVel, \
        R_AnklePos  , R_AnkleVel, R_Jac_wAnkleVel, R_dJdq_AnkleVel, \
        R_footPos   , R_footVel ,  \
        R_footRot   , R_footOmg, R_Jac_footOmg, dJwdq_ff_r, \
        L_toePos    , L_toeVel  , L_Jac_wToeVel  , L_dJdq_toeVel, \
        L_heelPos   , L_heelVel , L_Jac_wHeelVel , L_dJdq_heelVel, \
        L_AnklePos  , L_AnkleVel, L_Jac_wAnkleVel, L_dJdq_AnkleVel, \
        L_footPos   , L_footVel ,  \
        L_footRot   , L_footOmg, L_Jac_footOmg, dJwdq_ff_l =kin.robotFK(bodyRot       , bodyPos             , bodyOmg_b         , bodyVel_b,
                                                                                R_ToeBodyPos  , R_Jac_bodyToeVel    , R_dJdt_bodyToeVel ,
                                                                                R_HeelBodyPos , R_Jac_bodyHeelVel   , R_dJdt_bodyHeelVel,
                                                                                R_AnkleBodyPos, R_Jac_bodyAnkleVel  , R_dJdt_bodyAnkleVel, R_bodyFootRot, R_Jac_bodyFoot, R_dJdt_bodyFoot,
                                                                                L_ToeBodyPos  , L_Jac_bodyToeVel    , L_dJdt_bodyToeVel,
                                                                                L_HeelBodyPos , L_Jac_bodyHeelVel   , L_dJdt_bodyHeelVel,
                                                                                L_AnkleBodyPos, L_Jac_bodyAnkleVel  , L_dJdt_bodyAnkleVel, L_bodyFootRot, L_Jac_bodyFoot, L_dJdt_bodyFoot,
                                                                                R_Vel         , L_Vel)

        # calculate robot dynamics
        H, CG, AG, dAGdq, comPos, comVel, angMom = dyn.robotID(bodyRot, bodyPos, bodyOmg_b, bodyVel_b,
                                                         R_jntPos1, R_jntPos2, R_jntPos3, R_jntPos4, R_jntPos5,
                                                         L_jntPos1, L_jntPos2, L_jntPos3, L_jntPos4, L_jntPos5,
                                                         R_jntVel1, R_jntVel2, R_jntVel3, R_jntVel4, R_jntVel5,
                                                         L_jntVel1, L_jntVel2, L_jntVel3, L_jntVel4, L_jntVel5)

        # save as estimation data
        bot[EST]["time_stamp"] = np.array([data.time])
        bot[EST]["bodyPos"] = bodyPos
        bot[EST]["bodyVel"] = bodyVel
        #bot[EST]["bodyAcc"] = bodyAcc
        bot[EST]["bodyRot"] = bodyRot
        bot[EST]["bodyOmg"] = bodyOmg_b
        bot[EST]["body_yaw_ang"] = np.array([yaw_angle])
        bot[EST]["COMPos"] = comPos
        bot[EST]["COMVel"] = comVel
        bot[EST]["angMomemtum"] = angMom
        bot[EST]["H_matrix"] = H
        bot[EST]["CG_vector"] = CG
        bot[EST]["AG_matrix"] = AG
        bot[EST]["dAGdq_vector"] = dAGdq
        bot[EST]["foot_contacts"] = self.foot_contacts
        bot[EST]["R_footRot"] = R_footRot
        bot[EST]["R_footOmg"] = R_footOmg
        bot[EST]["R_foot_Jw"] = R_Jac_footOmg
        bot[EST]["R_foot_dJwdq"] = dJwdq_ff_r
        bot[EST]["R_footPos"] = R_footPos
        bot[EST]["R_footVel"] = R_footVel
        bot[EST]["R_toePos"] = R_toePos
        bot[EST]["R_toeVel"] = R_toeVel
        bot[EST]["R_Jac_wToeVel"] = R_Jac_wToeVel
        bot[EST]["R_dJ_toeVel*dq"] = R_dJdq_toeVel
        bot[EST]["R_heelPos"] = R_heelPos
        bot[EST]["R_heelVel"] = R_heelVel
        bot[EST]["R_Jac_wHeelVel"] = R_Jac_wHeelVel
        bot[EST]["R_dJ_heelVel*dq"] = R_dJdq_heelVel
        bot[EST]["R_anklePos"] = R_AnklePos
        bot[EST]["R_ankleVel"] = R_AnkleVel
        bot[EST]["R_Jac_wAnkleVel"] = R_Jac_wAnkleVel
        bot[EST]["R_dJ_AnkleVel*dq"] = R_dJdq_AnkleVel
        bot[EST]["L_footRot"] = L_footRot
        bot[EST]["L_footOmg"] = L_footOmg
        bot[EST]["L_foot_Jw"] = L_Jac_footOmg
        bot[EST]["L_foot_dJwdq"] = dJwdq_ff_l
        bot[EST]["L_footPos"] = L_footPos
        bot[EST]["L_footVel"] = L_footVel
        bot[EST]["L_toePos"] = L_toePos
        bot[EST]["L_toeVel"] = L_toeVel
        bot[EST]["L_Jac_wToeVel"] = L_Jac_wToeVel
        bot[EST]["L_dJ_toeVel*dq"] = L_dJdq_toeVel
        bot[EST]["L_heelPos"] = L_heelPos
        bot[EST]["L_heelVel"] = L_heelVel
        bot[EST]["L_Jac_wHeelVel"] = L_Jac_wHeelVel
        bot[EST]["L_dJ_heelVel*dq"] = L_dJdq_heelVel
        bot[EST]["L_anklePos"] = L_AnklePos
        bot[EST]["L_ankleVel"] = L_AnkleVel
        bot[EST]["L_Jac_wAnkleVel"] = L_Jac_wAnkleVel
        bot[EST]["L_dJ_AnkleVel*dq"] = L_dJdq_AnkleVel


class PID:
    def __init__(self, k_p,k_d, d_filter=0):
        self.k_p = np.array(k_p)
        self.k_d = np.array(k_d)
        self.last_pos = None
        self.d_filter = d_filter
        self.last_derivative = 0
        self.integral = 0
    
    def update(self, target, pos_mea, dt):
        pos_error = target - pos_mea
        pos_error = np.clip(pos_error, -0.1, 0.1)#assume 0.5 radian is the maximum error
        if self.last_pos is None:
            self.last_pos = pos_mea
        if self.last_derivative is None:
            self.last_derivative = 0
        vel_mea = (pos_mea - self.last_pos) / dt
        derivative = self.last_derivative*self.d_filter + vel_mea*(1-self.d_filter)
        self.last_derivative = derivative
        self.last_pos = pos_mea.copy()
        unconstrained = self.k_p * pos_error + self.k_d * (0-derivative)
        return unconstrained, vel_mea, np.abs(pos_error).sum()

XML = open("bruce.xml", "r").read()
top, bottom = XML.split('<!-- add geo here -->')
XML = [top,
'<geom type="mesh" pos="1 0 0" mesh="cube" contype="0" conaffinity="0"/>',
bottom]
XML_string = '\n'.join(XML)
model = mujoco.MjModel.from_xml_string(XML_string)

# model = mujoco.MjModel.from_xml_path('humanoid.xml')
data = mujoco.MjData(model)
# renderer = mujoco.Renderer(model)
# renderer.enable_depth_rendering()

model.opt.timestep = dt
model.dof_damping[6:] = 0.0005
model.dof_damping[16:] = 999# arms are not moving
# create the viewer object

data.qpos[:7] = np.array([0, 0, 0.45, 1, 0, 0, 0])

ar1, ar2, ar3 = -0.7,  1.3,  2.0
al1, al2, al3 =  0.7, -1.3, -2.0
X_OFFSET=-0.0003
bpr = np.array([0.04+X_OFFSET, -0.07, -0.42])  # right foot position  in body frame
bpl = np.array([0.04+X_OFFSET, +0.07, -0.42])  # left  foot position  in body frame
bxr = np.array([1., 0., 0.])          # right foot direction in body frame
bxl = np.array([1., 0., 0.])          # left  foot direction in body frame
lr1, lr2, lr3, lr4, lr5 = kin.legIK_foot(bpr, bxr, +1.)
ll1, ll2, ll3, ll4, ll5 = kin.legIK_foot(bpl, bxl, -1.)
armTarPos = [ar1, ar2, ar3, al1, al2, al3]
legTarPos = [lr1+PI_2, lr2-PI_2, lr3, lr4, lr5,
                ll1+PI_2, ll2-PI_2, ll3, ll4, ll5]


data.qpos[7:17]=np.array(legTarPos)


viewer = mujocoViewer.MujocoViewer(model, data)


bot = BRUCE()
legposSMArr=bot.All_PosCmd_SMArr
legtorqSMArr=bot.Leg_Torq_SMArr
threadSMArr=bot.Thread_SMArr
MM.init()
MM.connect()
threadSMArr[:]=0

# BRUCE SETUP
mujSim = MujocoSimulator()

MM.THREAD_STATE.set({"simulation": np.array([1.0])}, opt="only")  # thread is running

# simulate and render
dof = data.ctrl.shape[0]
targetPos = np.zeros(dof)
lastPos = np.zeros(dof)
vel = np.zeros(dof)
posIntegral = np.zeros(dof)
I = np.zeros(dof)

nameStr = model.names.decode().replace("\x00", ";")
maxTorque = 10
lastTorque = np.zeros(dof)
joint_idx_dict = {}

pid = PID(k_p=[200,200,200,200,200,
                200,200,200,200,200], 
        k_d=[0.1,0.1,0.1,0.1,0.1,
                0.1,0.1,0.1,0.1,0.1], d_filter=0.2)

FPS = 30
last_render_time = 0
sim_launch_time = time.time()
last_vel = None
I = np.zeros(10)
W = np.zeros(10)
T = np.zeros(10)
TT = []
cur_acc = np.zeros(10)

ar1, ar2, ar3 = -0.7,  1.3,  2.0
al1, al2, al3 =  0.7, -1.3, -2.0
X_OFFSET=-0.02
bpr = np.array([0.04+X_OFFSET, -0.07, -0.42])  # right foot position  in body frame
bpl = np.array([0.04+X_OFFSET, +0.07, -0.42])  # left  foot position  in body frame
bxr = np.array([1., 0., 0.])          # right foot direction in body frame
bxl = np.array([1., 0., 0.])          # left  foot direction in body frame
lr1, lr2, lr3, lr4, lr5 = kin.legIK_foot(bpr, bxr, +1.)
ll1, ll2, ll3, ll4, ll5 = kin.legIK_foot(bpl, bxl, -1.)
armTarPos = [ar1, ar2, ar3, al1, al2, al3]
legTarPos = [lr1+PI_2, lr2-PI_2, lr3, lr4, lr5,
                ll1+PI_2, ll2-PI_2, ll3, ll4, ll5]
legTarPos = np.array(legTarPos)

low_process.start()
high_process.start()
top_process.start()

ROOT_TAR = np.array([0.,0.])
STEP_TAR = np.array([0.,0.])
while viewer.is_alive:
    
    if time.time() - last_render_time > 1 / FPS:
        render = True
        last_render_time = time.time()
    else:
        render = False        

    
    if bot.thread_error():
        bot.stop_threading()

    mujSim.update_sensor_info(bot, data)
    mujSim.calculate_robot_model(bot,data)


    
    if time.time()-sim_launch_time <1.2:
        # reset the robot to the initial position
        data.qpos[:7] = np.array([0, 0, 0.42, 1, 0, 0, 0])
        data.qpos[7:17]=legTarPos
        data.qvel[:6] = np.zeros(6)


    cmd, cur_vel, error = pid.update(legTarPos, data.qpos[7:17], dt)

    if last_vel is None:
        last_vel = cur_vel
    cur_acc = (cur_vel-last_vel)/dt*0.7+cur_acc*0.3
    #cur_vel = cur_vel*0.7+last_vel*0.3
    acc_gain = np.clip(cur_acc*dt, -10, 10)
    vel_gain = np.clip(100*cur_vel*dt, -10, 10)
    T-=acc_gain
    T-=vel_gain
    last_vel = cur_vel


    pressed_key = viewer.get_keyPressed()
    if len(pressed_key)>0:
        print(ROOT_TAR)
        DX = 0.02
        for k in pressed_key:
            k = chr(k)
            if k == "W":
                #SN["boffset"][0] = 1
                ROOT_TAR += (ROT_OFFSET.T @ np.array([DX,0,0]))[:2]
            elif k == "S":
                ROOT_TAR -= (ROT_OFFSET.T @ np.array([DX,0,0]))[:2]
            elif k == "A":
                ROOT_TAR += (ROT_OFFSET.T @ np.array([0,DX,0]))[:2]
            elif k == "D":
                ROOT_TAR -= (ROT_OFFSET.T @ np.array([0,DX,0]))[:2]
            elif k == "Q":
                YAW_OFFSET -= 1
            elif k == "E":
                YAW_OFFSET += 1
            elif k == "X":
                ROOT_TAR*=0
    
    ROT_OFFSET[:] = RotMatZ(YAW_OFFSET)

    # bot[PLAN]["bodyRot"] = ROT_TAR #force update every frame
    # bot[PLAN]["R_footRot"] = ROT_TAR
    # bot[PLAN]["L_footRot"] = ROT_TAR

    if viewer.if_reset():
        for i in range(10):
            data.qpos[7:17]=np.array(legTarPos)
            T *=0
            data.qpos[:7] = np.array([0, 0, 0.45, -1, 0, 0, 0])
            data.qvel[:] =0
            data.qacc[:] =0
            data.ctrl[:] = 0
            last_vel = None
            cur_acc = np.zeros(10)
            mujoco.mj_step(model, data)

    TT.append(T[3])
    #final_cmd = np.clip(cmd,-3,3)
    final_cmd = cmd



    if threadSMArr[0]==1:
        #gs.write_position(legTarPos, armTarPos)
        data.ctrl[:10] = legtorqSMArr[2:12]
    else:#force set position and velocity
        data.ctrl[:10] = final_cmd
        data.qpos[17:] = np.array(armTarPos)
        # data.qvel[6:] = np.zeros(16)
    # print("Q", data.qpos[:5])self.viewport.
    mujoco.mj_step(model, data)


    if render:
        try:
            viewer.render(None)
            near = 0.09416321665048599
            FOVY = 90
            h = near * np.tan(FOVY / 2 * np.pi / 180)
            depth, near, far = viewer.render(1,readDepth=True,h=h,shape=(320,180))
            depth = (near*far)/(near+depth*(far-near)) # restore depth
            depth[depth>10] = 0
            depth = cv2.GaussianBlur(depth, (41, 41), 0)
            image_msg = bridge.cv2_to_imgmsg((depth*1000).astype(np.uint16), encoding="16UC1")
            # Publish the image
            depth_pub.publish(image_msg)
            depth *= (1.0+np.random.randn(depth.size).reshape(depth.shape)*0.02)
            #depth = cv2.GaussianBlur(depth, (21, 21), 0)
            center_depth =depth[depth.shape[0]//2,depth.shape[1]//2]

            #rot = matrix_to_quaternion(data.xmat[1].reshape(3,3).T*np.array([-1,-1,1])[:,None])
            TFbr.sendTransform(
                data.xpos[1]*np.array([-1,-1,1]),     # translation
                np.roll(data.qpos[3:7]*np.array([1,-1,-1,1]),-1), # rotation
                rospy.Time.now(),
                'camera',  # child frame
                'map'  # parent frame
            )

            #MKs = MarkerArray()
            #SN["footLoc"][:] *=0
            ROOT_MEA[:] = data.qpos[0:3]

            COM_error = ROOT_TAR - ROOT_MEA[:2]
            yaw_tar = -np.arctan2(COM_error[1], COM_error[0])/np.pi*180
            COM_error = np.array([COM_error[0],COM_error[1],0])
            COM_error = ROT_OFFSET @ COM_error

            vec1 = COM_error
            vec1 /= np.linalg.norm(vec1)
            vec2 = np.array([data.xmat[0,0], data.xmat[0,1],0])
            vec2 /= np.linalg.norm(vec2)
            cross_product = np.cross(vec1, vec2)[2]
            error_dist = np.linalg.norm(COM_error)
            angle_error = np.abs(cross_product)

            if error_dist>0.2 and angle_error<0.5:
                if cross_product>0.1:
                    YAW_OFFSET += 0.5
                elif cross_product<-0.1:
                    YAW_OFFSET -= 0.5
                print("YAW", YAW_OFFSET, angle_error)
            

            step_loc = np.array([SN["footLoc"][0],SN["footLoc"][1]])
            step_error = STEP_TAR - step_loc
            print("STEP", step_error)
            offsetX = step_error[0]*5
            offsetX = np.clip(offsetX, -0.4,0.4)
            offsetY = step_error[1]*0.2
            offsetY = np.clip(offsetY, -0.1,0.1)


            if angle_error<0.1 or angle_error>0.5:
                pass
                # SN["boffset"][0] = offsetX
                # SN["boffset"][1] = offsetY

            # point_marker = generate_marker(0,[[-step_loc_global[0],-step_loc_global[1],0],
            #                                     #[-SN["footLoc"][0],-SN["footLoc"][1],0],
            #                                     #[-ROOT_TAR[0],-ROOT_TAR[1],0],
            #                                     #-data.xpos[2:][4],
            #                                     -data.xpos[2:][9]],color=(0,0,1,1))
            # #MKs.markers.append(point_marker)
            # marker_pub.publish(point_marker)
            if True:
                depth = depth[::4,::4]
                height, width = depth.shape
                # Create a grid of pixel coordinates
                i, j = np.meshgrid(np.arange(width), np.arange(height), indexing='xy')+np.random.rand((width*height*2)).reshape(2,height,width)
                
                # Flatten the pixel coordinates
                K_inv = 1/calculate_Ky(FOVY,height)

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
                points = ROTC @ points.T
                points = points.T
                #points += np.random.randn(points.size).reshape(points.shape)*0.05
                point_cloud_msg = pc2.create_cloud(header, fields, points)
                point_cloud_pub.publish(point_cloud_msg)

                rate.sleep()
                cv2.waitKey(1)
            pass
        except Exception as e:
            print(e)
            break
            pass
    if not viewer.is_alive:
        break

viewer.close()
low_process.terminate()
print("Low Process Terminated")
high_process.terminate()
print("High Process Terminated")
top_process.terminate()
exit(0)