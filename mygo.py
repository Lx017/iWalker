from Startups.reset_thread import *
from Startups.run_bear import bearMain
from Play.initialize import initMain
from Startups.run_estimation import estMain
from Play.Walking.low_level import lowMain
from Play.Walking.high_level import highMain
from Play.Walking.top_level import topMain
import multiprocessing as mp
import time
import os
import subprocess
import cv2
from snp import SharedNp
from Settings.Bruce import *
SN = SharedNp("shm1.json")
SN["IMUBodyRot"][:] = np.eye(3)
SN["RotCorrect"][:] = np.eye(3)
SN["yaw"][:] = 0
SN["boffset"][:] = 0

TEST_CONTROL = True

bot = BRUCE()
bear_process=mp.Process(target=bearMain,daemon=True)
init_process=mp.Process(target=initMain,daemon=True)
est_process=mp.Process(target=estMain,daemon=True)
low_process=mp.Process(target=lowMain,daemon=True)
high_process=mp.Process(target=highMain,daemon=True)
top_process=mp.Process(target=topMain,daemon=True)
os.system("clear")

if TEST_CONTROL:
    bear_process.start()
    time.sleep(2)

    init_process.start()

    time.sleep(1)

    est_process.start()
    time.sleep(3)

    low_process.start()

WALKING = False
#plannerP = subprocess.Popen("python3 my_iplanner.py",shell=True)

from scipy.spatial.transform import Rotation as R
import tf
import tf2_ros

import rospy
br = tf.TransformBroadcaster()
rospy.init_node("estimation")
from geometry_msgs.msg import Pose, PoseStamped, TransformStamped, PointStamped
from visualization_msgs.msg import Marker, MarkerArray
pose_publisher = rospy.Publisher("pose", PoseStamped, queue_size=10)
air_vo_pose = PoseStamped()
need_pose_update = False
update_pose_time = 0
Target_traj = None
listener = tf.TransformListener()
TARGET_POINT = None
smoothed_XOffset = 0
smoothed_YOffset = 0
def iplanner_callback(msg):
    global smoothed_XOffset,smoothed_YOffset
    global TARGET_POINT
    global Target_traj
    Target_traj = msg
    pts = msg.markers[1].points
    idx = 0
    TARGET_POINT = np.array([pts[idx].x,pts[idx].y,0])
    SN["yaw"][:] -= TARGET_POINT[1]*7 #rotate with y value left/right

    steps = msg.markers[0].points
    first_step = np.array([steps[0].x,steps[0].y,0])
    print("first step: ",first_step, "yaw: ",SN["yaw"])
    smoothed_XOffset = first_step[0]

def AirVO_callback(msg):
    global air_vo_pose,need_pose_update
    air_vo_pose = msg
    need_pose_update = True
    update_pose_time = msg.header.stamp
    trans = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
    #rotation_quaternion = tf.transformations.quaternion_from_euler(0, 0, 3.141592653589793)

    # Rotate the original quaternion by multiplying it with the rotation quaternion
    #rotated_quaternion = tf.transformations.quaternion_multiply([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w],rotation_quaternion)
    rotated_quaternion = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
    br.sendTransform(trans,rotated_quaternion,rospy.Time.now(),"camera","airvo")
AirVO_sub = rospy.Subscriber("/AirVO/frame_pose", PoseStamped, AirVO_callback)
iPlanner_sub = rospy.Subscriber("/iplanner_out", MarkerArray, iplanner_callback)

AirVO_initRot = None
camera_initRot = np.eye(3)
camera_initPos = np.array([0,0,0.4])
camera_init_VO_tf = None
IMU_initRot = None
control_window_array = np.zeros((480,640,3),dtype=np.uint8)
cv2.imshow("control_window",control_window_array)
recalibration = False
camera_init_VO_Quat = None
X_home = 0
Y_home = 0
X_offset = 0
Y_offset = 0 

calibrating = False
while True:
    try:
        poseStamped = PoseStamped()
        pose = Pose()
        ros_time = rospy.Time.now()
        bodyPos = bot[EST]["bodyPos"].copy()
        IMUbodyRot = SN["IMUBodyRot"].copy()
        if need_pose_update and np.linalg.norm(IMUbodyRot[0])>0.99:
            air_vo_pos = np.array([air_vo_pose.pose.position.x, air_vo_pose.pose.position.y, air_vo_pose.pose.position.z])
            if calibrating:
                print(str(air_vo_pose.pose.position.x)[:5],str(air_vo_pose.pose.position.y)[:5],str(X_home)[:5],str(Y_home)[:5])
                X_home += air_vo_pos[0]*0.03
                Y_home -= air_vo_pos[1]*0.03
            #print(air_vo_pos)
            SN["poseOffset"][:3] = air_vo_pos
            SN["poseOffset"][:3] -= bodyPos
            air_vo_quat = np.array([air_vo_pose.pose.orientation.x, air_vo_pose.pose.orientation.y, air_vo_pose.pose.orientation.z, air_vo_pose.pose.orientation.w])
            if np.linalg.norm(air_vo_quat) > 0.99 and np.linalg.norm(air_vo_quat) < 1.01:
                matrix = R.from_quat(air_vo_quat).as_matrix()
                if recalibration:
                    AirVO_initRot = matrix
                    camera_init_VO_Rot = matrix.T
                    IMU_initRot = IMUbodyRot
                    camera_init_VO_Quat = R.from_matrix(camera_init_VO_Rot).as_quat()
                    camera_init_VO_Trans = camera_initPos - air_vo_pos
                    rot_offset = matrix @ IMUbodyRot.T
                    quat = R.from_matrix(rot_offset).as_quat()
                    rotCorrectMat = (matrix @ AirVO_initRot.T) @ IMU_initRot @ IMUbodyRot.T
                    rotCorrectQuat = R.from_matrix(rotCorrectMat).as_quat()
                    prevRotCorrectQuat = R.from_matrix(SN["RotCorrect"]).as_quat()
                    smoothedQuat = rotCorrectQuat# * 0.5 + prevRotCorrectQuat * 0.5
                    SN["RotCorrect"][:] = R.from_quat(smoothedQuat).as_matrix()
                    recalibration = False
            need_pose_update = False

        # if camera_init_VO_Quat is not None:
        #     offset = rospy.Duration(-5)  # 5 seconds
        #     br.sendTransform(camera_init_VO_Trans,camera_init_VO_Quat,ros_time,"airvo","map")
            
        pose.position.x = bodyPos[0]
        pose.position.y = bodyPos[1]
        pose.position.z = bodyPos[2]
        quat = R.from_matrix(SN["RotCorrect"] @ IMUbodyRot).as_quat()
        pose.orientation.x = quat[0]
        pose.orientation.y = quat[1]
        pose.orientation.z = quat[2]
        pose.orientation.w = quat[3]
        poseStamped.pose = pose
        poseStamped.header.stamp = ros_time
        poseStamped.header.frame_id = "camera"
        pose_publisher.publish(poseStamped)

        rospy.sleep(0.01)
        key = cv2.waitKey(1)
        key = chr(key & 255)
        if key == 'q':
            SN["yaw"][:] -= 1
            print(SN['yaw'])
        elif key == 'e':
            SN["yaw"][:] += 1
        elif key == "w":
            X_offset = 0.04
        elif key == "s":
            X_offset = -0.04
        elif key == "a":
            Y_home += 0.0015
            #Y_offset = 0.005
        elif key == "d":
            Y_home -= 0.0015
            #Y_offset = -0.005
        elif key == "f":
            X_offset = 0
            Y_offset = 0
        elif key == "c":
            calibrating = not calibrating
            print("calibrating: ",calibrating)
        elif key == "g":
            Y_home += 0.001
            print("XY_calibrate: ",X_home,Y_home)
        elif key == "j":
            Y_home -= 0.001
            print("XY_calibrate: ",X_home,Y_home)
        elif key == "h":
            X_home -= 0.004
            print("XY_calibrate: ",X_home,Y_home)
        elif key == "y":
            X_home += 0.004
            print("XY_calibrate: ",X_home,Y_home)
        elif key == ' ':
            recalibration = True
            control_window_array[:,:] = [0,255,0]

            if not WALKING and TEST_CONTROL:
                high_process.start()
                time.sleep(1)
                top_process.start()
                WALKING = True
            cv2.imshow("control_window",control_window_array)
            print("forward")
        elif key == 'r':
            break
        
        smoothed_XOffset = smoothed_XOffset * 0.95 + X_offset * 0.05
        smoothed_YOffset = smoothed_YOffset * 0.95 + Y_offset * 0.05
        
        SN["boffset"][0] = X_home + smoothed_XOffset
        SN["boffset"][1] = Y_home + smoothed_YOffset
    except KeyboardInterrupt:
        break


# dxl_process.join()
# print("dxl thread end!")
if TEST_CONTROL:
    bear_process.kill()
    low_process.kill()
    high_process.kill()
    top_process.kill()
#plannerP.kill()
print("bear thread end!")
# init_process.join()
# print("init thread end!")
# est_process.join()
# print("est thread end!")
#p5.join()
