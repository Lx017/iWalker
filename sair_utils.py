import torch
import numpy as np
import pypose as pp
from torch import Tensor,tensor,nn
import xml.etree.ElementTree as ET
import json
# import rclpy
# from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Quaternion, Transform
from sensor_msgs.msg import JointState
import numpy as np
from threading import Thread
from multiprocessing import shared_memory
import time
import scipy.sparse as sp
import osqp

#rclpy.init()
from geometry_msgs.msg import PointStamped, PoseStamped
#from builtin_interfaces.msg import Duration

#markerPubNode = rclpy.create_node("xpos_publisher")
import os

class SharedMemoryManager:
    def __init__(self, shm_name):

        self.shm_name = shm_name
        self.shm = None

        with open(self.shm_name, 'r') as f:
            self.metadata = json.load(f)

        if os.path.exists(self.shm_name) == False:
            raise FileNotFoundError(f"Shared memory metadata file {self.shm_name} not found.")

        total_size = sum(meta['size'] for meta in self.metadata.values())
        
        try:
            self.shm = shared_memory.SharedMemory(name=self.shm_name, create=True, size=total_size)
        except FileExistsError:
            self.shm = shared_memory.SharedMemory(name=self.shm_name, create=False)
            if self.shm.size < total_size:
                self.shm.close()
                self.shm.unlink()
                self.shm = shared_memory.SharedMemory(name=self.shm_name, create=True, size=total_size)
                print("Shared memory buffer already exists. Overwriting...")

        self.buffer = self.shm.buf[:total_size]

    def register(self, arrays_dict):
        offset = 0
        for name, arr in arrays_dict.items():
            size = arr.nbytes
            self.metadata[name] = {
                'shape': arr.shape,
                'dtype': str(arr.dtype),
                'offset': offset,
                'size': size
            }
            offset += size
        
        # Write metadata to JSON file
        with open(self.shm_name, 'w') as f:
            json.dump(self.metadata, f)

    def write(self, arrays_dict):
        if self.shm is None:
            # Calculate total size needed for shared memory buffer
            total_size = sum(arr.nbytes for arr in arrays_dict.values())

            # Create a shared memory buffer

            self.buffer = self.shm.buf[:total_size]
            
        else:
            # Update existing shared memory with new array data
            for name, arr in arrays_dict.items():
                if name in self.metadata:
                    meta = self.metadata[name]
                    offset = meta['offset']
                    size = arr.nbytes
                    if size > meta['size']:
                        raise ValueError(f"New array size for {name} exceeds allocated space in shared memory.")
                    # Copy array data to shared memory
                    self.buffer[offset:offset+size] = arr.tobytes()
                else:
                    raise KeyError(f"Array name {name} not found in existing metadata.")

            
    
    def read(self):
        """
        Load numpy arrays from the shared memory buffer.
        
        :return: Dictionary of numpy arrays recreated from shared memory
        """
        
        # Recreate numpy arrays from shared memory
        arrays_dict = {}
        for name, meta in self.metadata.items():
            arrays_dict[name] = self.get_RW_array(name).copy()
        
        return arrays_dict
    
    def get_RW_array(self,name):
        meta = self.metadata[name]
        shape = tuple(meta['shape'])
        dtype = np.dtype(meta['dtype'])
        offset = meta['offset']
        size = meta['size']
        
        # Create numpy array backed by shared memory buffer
        array = np.ndarray(shape, dtype=dtype, buffer=self.buffer[offset:offset+size])
        return array


def ArrowMarker(pos,vec,color=[1.,1.,1.]):

    force2quat, mag = vector_to_quat(-vec)
    return {"x":pos[0],
                        "y":pos[1],
                        "z":pos[2],
                            "type": Marker.ARROW,
                            "sx":mag*0.3,
                            "sy":0.1,
                            "sz":0.1,
                            "quat":force2quat,
                            "r":color[0],"g":color[1],"b":color[2],}

def shmArr(name, shape, dtype=np.float32, reset=False):
    try:
        shm = shared_memory.SharedMemory(name=name, create=False)
    except:
        typeLen = 1
        if dtype==np.uint8:
            typeLen = 1
        elif dtype==np.float32:
            typeLen = 4
        elif dtype==np.float64:
            typeLen = 8
        elif dtype==np.int32:
            typeLen = 4
        shm = shared_memory.SharedMemory(name=name, create=True, size=np.prod(shape)*typeLen)
    arr = np.frombuffer(shm.buf, dtype=dtype).reshape(shape)
    if reset:
        arr[:] = 0
    return arr

I_SE3 = pp.SE3(tensor([0,0,0,0,0,0,1], dtype=torch.float32, requires_grad=False))

def to_tensor(data,grad=False):
    return torch.tensor(data, dtype=torch.float32, requires_grad=grad)

def toFloats(str:str):
    list = str.split(" ")
    for i in range(len(list)):
        list[i] = float(list[i])

    if len(list)==1:
        return list[0]
    
    return list

print_record={}
def print_every(msg,interval=100):
    if msg in print_record:
        print_record[msg]+=1
        if print_record[msg]%interval==0:
            print(msg)
    else:
        print_record[msg]=1
        print(msg)

def parse_element(element):
    """
    Recursively parse an XML element and convert it into a dictionary.
    """
    parsed_element = {}
    if element.text and element.text.strip():
        parsed_element['text'] = element.text.strip()

    for key, value in element.attrib.items():
        parsed_element[key] = value

    children = list(element)
    if children:
        child_dict = {}
        for child in children:
            child_name = child.tag
            child_parsed = parse_element(child)
            if child_name not in child_dict:
                child_dict[child_name] = []
            child_dict[child_name].append(child_parsed)
        parsed_element.update(child_dict)

    return parsed_element

def mujoco_xml_to_dict(xml_path):
    """
    Convert a MuJoCo XML file to a Python dictionary.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    parsed_root = parse_element(root)
    return parsed_root


class Robot_SM:
    
    def __init__(self,name) -> None:
        self.registry = {}
        self.len = 0
        self.name = str(name)

    def register(self, name: str, shape):
        self.registry[name] = [int(self.len), shape]# ensure int type
        self.len += np.prod(shape)

    def init(self):
        try:# link
            self.shm_dict = shared_memory.SharedMemory(name=self.name+"_reg")
            sm_reg_arr = np.frombuffer(self.shm_dict.buf, dtype=np.uint8)
            dict_str = sm_reg_arr.tobytes().decode()
            self.registry = json.loads(dict_str)
            self.shm = shared_memory.SharedMemory(name=self.name)
            self.tensor = torch.frombuffer(self.shm.buf, dtype=torch.float32)
        except:# create
            registry_bytestr = json.dumps(self.registry).encode()
            registry_arr  = np.frombuffer(registry_bytestr, dtype=np.uint8)
            self.shm_dict = shared_memory.SharedMemory(name=self.name+"_reg", create=True, size=registry_arr.size)
            sm_reg_arr = np.frombuffer(self.shm_dict.buf, dtype=np.uint8)
            sm_reg_arr[:] = registry_arr[:]# write registry to shared memory
            self.shm = shared_memory.SharedMemory(name=self.name, create=True, size=self.len*4)
            self.tensor = torch.frombuffer(self.shm.buf, dtype=torch.float32)
            self.tensor[:] = -123

    def get(self, key):
        start = self.registry[key][0]
        shape = self.registry[key][1]
        size = int(np.prod(shape))
        end = start+size
        return self.tensor[start:end].reshape(shape)
    
    def set(self, key, data:Tensor):
        start = self.registry[key][0]
        shape = self.registry[key][1]
        size = int(np.prod(shape))
        end = start+size
        self.tensor[start:end] = data.data.flatten()



def vector_to_quat(v):
    """
    Given a 3D unit vector, calculates the quaternion (x, y, z, w) that rotates
    the x-axis to align with this vector.
    
    Parameters:
    - v: Target 3D unit vector as a numpy array or list.
    
    Returns:
    - Quaternion as a numpy array [x, y, z, w].
    """
    # Ensure v is a normalized unit vector
    mag = np.linalg.norm(v)
    v = v / mag
    
    # x-axis unit vector
    x_axis = np.array([1, 0, 0])
    
    # Calculate the angle between the x-axis and the vector v
    dot_product = np.dot(x_axis, v)
    angle = np.arccos(dot_product)
    
    # Calculate the rotation axis (cross product of x-axis and v)
    rotation_axis = np.cross(x_axis, v)
    if np.linalg.norm(rotation_axis) < 1e-6:
        # If the rotation_axis is very small, it means the vectors are parallel/anti-parallel
        if dot_product > 0:
            # Parallel vectors - no rotation needed
            return np.array([0, 0, 0, 1]), mag
        else:
            # Anti-parallel vectors - 180 degrees rotation around any perpendicular axis
            # Here we choose y-axis for simplicity
            return np.array([0, 1, 0, 0]), mag
    
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    
    # Calculate the quaternion components
    qx = rotation_axis[0] * np.sin(angle / 2)
    qy = rotation_axis[1] * np.sin(angle / 2)
    qz = rotation_axis[2] * np.sin(angle / 2)
    qw = np.cos(angle / 2)

    quat = np.array([qx, qy, qz, qw])

    if np.isnan(quat).any():
        quat = np.array([0., 0., 0., 1.])
    
    return quat, mag

def createSMArr(name, shape, dtype=np.float32):
    try:
        shm = shared_memory.SharedMemory(name=name, create=False)
    except:
        shm = shared_memory.SharedMemory(name=name, create=True, size=np.prod(shape)*4)
    arr = np.frombuffer(shm.buf, dtype=dtype).reshape(shape)
    arr[:] = 0
    return arr

class SM:
    leg_pos_mea = createSMArr("leg_pos_mea", (10,))
    leg_vel_mea = createSMArr("leg_vel_mea", (10,))
    leg_trq = createSMArr("leg_trq", (10,))

# print(SM.leg_pos_mea)
# print(SM.leg_trq)
    

class VAI:
    def __init__(self,in_filter=0.8,out_filter=0.8):
        self.in_filter = in_filter
        self.out_filter = out_filter
        self.reset()

    def reset(self):
        self.last_pos = None
        self.last_vel = 0
        self.last_acc = 0
        self.last_out = None

    def measure(self, pos_mea, dt):
        if self.last_pos is None:
            self.last_pos = pos_mea

        self.pos_mea = pos_mea.copy()

        self.vel_mea = (pos_mea - self.last_pos) / dt
        self.vel_smooth = self.last_vel*(1-self.in_filter) + self.vel_mea*self.in_filter

        self.acc_mea = (self.vel_mea - self.last_vel) / dt
        self.acc_smooth = self.last_acc*(1-self.in_filter) + self.acc_mea*self.in_filter

        self.last_vel = self.vel_smooth
        self.last_acc = self.acc_smooth
        self.last_pos = self.pos_mea

        return self.vel_smooth, self.acc_smooth
    
    def update(self, pos_tar, vel_tar, pos_mea, dt):
        pos_error = pos_tar - pos_mea
        pos_error = np.clip(pos_error, -1, 1)#assume 1 radian is the maximum error, improve stability
        
        vel,acc = self.measure(pos_mea,dt)

        pos_comp = dt * pos_error #+ 0.1 * (0-vel)*0

        vel_error = vel_tar-vel
        vel_comp = np.clip(np.random.normal(200,0,1)*vel_error*dt, -1000*0.001, 1000*0.001)

        acc_tar = 0# + 0.05 * vel_error/dt
        acc_error = acc_tar-acc
        acc_comp = np.clip(np.random.normal(2,0,1)*acc_error*dt, -1000*0.001, 1000*0.001)

        # if self.last_out is None:
        #     self.last_out = acc_comp+vel_comp

        error={
            "pos_error":pos_error,
            "vel_error":vel_tar-vel,
        }

        return pos_comp, acc_comp,vel_comp,error
    
def star_matrix(vec3):
    """
    Given a 3D vector, returns the skew-symmetric matrix representation of the vector.
    
    Parameters:
    - vec3: A 3D vector as a numpy array or list.
    
    Returns:
    - The skew-symmetric matrix representation of the vector as a numpy array.
    """
    return np.array([
        [0, -vec3[2], vec3[1]],
        [vec3[2], 0, -vec3[0]],
        [-vec3[1], vec3[0], 0]
    ])

class ForceTorqueDecompositor:
    def __init__(self):
        self.osqp = osqp.OSQP()
        self.ifSetup = False
        self.lastRes = None
        

    def solve(self, targetForce, targetTorque, positions):
        # forceNum = len(positions)
        # P = np.eye(forceNum)
        # P_ = sp.csc_matrix(P)
        # target = np.ones(4)
        # target[1:4] = targetTorque
        # M = np.stack([np.cross(pos, targetForce) for pos in positions]).T
        # Q = M.T @ M
        # p = (M.T @ targetTorque).T
        # A = np.concatenate([np.ones((1,forceNum)), M])
        # A_ = sp.csc_matrix(A)
        # if self.ifSetup:
        #     self.osqp.update(Ax=A_.data,l=target,u=target)
        # else:
        #     self.ifSetup = True
        #     self.osqp.setup(P=P_, A=A_, l=target, u=target,verbose=False)

        # res = self.osqp.solve()
        # if res.x[0] is None:
        #     print("no solution")
        #     return [0*targetForce for i in range(forceNum)]
        
        # # quadCost = res.x @ Q @ res.x + 2* p @ res.x + np.dot(targetTorque, targetTorque)
        # # print("quadCost:::::::::",quadCost)
        # return [res.x[i]*targetForce for i in range(forceNum)]
    

        # OSQP data preparation
        forceNum = len(positions)
        target = np.ones(1)
        #target[1:4] = targetTorque
        M = np.stack([np.cross(pos, targetForce) for pos in positions]).T
        Q = M.T @ M
        q = - M.T @ targetTorque
        P = Q
        P_ = sp.csc_matrix(P)
        A = np.ones((1,forceNum))
        #A = np.concatenate([np.ones((1,forceNum)), M])
        A_ = sp.csc_matrix(A)
        if self.ifSetup:
            self.osqp.update(Px=P_.data,q=q,Ax=A_.data,l=target, u=target)
        else:
            self.ifSetup = True
            self.osqp.setup(P=P_, q=q, A=A_, l=target, u=target,verbose=True)

        res = self.osqp.solve()
        if res.x[0] is None:
            print("no solution")
            return self.lastRes
        
        quadCost = res.x @ Q @ res.x + 2* q.T @ res.x + np.dot(targetTorque, targetTorque)
        print("quadCost",quadCost)
        self.lastRes = [res.x[i]*targetForce for i in range(forceNum)]
        return self.lastRes
    
class LeastSquareSolver:
    def __init__(self):
        self.osqp = osqp.OSQP()#recreate the object
        self.lastRes = None

    def solve(self,M, target,inputCost,targetCost,constraintMask=None):

        targetCost = np.array(targetCost)
        targetCostMat = np.diag(np.sqrt(targetCost))
        M = targetCostMat @ M
        P = M.T @ M + np.diag(inputCost)
        q = - M.T @ target
        P_ = sp.csc_matrix(P)

        lb = -np.inf*np.ones(len(inputCost))
        ub = np.inf*np.ones(len(inputCost))

        if constraintMask is not None:
            lb[constraintMask==0] = 0
            ub[constraintMask==0] = 0


        
            
        if self.osqp.if_setup:
            P_ = sp.triu(P_)
            self.osqp.update(Px=P_.data,q=q)
        else:
            A = np.eye(len(inputCost))
            A_ = sp.csc_matrix(A)
            self.osqp.setup(P=P_,  q=q, A=A_,l=lb,u=ub, verbose=False)


        res = self.osqp.solve().x
        self.lastRes = res

        pqObjective =0.5*res @ P @ res + q @ res
        quadCost = 2*pqObjective + np.dot(target, target)
        #print("quadCost",quadCost)

        # if quadCost > 10:
        #     self.osqp = osqp.OSQP()#recreate the object
        #     self.osqp.setup(P=P_, q=q, verbose=True, warm_start=False,adaptive_rho=False)
        #     res = self.osqp.solve().x
        #     quadCost = res @ P @ res + 2* q @ res + np.dot(target, target)

        return res, quadCost
    

def find_position_vector(F, T):
    # Calculate the magnitude squared of the force vector
    F_mag_squared = np.dot(F, F)
    
    # Check if the force vector is zero
    if F_mag_squared == 0:
        raise ValueError("The force vector F cannot be zero.")
    
    # Calculate the cross product of T and F
    T_cross_F = np.cross(T, F)
    
    # Calculate the position vector r
    r = T_cross_F / F_mag_squared
    
    return r

class ForceAssigner:
    def __init__(self):
        self.lastRes = None
        

    def solve(self, targetForce, targetTorque, positions):
        RES = 100
        torqueJac = np.stack([np.cross(pos, targetForce) for pos in positions])
        res = [0 for pos in positions]
        count = RES
        unitTorque = torqueJac/RES
        curTorque = np.zeros(3)
        while count>0:
            count-=1
            results = curTorque + unitTorque
            l2dist = np.linalg.norm(results-targetTorque,axis=1)
            argmin = np.argmin(l2dist)
            #print(l2dist,"argmin",argmin)
            res[argmin] += 1
            curTorque = results[argmin]

        return (np.array(res)/RES)[:,None] * targetForce[None]
        
class HardSolver:
    def __init__(self):
        self.osqp = osqp.OSQP()
        self.ifSetup = False
        

    def solve(self, A, target,costs,constraintMask=None):
        # OSQP data preparation
        target = target.copy()
        lb = -np.inf*np.ones_like(target)
        ub = np.inf*np.ones_like(target)
        if constraintMask is not None:
            constraintMask = np.array(constraintMask)
            lb[constraintMask==1] = target[constraintMask==1]
            ub[constraintMask==1] = target[constraintMask==1]
        else:
            lb[:] = target
            ub[:] = target
        A_ = sp.csc_matrix(A)
        if self.ifSetup:
            self.osqp.update(Ax=A_.data,l=lb, u=ub)
        else:
            P = np.diag(costs)
            P_ = sp.csc_matrix(P)
            self.ifSetup = True
            self.osqp.setup(P=P_, A=A_, l=lb, u=ub,verbose=False)

        P = A.T @ A
        q = - A.T @ target
        
        res = self.osqp.solve()

        pqObjective =0.5*res.x @ P @ res.x + q @ res.x
        quadCost = 2*pqObjective + np.dot(target, target)

        if res.x[0] is None:
            return "no solution"
        return res.x
    
def decompose_vector(v1, v2, v3, V):
    """
    Decompose the vector V into a linear combination of vectors v1, v2, and v3.
    
    Parameters:
    - v1, v2, v3: The vectors used for the decomposition (not necessarily unit vectors).
    - V: The target vector to decompose.
    
    Returns:
    - A tuple of coefficients (a, b, c) such that a*v1 + b*v2 + c*v3 = V.
    """
    # Create a matrix A consisting of vectors v1, v2, and v3
    A = np.column_stack((v1, v2, v3))
    
    # Ensure V is a proper column vector if not already
    V = np.array(V).reshape(-1, 1)
    
    # Solve the system of equations A * [a, b, c]^T = V for [a, b, c]
    coefficients = np.linalg.solve(A, V)
    
    # Return the coefficients as a flattened array (for easier handling)
    return coefficients

class MassPointSystem:
    def __init__(self, mass):
        self.mass = mass
        self.cur_pos = np.zeros(3)
        self.pos_hist = []

    def update_state(self, pos):
        self.cur_pos = pos
        self.pos_hist.append(pos)
        if len(self.pos_hist) > 1000:
            self.pos_hist.pop(0)

    def ros_viz():
        pass


publishers = {}
def publish_markers(topic, list,pos_scale=10):
    if topic not in publishers:
        publishers[topic] = markerPubNode.create_publisher(MarkerArray, topic, 10)
    publisher = publishers[topic]
    publish_markers_with_publisher(publisher, list,pos_scale)

def publish_markers_with_publisher(publisher, list,pos_scale=10):
    markerArray = MarkerArray()
    marker_id = 0
    for marker_data in list:
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = marker_data["type"]
        marker.action = marker.ADD
        marker.scale.x = float(marker_data["sx"]) if "sx" in marker_data else 0.15
        marker.scale.y = float(marker_data["sy"]) if "sy" in marker_data else 0.15
        marker.scale.z = float(marker_data["sz"]) if "sz" in marker_data else 0.15
        marker.color.a = float(marker_data["a"]) if "a" in marker_data else 1.
        marker.color.r = float(marker_data["r"]) if "r" in marker_data else 0.
        marker.color.g = float(marker_data["g"]) if "g" in marker_data else 0.
        marker.color.b = float(marker_data["b"]) if "b" in marker_data else 0.
        marker.id = marker_id  # Assign unique ID
        marker_id += 1
        marker.pose.position.x = marker_data["x"]*pos_scale
        marker.pose.position.y = marker_data["y"]*pos_scale
        marker.pose.position.z = marker_data["z"]*pos_scale
        if "quat" in marker_data:
            marker.pose.orientation.x = float(marker_data["quat"][0])
            marker.pose.orientation.y = float(marker_data["quat"][1])
            marker.pose.orientation.z = float(marker_data["quat"][2])
            marker.pose.orientation.w = float(marker_data["quat"][3])
        markerArray.markers.append(marker)
    publisher.publish(markerArray)

def publish_point_traj(topic ,marker_data,pos_scale=10):
    if topic not in publishers:
        publishers[topic] = markerPubNode.create_publisher(PointStamped, topic, 10)
    publisher = publishers[topic]

    stampPoint = PointStamped()
    stampPoint.header.frame_id = "map"
    stampPoint.point.x = marker_data["x"]*pos_scale
    stampPoint.point.y = marker_data["y"]*pos_scale
    stampPoint.point.z = marker_data["z"]*pos_scale
    publisher.publish(stampPoint)

hist_dict={}
def record_and_publish_hist(topic, marker_data,max_hist_len=100):
    if topic not in hist_dict:
        hist_dict[topic] = []
    hist_list = hist_dict[topic]
    hist_list.append(marker_data)
    if len(hist_list) > max_hist_len:
        hist_list.pop(0)
    publish_markers(topic,hist_list)

def publish_traj(topic, traj, color=[1.,1.,1.]):
    market_list = []
    for i in range(traj.shape[0]):
        marker = {"x":traj[i,0],"y":traj[i,1],"z":traj[i,2],
                                "type": Marker.SPHERE,
                                "sx":0.05,
                                "sy":0.05,
                                "sz":0.05,
                                "r":color[0],"g":color[1],"b":color[2],}
        market_list.append(marker)
    publish_markers(topic,market_list)
