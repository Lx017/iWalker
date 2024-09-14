from sair_utils import *

class Body:
    def __init__(self, name: str, pos=[0,0,0], quat=[0,0,0,1], 
                 IPos=[0,0,0], IQuat=[0,0,0,0], mass: float=0, diagInertia=[0,0,0]):
        self.children: list[Body]= [] # list of children
        self.name = name
        self.mass = to_tensor(mass)
        self.IPos = to_tensor(IPos)
        self.IQuat = to_tensor(IQuat)
        self.quat = to_tensor(quat)
        self.diagInertia = to_tensor(diagInertia)
        self.pos = to_tensor(pos)
        self.joints: dict[str, Joint] = {}
        self.map: dict[Joint|Body, Joint]= {}
        self.paths: list[list[Joint,Body]] = []
        
    def add_joint(self, joint: "Joint"):
        self.joints[joint.name] = joint
   
    def search_path(self,path=[]):
        paths = []
        for key, joint in self.joints.items():
            joint: Joint
            if joint not in path:
                body = joint.get_opp(self)
                newPath = path+[joint,body]
                paths += body.search_path(newPath)

        if len(paths) == 0:
            return [path]
        return paths
    
    def generate_map(self):
        paths = self.search_path()
        self.paths = paths.copy()
        for path in paths:
            for joint in path:
                joint: Joint
                self.map[joint] = path[0]
        return self.map, paths

    def __repr__(self) -> str:
        return "Body "+self.name


class Joint:
    def __init__(self, name: str, body1: Body, body2:Body, loc2=[0,0,0], 
                 axis=[0,0,1], range= [-3.14, 3.14]):

        self.name = name
        axis_tensor = to_tensor(axis)
        axis_tensor.data = torch.nn.functional.normalize(axis_tensor, p=2, dim=0)
        self.axis = axis_tensor
        self.body1, self.body2 = body1, body2
        self.range = range
        # in bruce mujoco XML, loc2 should always be [0,0,0]
        # as body2's frame is defined first
        # However, in urdf, joint's loc defined body2's frame
        # in mujoco, body's parent is body
        # in urdf, body's parent is joint
        if loc2[0] != 0 or loc2[1] != 0 or loc2[2] != 0:
            raise ValueError("pos1 should be [0,0,0]")
        #body2.quat = tensor([0,0,0,1])
        self.loc1 = body2.pos #joint position in RB1's frame\
        self.quat1 = pp.SO3(body2.quat)
        #self.loc2 = TO_BE_IMPLEMENTED
        body1.add_joint(self)
        body2.add_joint(self)
        
    def get_opp(self, body: Body):
        if body == self.body1:
            return self.body2
        if body == self.body2:
            return self.body1
        raise ValueError("invalid input body")
    
    def get_joint_loc(self, body: Body):
        '''
        get the joint location relative to body's defined center, not body frame's origin
        '''
        if body == self.body1:
            return self.loc1, self.quat1
        if body == self.body2:
            raise NotImplementedError
        raise ValueError("invalid input body")

    
    def __repr__(self) -> str:
        return "Joint "+self.name
    
    
TO_BE_IMPLEMENTED = [0,0,0]

TAR_POS = tensor(0)
TAR_VEL = tensor(1)
TAR_ACC = tensor(2)
TAR_TORQ = tensor(3)
EST_POS = tensor(4)
EST_VEL = tensor(5)
EST_ACC = tensor(6)
EST_TORQ = tensor(7)
TAR_ALL = tensor([0, 1, 2, 3])
EST_ALL = tensor([4, 5, 6, 7])



class RobotManager(nn.Module):
    """
    A robot takes a chain of bodies and joints, representing a robot instance
    """
    def __init__(self, main: Body):
        self.bodies={main.name:main}
        self.joints={}
        self.joint_state_dict={}
        self.main=main
        import random
        self.id = int(random.randint(0,100000))
        self.mem = Robot_SM(self.id)

    def get_limb(self, limb_name: str) -> Joint:
        return self.bodies[limb_name]

    def __getitem__(self, key) -> Joint:
        return self.bodies[key]
    
    def get_transform(self):
        state = self.state[self.joints[self.name]]
        angle = state[0:1]
        w = torch.cos(angle / 2)
        axis = self.axis * torch.sin(angle / 2)
        data = torch.cat([self.pos, axis, w], dim=-1)
        return pp.SE3(data)
    
    def get_body_pos(self, limb_name: str, point: torch.Tensor):
        limb = self.get_limb(limb_name)
        return limb.get_global_transform() * point
    
    def XMLload(robotDict, parent=None):
        pos = robotDict["pos"] if "pos" in robotDict else "0 0 0"
        pos = toFloats(pos)

        quat = robotDict["quat"] if "quat" in robotDict else "1 0 0 0"
        quat = toFloats(quat)

        name = robotDict["name"]
        joint_dict = robotDict["joint"][0]

        inertial = robotDict["inertial"][0]
        if len(robotDict["inertial"]) > 1:
            print("Warning: More than one inertial element found")

        IPos = toFloats(inertial["pos"]) if "pos" in inertial else [0,0,0]
        IQuat = toFloats(inertial["quat"]) if "quat" in inertial else [1,0,0,0]
        mass = toFloats(inertial["mass"])
        diagInertia = toFloats(inertial["diaginertia"])

        quat.append(quat.pop(0))#change to xyzw
        IQuat.append(IQuat.pop(0))#change to xyzw
        obj = Body(name, pos, quat, IPos, IQuat, mass, diagInertia)

        if len(robotDict["joint"]) != 1:
            raise "Warning: joint not accepted"
        else:
            joint_name = joint_dict["name"]
            joint_type = joint_dict["type"] if "type" in joint_dict else "default"
            if joint_type != "free":
                joint_axis = toFloats(joint_dict["axis"])
                joint_range = toFloats(joint_dict["range"])
                joint_anchor = toFloats(joint_dict["pos"])
                Joint(joint_name, parent, obj, joint_anchor, joint_axis, joint_range)

        if "body" in robotDict:
            for child in robotDict["body"]:
                RobotManager.XMLload(child,obj)

        return obj
    
    def loadFromXML(xmlPath):
        xmlDict = mujoco_xml_to_dict(xmlPath)
        root = xmlDict["worldbody"][0]["body"][0]
        robot = RobotManager(RobotManager.XMLload(root))
        robot.setup()
        return robot
    
    def set_state(self, attr_idx, data: Tensor, idx_dict=None):
        with torch.no_grad():
            idx_dict: dict[str, Tensor]
            if idx_dict is None:
                # Set the state of the robot directly
                self.state[:,attr_idx] = data
                return
            
            
            valid_jnt_names = []
            valid_values = []
            for name, idx in idx_dict.items():
                # Set the state of the robot using an index dictionary
                if name in self.joints:
                    value = data[idx]
                    valid_jnt_names.append(name)
                    valid_values.append(value)
                    state_idx = self.joint_state_dict[name]
                    self.state[state_idx][attr_idx] = value
                else:
                    print_every("Warning: Joint not found in robot",10000)
            

            jnt_state_msg = JointState()
            jnt_state_msg.header.stamp = self.pub_node.get_clock().now().to_msg()
            jnt_state_msg.name = valid_jnt_names
            jnt_state_msg.position = valid_values
            self.state_publisher.publish(jnt_state_msg)
    
    def get_path(self, src_body_name: str, dst_body_name: str, print_path=False):
        src = self.bodies[src_body_name]
        dst = self.bodies[dst_body_name]
        path = [src]
        
        while src != dst:
            joint = src.map[dst]
            path.append(joint)
            src = joint.get_opp(src)
            path.append(src)

        if print_path:
            string = ""
            for part in path:
                string += str(part) + " -> "
            print(string)
        return path
    
    def forward(self,frame_body_name, ros_viz=False):
        #taking the frame_body as the root, 
        #calculate the kinematics in the frame_body's frame
        frame_body=self.bodies[frame_body_name]
        _t=time.time()
        for path in frame_body.paths:
            dst = path[-1]
            cur = frame_body
            while cur != dst:
                next_joint = cur.map[dst]
                next_body = next_joint.get_opp(cur)
                jnt_state = self.get_jnt_state(next_joint.name, EST_POS)

                jnt_loc, jnt_quat = next_joint.get_joint_loc(cur) #defined joint location relative to one end
                path_idx, buffer_idx,len = self.SE3_buffer_cfg[next_joint.name]
                buffer_block = self.SE3_buffer[path_idx,buffer_idx:len+buffer_idx]
                buffer_block[0,0:3] = jnt_loc
                buffer_block[0,3:7] = jnt_quat
                # print("quat", time.time()-_t)
                # _t = time.time()

                jnt_pos = jnt_state.unsqueeze(-1) 
                w = torch.cos(jnt_pos / 2)
                axis = next_joint.axis * torch.sin(jnt_pos / 2)
                buffer_block[1,3:6] = axis
                buffer_block[1,6:7] = w

                cur = next_body
        print("chain_forward", time.time()-_t)

        with torch.no_grad():
            #self.fw_buffer = self.SE3_buffer.clone()
            matrix = self.SE3_buffer.matrix()
            _t=time.time()
            # for i in range(1,matrix.shape[1]):
            #     matrix[:,i] = matrix[:,i-1] @ matrix[:,i]
            # self.fw_buffer = matrix
            self.fw_buffer = pp.cumprod(matrix,dim=1, left=False)
            #self.fw_buffer = frozen(self.fw_buffer)
            # for j in range(1,self.SE3_buffer.shape[1]):
            #     self.fw_buffer[:,j] = self.fw_buffer[:,j-1] @ self.fw_buffer[:,j]
            #self.fw_buffer = script_n(self.SE3_buffer)
            print("cummul", time.time()-_t)
        if ros_viz:
            self.vis_pub()

    def get_jnt_state(self, joint_name: str, attr: Tensor):
        return self.state[self.joint_state_dict[joint_name]][attr]
    
    def setup(self):
        _, paths = self.main.generate_map()#generate map from root, getting all joints and bodies
        jointIdx=0
        STATE_PER_JOINT = 8
        self.SE3_buffer_cfg={}
        max_row_len = 0
        for path, path_idx in zip(paths, range(len(paths))):
            ptr = 0
            for part, part_idx in zip(path, range(len(path))):
                if isinstance(part, Joint):
                    self.joints[part.name] = part
                    self.joint_state_dict[part.name] = jointIdx
                    self.SE3_buffer_cfg[part.name] = [path_idx,ptr,2]
                    ptr+=2
                    max_row_len = max(max_row_len,ptr)
                    self.mem.register(part.name, [STATE_PER_JOINT])
                    jointIdx+=1
                elif isinstance(part, Body):
                    self.bodies[part.name] = part
                    self.mem.register(part.name, [7])
                    part.generate_map()
                else:
                    raise ValueError("Unknown type in path")
                
        self.SE3_buffer=pp.SE3(torch.zeros((len(paths),max_row_len,7)))
        self.state=torch.zeros((jointIdx,STATE_PER_JOINT),requires_grad=True,dtype=torch.float32)
        self.mem.init()
        self.mem.set('elbow_pitch_link_l',tensor([0.1,0.1,0.1,0.1,0.1,0.1,0.1]))
        self.test_link_sm=Robot_SM(self.id)
        self.test_link_sm.init()
        self.test_link_sm.get('elbow_pitch_link_l')

        
    def vis_pub(self):
        marker_array = MarkerArray()

        marker_id = 0  # Unique ID for each marker
        arrawCfg = {
            "hip_roll_r": [0.2,[0.,1.,0.]],
            "hip_roll_l": [0.2,[0.,1.,0.]],
            "hip_yaw_r": [0.2,[0.,0.,1.]],
            "hip_yaw_l": [0.2,[0.,0.,1.]],
            "shoulder_pitch_r": [0.2,[1.,1.,0.]],
            "shoulder_pitch_l": [0.2,[1.,1.,0.]],
            "ankle_pitch_r": [0.5,[1.,0.,1.]],
            "ankle_pitch_l": [0.5,[1.,0.,1.]],
        }

        SE3_buffer = pp.mat2SE3(self.fw_buffer)
        for key in self.joints.keys():
            if key == "bot_base":
                continue
            i,j,k = self.SE3_buffer_cfg[key]
            value = self.mem.get(key)
            try:
                value = SE3_buffer[i,j+1]
            except:
                continue
            marker = Marker()
            marker.header.frame_id = "map"
            marker.type = marker.ARROW
            marker.action = marker.ADD
            marker.id = marker_id  # Assign unique ID
            marker_id += 1

            # Set the size of the cube
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.a = 1.0

            if key in arrawCfg:
                marker.scale.x = arrawCfg[key][0]
                marker.color.r = arrawCfg[key][1][0]
                marker.color.g = arrawCfg[key][1][1]
                marker.color.b = arrawCfg[key][1][2]
            else:
                marker.scale.x = 1.0
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0


            # Set position
            value = value.data
            posScale = 10
            marker.pose.position.x = value[0].item()*posScale
            marker.pose.position.y = value[1].item()*posScale
            marker.pose.position.z = value[2].item()*posScale

            # Set orientation for each cube individually
            # This is a placeholder for how you might calculate or retrieve the orientation
            # You'll need to replace this with your actual orientation data
            marker.pose.orientation = Quaternion(x=value[3].item(), 
                                                 y=value[4].item(), 
                                                 z=value[5].item(), 
                                                 w=value[6].item())

            marker_array.markers.append(marker)

        # Publish the MarkerArray
        self.world_publisher.publish(marker_array)

