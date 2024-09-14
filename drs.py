#!/usr/bin/env python

import pyrealsense2 as rs
import numpy as np
import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from threading import Thread
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
import std_msgs.msg
DEPTH = False
def publish_images():
    # Initialize ROS node
    rospy.init_node('realsense_publisher', anonymous=True)

    # Create publishers for the topics
    pub_camera1 = rospy.Publisher('/camera/infra1/image_rect_raw', Image, queue_size=10)
    pub_camera2 = rospy.Publisher('/camera/infra2/image_rect_raw', Image, queue_size=10)
    pub_depth = rospy.Publisher('/camera/depth/image_rect_raw', Image, queue_size=10)
    pointcloud_pub = rospy.Publisher('/point_cloud', PointCloud2, queue_size=10)
    # Set the publishing rate
    rate = rospy.Rate(20) # 30 Hz

    # Create a CvBridge object
    bridge = CvBridge()

    # Configure RealSense
    ctx = rs.context()
    device1 = ctx.devices[0]
    depth_sensor = device1.query_sensors()[0]
    laser_pwr = depth_sensor.get_option(rs.option.laser_power)
    serial_number = device1.get_info(rs.camera_info.serial_number)
    config = rs.config()
    config.enable_device(serial_number)
    config.enable_stream(rs.stream.infrared, 1)
    config.enable_stream(rs.stream.infrared, 2)
    config.enable_stream(rs.stream.depth)
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Start the pipeline
    pipe = rs.pipeline()
    profile = pipe.start(config)

    frame =0 
    depth_sensor.set_option(rs.option.laser_power, 0)

    pc = rs.pointcloud()

    ROT1 = np.array([[0, 0, 1], 
                     [0, 1, 0], 
                     [-1, 0, 0]])
    ROT2 = np.array([[1,0,0],
                    [0,0,1],
                    [0,-1,0]])
    ROT = ROT2@ROT1
    while not rospy.is_shutdown():
        frame += 1
        # Wait for frames
        frames = pipe.wait_for_frames()
        ros_time = rospy.Time.now()
        aligned_frames = align.process(frames)

        # Get individual 

        # Convert frames to numpy arrays
        depth = aligned_frames.get_depth_frame() 
        POINT_CLOUD = False
        if POINT_CLOUD:      
            pointcloud = pc.calculate(depth)
            pointcloud = np.asarray(pointcloud.get_vertices(3))[::3,::3,:].reshape(-1,3)
            pointcloud = pointcloud[pointcloud[:,2]>0]
            pointcloud = ROT.dot(pointcloud.T).T

            scale = 0.5
            pointcloud = pointcloud*scale
            header = std_msgs.msg.Header()
            header.stamp = ros_time
            header.frame_id = 'camera'  # Set the coordinate frame

            # Create the PointCloud2 message
            fields = [
                PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1),
            ]
            
            point_cloud_msg = point_cloud2.create_cloud(header, fields, pointcloud)

            # Publish the message
            pointcloud_pub.publish(point_cloud_msg)

        depth = np.asanyarray(depth.get_data())
        depth = cv2.resize(depth, (160, 90))
        depth[depth>10000] = 10000
        depth[depth<800] = 800
        #depth = cv2.GaussianBlur(depth, (21, 21), 0)
        image_msg = bridge.cv2_to_imgmsg((depth).astype(np.uint16), encoding="16UC1")
        pub_depth.publish(image_msg)
        
        f1 = aligned_frames.get_infrared_frame(1)
        f2 = aligned_frames.get_infrared_frame(2)
        f1 = np.asanyarray(f1.get_data())
        f2 = np.asanyarray(f2.get_data())
        ros_image1 = bridge.cv2_to_imgmsg(f1, encoding="mono8")
        ros_image2 = bridge.cv2_to_imgmsg(f2, encoding="mono8")
        ros_image1.header.stamp = ros_time
        ros_image2.header.stamp = ros_time
        print(ros_image1.header.stamp)
        # Publish the images
        pub_camera1.publish(ros_image1)
        pub_camera2.publish(ros_image2)

        height, width = depth.shape
        # Create a grid of pixel coordinates
        i, j = np.meshgrid(np.arange(width), np.arange(height), indexing='xy')+np.random.rand((width*height*2)).reshape(2,height,width)



        # Display the images (optional, for visualization)
        # cv2.imshow("Infrared 1", f1)
        # cv2.imshow("Infrared 2", f2)
        # cv2.imshow("Depth", depth_image)
        # cv2.waitKey(1)

        # Sleep to maintain the publishing rate
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_images()
    except rospy.ROSInterruptException:
        pass
