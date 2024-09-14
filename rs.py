#!/usr/bin/env python

import pyrealsense2 as rs
import numpy as np
import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from threading import Thread

DEPTH = False
def publish_images():
    # Initialize ROS node
    rospy.init_node('realsense_publisher', anonymous=True)

    # Create publishers for the topics
    pub_camera1 = rospy.Publisher('/camera/infra1/image_rect_raw', Image, queue_size=10)
    pub_camera2 = rospy.Publisher('/camera/infra2/image_rect_raw', Image, queue_size=10)
    pub_depth = rospy.Publisher('/camera/depth/image_rect_raw', Image, queue_size=10)

    # Set the publishing rate
    rate = rospy.Rate(30) # 30 Hz

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
    if not DEPTH:
        depth_sensor.set_option(rs.option.laser_power, 0)
    while not rospy.is_shutdown():
        frame += 1
        # Wait for frames
        frames = pipe.wait_for_frames()
        aligned_frames = align.process(frames)

        # Get individual 

        # Convert frames to numpy arrays
        if frame % 2 == 0 and DEPTH:

            depth = aligned_frames.get_depth_frame()
            depth = np.asanyarray(depth.get_data())
            depth = cv2.resize(depth, (160, 90))
            depth[depth>10000] = 10000
            depth[depth<800] = 800
            depth = cv2.GaussianBlur(depth, (21, 21), 0)
            image_msg = bridge.cv2_to_imgmsg((depth).astype(np.uint16), encoding="16UC1")
            pub_depth.publish(image_msg)
            
            depth_sensor.set_option(rs.option.laser_power, 0)
        else:
            # Convert numpy arrays to ROS Image messagesframes
            f1 = aligned_frames.get_infrared_frame(1)
            f2 = aligned_frames.get_infrared_frame(2)
            f1 = np.asanyarray(f1.get_data())
            f2 = np.asanyarray(f2.get_data())
            ros_image1 = bridge.cv2_to_imgmsg(f1, encoding="mono8")
            ros_image2 = bridge.cv2_to_imgmsg(f2, encoding="mono8")
            ros_image1.header.stamp = rospy.Time.now()
            ros_image2.header.stamp = rospy.Time.now()
            print(ros_image1.header.stamp)
            # Publish the images
            pub_camera1.publish(ros_image1)
            pub_camera2.publish(ros_image2)
            if DEPTH:
                depth_sensor.set_option(rs.option.laser_power, 100)

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
