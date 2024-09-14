from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Quaternion
import rospy
import tf
import numpy as np
import random


def vector_to_quaternion(vector):
    """
    Converts a 3D vector into a quaternion for use in ROS markers.

    Args:
        vector (np.ndarray): 3D vector representing the direction of the arrow.

    Returns:
        Quaternion: The quaternion corresponding to the arrow's orientation.
    """
    # Normalized direction vector
    norm_vector = vector / np.linalg.norm(vector)

    # Assuming the vector represents direction, we can compute the rotation
    # For simplicity, we rotate the vector [1, 0, 0] (x-axis) to the desired vector
    rotation_axis = np.cross([1, 0, 0], norm_vector)
    rotation_axis_norm = np.linalg.norm(rotation_axis)

    if rotation_axis_norm != 0:
        rotation_axis = rotation_axis / rotation_axis_norm
        angle = np.arccos(np.dot([1, 0, 0], norm_vector))  # Angle between the two vectors

        # Quaternion from angle and axis
        quat = tf.transformations.quaternion_about_axis(angle, rotation_axis)
    else:
        # If the vector is already aligned with the x-axis, no rotation is needed
        quat = [0, 0, 0, 1]  # Identity quaternion
    
    return Quaternion(*quat)

def generate_ArrowMarker(id, point, vector, color=(1.0, 0.0, 0.0, 1.0), scale=0.1):
    """
    Generates an Arrow Marker message to visualize a 3D vector.

    Args:
        id (int): ID of the marker.
        point (np.ndarray): Starting point of the arrow (shape: 3,).
        vector (np.ndarray): Direction vector of the arrow (shape: 3,).
        color (tuple): RGBA tuple for the color of the arrow.
        scale (float): Scale of the arrow.

    Returns:
        Marker: The generated arrow marker message.
    """
    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "arrow_" + str(id)
    marker.id = id
    marker.type = Marker.ARROW
    marker.action = Marker.ADD

    # Set the scale of the arrow (length and diameter)
    marker.scale.x = np.linalg.norm(vector) * scale  # Length of the arrow
    marker.scale.y = scale * 0.1  # Arrow width
    marker.scale.z = scale * 0.1  # Arrow height

    # Set the color of the arrow
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = color[3]

    # Set the starting point (position of the arrow)
    start = Point()
    start.x = point[0]
    start.y = point[1]
    start.z = point[2]
    marker.pose.position = start

    # Set the orientation (converted from vector to quaternion)
    marker.pose.orientation = vector_to_quaternion(vector)

    return marker


def generate_3DPointMarker(id, points, color=(1.0, 0.0, 0.0, 1.0), scale=0.1):
    """
    Generates a Marker message from a set of 3D points.

    Args:
        points (np.ndarray): Nx3 array of 3D points.
        frame_id (str): The frame ID to attach the marker.
        marker_id (int): The ID of the marker.
        color (tuple): RGBA tuple for the color of the points.
        scale (float): Scale of the points.

    Returns:
        Marker: The generated marker message.
    """

    POS_SCALE = 1
    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "points_"+str(id)
    marker.id = id
    marker.type = Marker.SPHERE_LIST
    marker.action = Marker.ADD

    # Set the scale of the points
    marker.scale.x = scale
    marker.scale.y = scale
    marker.scale.z = scale

    # Set the color of the points (RGBA)
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = color[3]

    # Add points to the marker
    for point in points:
        pt = Point()
        if hasattr(point, 'item'):
            pt.x = point[0].item() * POS_SCALE
            pt.y = point[1].item() * POS_SCALE
            if len(point) == 2:
                pt.z = 0
            else:
                pt.z = point[2].item() * POS_SCALE
        else:
            pt.x = point[0] * POS_SCALE
            pt.y = point[1] * POS_SCALE
            pt.z = point[2] * POS_SCALE
        marker.points.append(pt)

    return marker

def matrix_to_quaternion(matrix):
    """
    Converts a 3x3 rotation matrix to a quaternion.
    """
    quat = tf.transformations.quaternion_from_matrix(np.vstack((np.hstack((matrix, np.array([[0], [0], [0]]))), np.array([0, 0, 0, 1]))))
    return quat

def RotMatZ(angle_degrees):
    """
    Creates a rotation matrix for a rotation around the Z-axis.

    Parameters:
    angle_degrees (float): The rotation angle in degrees.

    Returns:
    np.ndarray: The 3x3 rotation matrix.
    """
    angle_radians = np.radians(angle_degrees)
    cos_a = np.cos(angle_radians)
    sin_a = np.sin(angle_radians)
    
    rotation_matrix = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ])
    
    return rotation_matrix


def calculate_Ky(fovy,height):
    """
    Calculate the intrinsic camera matrix K.

    Parameters:
    - fovy (float): Field of view in the y-direction (in degrees).
    - width (int): Width of the image.
    - height (int): Height of the image.

    Returns:
    - K (numpy.ndarray): Intrinsic camera matrix.
    """
    # Convert fovy from degrees to radians
    fovy_rad = np.deg2rad(fovy)

    # Calculate focal length in the y-direction
    fy = height / (2 * np.tan(fovy_rad / 2))

    return fy

def quaternion_multiply(q1, q2):
    """
    Multiplies two quaternions in xyzw format.

    Parameters:
    q1, q2: Lists or arrays of quaternion components [x, y, z, w]

    Returns:
    A list representing the product quaternion in [x, y, z, w] format.
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    # Quaternion multiplication formula
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return [x, y, z, w]
