#!/usr/bin/env python
from __future__ import print_function

import roslib

import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
import std_msgs.msg
import math
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
import sensor_msgs.point_cloud2 as pcl2
import files_management
import depth


class Image_pub:

    def __init__(self):
        self.output_file = None
        self.cv_image1 = None
        self.image_pub = rospy.Publisher("depthimage", Image, queue_size=5)
        self.pcl_pub = rospy.Publisher("my_pcl_topic", PointCloud2, queue_size=100)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(sys.argv[1], Image,
                                          self.image1_callback)

        self.image_sub = rospy.Subscriber(sys.argv[2], Image, self.image2_callback)
        # self.image_sub = rospy.Subscriber("topic1", Image, self.image1_callback)

        # self.image_sub = rospy.Subscriber("topic2", Image, self.image2_callback)

    def image1_callback(self, data):
        try:
            self.cv_image1 = self.bridge.imgmsg_to_cv2(data, "bgr8")

        except CvBridgeError as e:
            print(e)

    def image2_callback(self, data):
        global cv_image2
        try:
            cv_image2 = self.bridge.imgmsg_to_cv2(data, "bgr8")

        except CvBridgeError as e:
            print(e)

        p_left = np.array([[1., 0., 3.1969315809942782e+02, 0.],
                           [0., 1., 1.7933051140233874e+02, 0.],
                           [0., 0., 1., 0.]])
        p_right = np.array([[1., 0., 3.1969315809942782e+02, 0.],
                            [0., 1., 1.7933051140233874e+02, 4.0590762653907342e-01],
                            [0., 0., 1., 0.]])

        # Compute the disparity map using the function above
        rospy.logwarn("This is right image type %s" % str(type(self.cv_image1)))
        rospy.logwarn("This is left image type %s" % str(type(cv_image2)))

        disp_left = depth.compute_left_disparity_map(self.cv_image1, cv_image2)

        k_left, r_left, t_left = depth.decompose_projection_matrix(p_left)
        k_right, r_right, t_right = depth.decompose_projection_matrix(p_right)

        depth_map_left = depth.calc_depth_map(disp_left, k_left, t_left, t_right)

        # plt.figure(figsize=(8, 8), dpi=100)
        # plt.imshow(depth_map_left, cmap='flag')
        # plt.show()

        h, w = self.cv_image1.shape[:2]
        focal_length = 0.8 * w
        # Perspective transformation matrix
        Q = np.float32([[1, 0, 0, -w / 2.0],
                        [0, -1, 0, h / 2.0],
                        [0, 0, 0, -focal_length],
                        [0, 0, 1, 0]])

        points_3D = cv2.reprojectImageTo3D(disp_left, Q)

        colors = cv2.cvtColor(self.cv_image1, cv2.COLOR_BGR2RGB)
        mask_map = disp_left > disp_left.min()
        output_points = points_3D[mask_map]
        output_colors = colors[mask_map]
        output_file = 'output.ply'

        depth.create_output(output_points, output_colors, output_file)
        vertices = np.hstack(
            [output_points.reshape(-1, 3), np.ones((output_points.shape[0], 1), dtype='float'), output_colors])
        np.save('points_rgb', vertices)

        rospy.loginfo("Initializing sample pcl2 publisher node...")
        # give time to roscore to make the connections
        rospy.sleep(1.)

        fields = [PointField('x', 0, PointField.FLOAT32, 1), PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1), PointField('intensity', 12, PointField.FLOAT32, 1),
                  PointField('r', 16, PointField.FLOAT32, 1), PointField('g', 20, PointField.FLOAT32, 1),
                  PointField('b', 24, PointField.FLOAT32, 1)]

        header = std_msgs.msg.Header()
        header.frame_id = "map"
        header.stamp = rospy.Time.now()
        pc2 = point_cloud2.create_cloud(header, fields, vertices)

        self.pcl_pub.publish(pc2)
        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(depth_map_left, "32FC1"))

        except CvBridgeError as e:
            print(e)


def main(args):
    ic = Image_pub()
    rospy.init_node('image', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv[1:])
