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

class Image_pub:

    def __init__(self):
        self.image_pub = rospy.Publisher("image", Image)

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/image_publisher_1648223419742641943/image_raw", Image, self.callback)

        self.image_sub = rospy.Subscriber("/image_publisher_1648235646332748084/image_raw", Image, self.callback)

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

        except CvBridgeError as e:
            print(e)

        kernel = np.ones((5, 5), np.float32) / 25
        dst = cv2.filter2D(cv_image, -1, kernel)

        cv2.imshow("Image window", cv_image)
        cv2.waitKey(3)

        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(dst, "bgr8"))
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
    main(sys.argv)
