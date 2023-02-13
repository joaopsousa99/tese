#! /usr/bin/python3
from __future__ import print_function
import itertools
from math import (sin, cos, atan2, sqrt)
import cv2
import numpy as np
import rospy
from cv_bridge import (CvBridge, CvBridgeError)
from geometry_msgs.msg import Point, PoseStamped
from scipy.spatial import distance
from sensor_msgs.msg import Image


processing = False
new_msg = False
msg = None


class targetDetector:
    def __init__(self):
        self.frameCounter = 0
        self.targetCounter = 0

        self.bridge = CvBridge()

        self.imageSub = rospy.Subscriber("pylon_camera_node/image_rect", Image, self.callback)
        self.targetCentrePub = rospy.Publisher("target/centre", Point, queue_size=1)

        self.ellipseDebugger = rospy.Publisher("target/ellipse_debugging", Image, queue_size=1)
        self.ringDebugger = rospy.Publisher("target/ring_debugging", Image, queue_size=1)
        self.targetDebugger = rospy.Publisher("target/target_debugging", Image, queue_size=1)

        self.poseSub = rospy.Subscriber("mavros/local_position/pose", PoseStamped, self.poseCallback)

        self.BINARIZATION_THRESH = 127
        self.FITTED_ELLIPSE_AREA_TOLERANCE = 5
        self.CONCENTRIC_TOLERANCE = 10
        self.ECCENTRIC_SIMILARITY_TOLERANCE = 0.2
        self.ANGLE_DIFF_TOLERANCE = 2
        self.ECCENTRIC_TOLERANCE = 0.1
        self.BLACK_TOLERANCE_PERCENT = 0.30
        self.OUTER_RADII_TOLERANCE = 0.1
        self.INNER_RADII_TOLERANCE = 0.1
        self.SMALL_RADII_TOLERANCE = 0.1

        self.FOCAL_DISTANCE = (1942.8336 + 1936.85487)/2
        self.WIDTH_IN_MM = 500
        self.OUTER_RADII_RATIO_IDEAL = 0.8
        self.INNER_RADII_RATIO_IDEAL = 0.6
        self.SMALL_RADII_RATIO_IDEAL = 1/3 

        self.relAltMeters = 0.0

    def ros2cv(self, ros_img_msg, cv_format):
        try:
            return self.bridge.imgmsg_to_cv2(ros_img_msg, cv_format)
        except CvBridgeError as e:
            print(e)

    def callback(self, data):
        global processing, new_msg, msg

        if not processing:
            new_msg = True
            msg = data

    def msg_processing(self, msg):
        self.frameCounter = self.frameCounter + 1

        imgGray = self.ros2cv(msg, "mono8")
        if imgGray is None:
            print("ERROR: img is None")
            self.targetCentrePub.publish(Point(0, 0, -1))
        else:
            imgGray = cv2.bilateralFilter(imgGray, 5, 175, 175)
            imgMax = np.amax(imgGray)
            
            imgGray = np.uint8((imgGray - np.amin(imgGray))/(imgMax - np.amin(imgGray))*255)

            threshType = cv2.THRESH_BINARY + cv2.THRESH_OTSU
            _, th = cv2.threshold(imgGray, self.BINARIZATION_THRESH, imgMax, threshType)

            contours, _ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            ellipses = self.ellipseFilter(contours, imgGray)

            if len(ellipses) < 2:
                self.targetCentrePub.publish(Point(0, 0, -1))
                return

            rings = self.ringDetector(imgGray, contours, ellipses)

            if len(rings) < 1:
                self.targetCentrePub.publish(Point(0, 0, -2))
                return

            targetCentre = self.targetDetector(rings, imgGray)
            if not (targetCentre.x == 0 and targetCentre.y == 0):
                targetCentre.z = self.relAltMeters
            
            self.targetCentrePub.publish(targetCentre)

            if targetCentre.x != 0 and targetCentre.y != 0:
                self.targetCounter = self.targetCounter + 1

    def ellipseFilter(self, contours, debugimg):
        nContours = len(contours)
        ellipses = []
        debugimg = cv2.merge((debugimg, debugimg, debugimg))

        for c in range(nContours):
            if len(contours[c]) > 4:
                ((x, y), (a, b), phi) = cv2.fitEllipse(contours[c])

                fittedEllipseArea = a*b*np.pi/4
                minContourArea = fittedEllipseArea*(1 - self.FITTED_ELLIPSE_AREA_TOLERANCE/100)
                maxContourArea = fittedEllipseArea*(1 + self.FITTED_ELLIPSE_AREA_TOLERANCE/100)

                # verifica se a área do contorno está entre as áreas mínima e máxima
                contourArea = cv2.contourArea(contours[c])
                greaterThanMinArea = contourArea > minContourArea
                lessThanMaxArea = contourArea < maxContourArea

                if greaterThanMinArea and lessThanMaxArea:
                    ellipses.append((x, y, a, b, phi, c, np.amax(np.array([a,b]))))
                    debugimg = cv2.ellipse(debugimg, (int(x), int(y)), (int(a/2), int(b/2)), phi, 0, 360, (0, 255, 0), 2)
                elif not (np.isnan(x) or np.isnan(y) or np.isnan(a) or np.isnan(b) or np.isnan(phi)):
                    debugimg = cv2.ellipse(debugimg, (int(x), int(y)), (int(a/2), int(b/2)), phi, 0, 360, (0, 0, 255), 2)

        self.ellipseDebugger.publish(self.bridge.cv2_to_imgmsg(debugimg, encoding="passthrough"))
        return (ellipses)

    def ringDetector(self, imgGray, contours, ellipses):
        imgMax = np.amax(imgGray)
        imgMin = np.amin(imgGray)
        dimg = imgGray.copy()
        dimg = cv2.merge((dimg, dimg, dimg))

        ellipseCombinations = itertools.combinations(ellipses, 2)
        rings = []

        for indexInEllipseCombinations, pair in enumerate(ellipseCombinations):
            c1  = (pair[0][0], pair[0][1])
            c2  = (pair[1][0], pair[1][1])
            a   = (pair[0][2], pair[1][2])
            b   = (pair[0][3], pair[1][3])
            phi = (pair[0][4], pair[1][4])
            c   = (pair[0][5], pair[1][5])

            if distance.euclidean(c1, c2) >= self.CONCENTRIC_TOLERANCE:
                continue

            # excentricidade = √(b²-a²)/b, assumindo que b≥a
            eccentricity = (sqrt(b[0]**2 - a[0]**2)/b[0],
                            sqrt(b[1]**2 - a[1]**2)/b[1])
            eccentricSimilarity = abs(eccentricity[0] - eccentricity[1])
            notSimilarlyEccentric = eccentricSimilarity > self.ECCENTRIC_SIMILARITY_TOLERANCE

            if notSimilarlyEccentric:
                continue

            if eccentricSimilarity <= self.ECCENTRIC_TOLERANCE:
                angleDiff = 0
            else:
                angleDiff = abs(atan2(sin(phi[0] - phi[1]), cos(phi[0] - phi[1])))

            anglesAreTooDifferent = angleDiff > self.ANGLE_DIFF_TOLERANCE

            if anglesAreTooDifferent:
                continue

            radiiRatioA = a[0]/a[1]
            radiiRatioB = b[0]/b[1]

            outerRadiiRatioFlagA = abs(radiiRatioA - self.OUTER_RADII_RATIO_IDEAL) < self.OUTER_RADII_TOLERANCE
            outerRadiiRatioFlagB = abs(radiiRatioB - self.OUTER_RADII_RATIO_IDEAL) < self.OUTER_RADII_TOLERANCE
            isOuterRingCandidate = outerRadiiRatioFlagA and outerRadiiRatioFlagB

            innerRadiiRatioFlagA = abs(radiiRatioA - self.INNER_RADII_RATIO_IDEAL) < self.INNER_RADII_TOLERANCE
            innerRadiiRatioFlagB = abs(radiiRatioB - self.INNER_RADII_RATIO_IDEAL) < self.INNER_RADII_TOLERANCE
            isInnerRingCandidate = innerRadiiRatioFlagA and innerRadiiRatioFlagB

            smallRadiiRatioFlagA = abs(radiiRatioA - self.SMALL_RADII_RATIO_IDEAL) < self.SMALL_RADII_TOLERANCE
            smallRadiiRatioFlagB = abs(radiiRatioB - self.SMALL_RADII_RATIO_IDEAL) < self.SMALL_RADII_TOLERANCE
            isSmallRingCandidate = smallRadiiRatioFlagA and smallRadiiRatioFlagB

            ringFlags = (isOuterRingCandidate, isInnerRingCandidate, isSmallRingCandidate)

            if sum(ringFlags) != 1:
                continue

            ringPointsList = []
            _, _, widthInPixels, _ = cv2.boundingRect(contours[c[1]])

            blackThreshold = self.BLACK_TOLERANCE_PERCENT * (imgMax - imgMin) + imgMin

            ringMask = np.zeros_like(imgGray)
            ringContours = [contours[c[0]], contours[c[1]]]
            cv2.drawContours(ringMask, ringContours, -1, color=255, thickness=-1)

            ringPoints = np.where(ringMask == 255)

            ringPointsList.append(imgGray[ringPoints[0], ringPoints[1]])
            ringPointsList = np.array(ringPointsList[0])

            nBlackPoints = np.count_nonzero(ringPointsList < blackThreshold)
            blackPercentage = nBlackPoints/len(ringPointsList)*100

            if blackPercentage <= 80:
                continue

            ringEllipsesCentroid = ((c1[0] + c2[0])/2, (c1[1] + c2[1])/2)
            meanRingOrientation = (phi[0] + phi[1])/2

            # retorna o índica da flag que está a True
            #   0 - anel exterior
            #   1 - anel interior
            #   2 - anel pequeno
            typeOfRing = [i for i, x in enumerate(ringFlags) if x][0]

            if typeOfRing == 0:
                biggestRadius = np.amax(np.array([a[0], b[0], a[1], b[1]]))
            else:
                biggestRadius = 0

            rings.append((indexInEllipseCombinations, typeOfRing, ringEllipsesCentroid, meanRingOrientation, ringContours, biggestRadius))

            dimg = cv2.drawContours(dimg, [contours[c[0]], contours[c[1]]], -1, (0, 255, 0), -1)

        self.ringDebugger.publish(self.bridge.cv2_to_imgmsg(dimg, encoding="passthrough"))
        return rings

    def targetDetector(self, rings, img):
        # rings = (indexInEllipseCombinations, typeOfRing, ringEllipsesCentroid, meanRingOrientation, ringContours)
        debugImg = img.copy()
        debugImg = cv2.merge((debugImg, debugImg, debugImg))
        
        radius = int(max(min(-30.7692307692*self.relAltMeters + 265.384615385, 250), 25))

        if len(rings) == 1:
            x = rings[0][2][0]
            y = rings[0][2][1]
            z = 0
            targetCentre = Point(x, y, z)

            cv2.line(debugImg, (0, int(y)), (1920, int(y)), (255, 0, 0), 2)
            cv2.line(debugImg, (int(x), 0), (int(x), 1080), (255, 0, 0), 2)
            print(f"biggestRadius = {rings[0][5]}", end=" ")

        elif len(rings) == 2:
            ringsAreDifferent = rings[0][1] != rings[1][1]
            ringsAreNotValid = rings[0][1] == 0 and rings[1][1] == 2 or rings[0][1] == 2 and rings[1][1] == 0

            if ringsAreDifferent and not ringsAreNotValid:
                x = (rings[0][2][0] + rings[1][2][0])/2
                y = (rings[0][2][1] + rings[1][2][1])/2
                z = 0
                targetCentre = Point(x, y, z)

                cv2.line(debugImg, (0, int(y)), (1920, int(y)), (255, 0, 0), 2)
                cv2.line(debugImg, (int(x), 0), (int(x), 1080), (255, 0, 0), 2)

                print(f"biggestRadius = {rings[0][5]} {rings[1][5]}", end=" ")
            else:
                print("2 anéis e têm o mesmo tipo ou são o exterior e o pequeno")
                targetCentre = Point(0, 0, -1)

        elif len(rings) == 3:
            ringsAreDifferent = rings[0][1] != rings[1][1] and rings[0][1] != rings[2][1] and rings[1][1] != rings[2][1]

            if ringsAreDifferent:
                x = (rings[0][2][0] + rings[1][2][0] + rings[2][2][0])/3
                y = (rings[0][2][1] + rings[1][2][1] + rings[2][2][1])/3
                z = 0
                targetCentre = Point(x, y, z)

                cv2.line(debugImg, (0, int(y)), (1920, int(y)), (255, 0, 0), 2)
                cv2.line(debugImg, (int(x), 0), (int(x), 1080), (255, 0, 0), 2)

                print(f"biggestRadius = {rings[0][5]} {rings[1][5]} {rings[2][5]}", end=" ")
            else:
                print("3 anéis e não são todos diferentes")
                targetCentre = Point(0, 0, -1)

        else:
            targets = []

            for ringCombo in itertools.combinations(rings, 2):
                ringType   = (ringCombo[0][1], ringCombo[1][1])
                ringCentre = (ringCombo[0][2], ringCombo[1][2])
                angles     = (ringCombo[0][3], ringCombo[1][3])

                if ringType[0] == ringType[1]:
                    print("4+ anéis e há pelos menos 2 iguais")
                    continue

                if distance.euclidean(ringCentre[0], ringCentre[1]) >= self.CONCENTRIC_TOLERANCE:
                    print("4+ anéis e há pelos menos 2 não concêntricos")
                    continue

                angleDiff = abs(atan2(sin(angles[0] - angles[1]), cos(sin(angles[0] - angles[1]))))

                if angleDiff > self.ANGLE_DIFF_TOLERANCE:
                    print("4+ anéis e há pelos menos 2 com orientações diferentes")
                    continue

                targets.append(ringCombo)

            for ringCombo in itertools.combinations(rings, 3):
                ringType = (ringCombo[0][1], ringCombo[1][1], ringCombo[2][1])
                ringCentre = (ringCombo[0][2], ringCombo[1][2], ringCombo[2][2])
                angles = (ringCombo[0][3], ringCombo[1][3], ringCombo[2][3])

                allDifferentTypes = ringType[0] != ringType[1] and ringType[0] != ringType[2] and ringType[1] != ringType[2]

                if not allDifferentTypes:
                    print("4+ anéis e há pelo menos 2 não concêntricos")
                    continue

                dist01 = distance.euclidean(ringCentre[0], ringCentre[1])
                dist02 = distance.euclidean(ringCentre[0], ringCentre[2])
                dist12 = distance.euclidean(ringCentre[1], ringCentre[2])
                concentric = dist01 >= self.CONCENTRIC_TOLERANCE and dist02 >= self.CONCENTRIC_TOLERANCE and dist12 >= self.CONCENTRIC_TOLERANCE

                if not concentric:
                    print("4+ anéis e há pelos menos 2 não concêntricos")
                    continue

                angleDiff01 = abs(atan2(sin(angles[0] - angles[1]), cos(sin(angles[0] - angles[1]))))
                angleDiff02 = abs(atan2(sin(angles[0] - angles[2]), cos(sin(angles[0] - angles[2]))))
                angleDiff12 = abs(atan2(sin(angles[1] - angles[2]), cos(sin(angles[1] - angles[2]))))
                anglesAreEqual = angleDiff01 > self.ANGLE_DIFF_TOLERANCE and angleDiff02 > self.ANGLE_DIFF_TOLERANCE and angleDiff12 > self.ANGLE_DIFF_TOLERANCE

                if not anglesAreEqual:
                    print("4+ anéis e há pelos menos 2 com orientações diferentes")
                    continue

                targets.append(ringCombo)

            if len(targets) >= 1:
                if len(targets) > 1:
                    cv2.imwrite("/home/jp/IMAGE.JPG", img)

                targetCentreX = (targets[0][0][2][0] + targets[0][1][2][0])/2
                targetCentreY = (targets[0][0][2][1] + targets[0][1][2][1])/2

                targetCentre = Point(targetCentreX, targetCentreY, 0)
                cv2.line(debugImg, (0, int(targetCentreY)),
                         (1280, int(targetCentreY)), (255, 0, 0), 2)
                cv2.line(debugImg, (int(targetCentreX), 0),
                         (int(targetCentreX), 720), (255, 0, 0), 2)
                print(f"biggestRadius = aaaaaaaaaaa")
            else:
                print(f"biggestRadius = bbbbbbbbbbb")
                targetCentre = Point(0, 0, -1)

        if "targetCentre" in locals():
            if distance.euclidean((targetCentre.x, targetCentre.y), (640, 360)) > radius:
                cv2.circle(debugImg, (640, 360), radius, (0, 0, 255), 10)
            else:
                cv2.circle(debugImg, (640, 360), radius, (0, 255, 0), 10)
        else:
            cv2.circle(debugImg, (640, 360), radius, (0, 255, 0), 10)

        self.targetDebugger.publish(self.bridge.cv2_to_imgmsg(debugImg, encoding="passthrough"))

        return targetCentre

    def poseCallback(self, data):
        self.relAltMeters = data.pose.position.z


def main():
    global processing, new_msg, msg

    rospy.init_node("target", anonymous=True)
    rate = rospy.Rate(30)
    td = targetDetector()

    while not rospy.is_shutdown():
        if new_msg:
            processing = True
            new_msg = False
            td.msg_processing(msg)
            rate.sleep()
            processing = False


if __name__ == '__main__':
    main()
