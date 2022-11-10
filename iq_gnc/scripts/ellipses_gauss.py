#! /usr/bin/python3
from __future__ import print_function
import itertools
from math import (sin, cos, atan2, sqrt)
import cv2
import numpy as np
import rospy
from cv_bridge import (CvBridge, CvBridgeError)
from geometry_msgs.msg import Point, PoseStamped
from iq_gnc.PrintColours import *
from iq_gnc.py_gnc_functions import *
from scipy.spatial import distance
from sensor_msgs.msg import Image


# hack que encontrei no youtube que salta mensagens
# enquanto não terminar de processar a atual
processing = False
new_msg = False
msg = None


class targetDetector:
    def __init__(self):
        self.frameCounter = 0
        self.targetCounter = 0
        # subscritores e publicadores
        self.bridge = CvBridge()
        # self.imageSub = rospy.Subscriber("camera/image_raw", Image, self.callback)
        self.imageSub = rospy.Subscriber("pylon_camera_node/image_rect", Image, self.callback)
        self.targetCenterPub = rospy.Publisher("target/center", Point, queue_size=1)
        self.ellipseDebugger = rospy.Publisher("target/ellipse_debugging", Image, queue_size=1)
        self.ringDebugger = rospy.Publisher("target/ring_debugging", Image, queue_size=1)
        self.targetDebugger = rospy.Publisher("target/target_debugging", Image, queue_size=1)

        # self.poseSub = rospy.Subscriber("mavros/global_position/local", Odometry, self.poseCallback)
        self.poseSub = rospy.Subscriber("mavros/local_position/pose", PoseStamped, self.poseCallback)

        # parâmetros
        self.BINARIZATION_THRESH = 127
        self.AREA_TOLERANCE = 5
        self.CONCENTRIC_TOLERANCE = 10
        self.ECCENTRIC_SIMILARITY_TOLERANCE = 0.2
        self.ANGLE_DIFF_TOLERANCE = 2
        self.ECCENTRIC_TOLERANCE = 0.1
        self.BLACK_TOLERANCE_PERCENT = 0.30
        self.OUTER_RADII_TOLERANCE = 0.1
        self.INNER_RADII_TOLERANCE = 0.1
        self.SMALL_RADII_TOLERANCE = 0.1

        # valores reais
        # self.FOCAL_DISTANCE = 4.4160987654321
        # self.WIDTH_IN_MM = 202.5
        self.FOCAL_DISTANCE = (1942.8336 + 1936.85487)/2  # média de fx e fy
        # self.WIDTH_IN_MM = 160
        self.WIDTH_IN_MM = 500
        self.OUTER_RADII_RATIO_IDEAL = 400/500 # 0.8
        self.INNER_RADII_RATIO_IDEAL = 150/250 # 0.6
        self.SMALL_RADII_RATIO_IDEAL = 10/30   # 0.(3)

        self.relAlt = 0.0  # em metros

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
        # print("-" * 55)
        self.frameCounter = self.frameCounter + 1
        print(f'frame no. {self.frameCounter}')
        # converte a mensagem ROS Image para uma imagem OpenCV
        img = self.ros2cv(msg, "bayer_gbrg8")
        if img is None:
            print("ERROR: img is None")

        # o detetor de contornos trabalha sobre imagens de apenas um canal
        # pré-processamento para remover ruído
        imgGray = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2GRAY)
        imgGray = cv2.bilateralFilter(imgGray, 10, 175, 175)
        imgMax = np.amax(imgGray)

        # o detetor de contornos precisa de imagens binárias para funcionar bem
        # threshType = cv2.THRESH_BINARY + cv2.THRESH_OTSU
        # _, th = cv2.threshold(imgGray, (imgMax - np.amin(imgGray))/2+np.amin(imgGray), imgMax, threshType)
        # _, th = cv2.threshold(imgGray, self.BINARIZATION_THRESH, imgMax, threshType)
        th = cv2.adaptiveThreshold(imgGray, imgMax, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        # th = cv2.Canny(imgGray, 50, 150)

        contours, _ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        ellipses = self.ellipseFilter(contours, th)

        # o anel de fora tem 2 elipses
        # no entanto, o drone pode estar demasiado perto ou longe para detetá-las todas
        if len(ellipses) < 2:
            self.targetCenterPub.publish(Point(0, 0, -1))
            return

        # self.debugPub.publish(self.bridge.cv2_to_imgmsg(th, encoding="passthrough"))
        rings = self.ringDetector(imgGray, contours, ellipses)

        if len(rings) < 1:
            self.targetCenterPub.publish(Point(0, 0, -2))
            return

        targetCenter = self.targetDetector(rings, imgGray)
        if not (targetCenter.x == 0 and targetCenter.y == 0):
            targetCenter.z = self.relAlt
        
        self.targetCenterPub.publish(targetCenter)
        self.targetCounter = self.targetCounter + 1
        print(f'target no. {self.targetCounter}')

    def ellipseFilter(self, contours, img):
        nContours = len(contours)
        ellipses = []
        # debugimg = np.zeros_like(img)
        debugimg = img.copy()
        debugimg = cv2.merge((debugimg, debugimg, debugimg))

        for c in range(nContours):
            # se o contorno tiver mais de 4 pontos
            if len(contours[c]) > 4:
                # (centroide, eixos menor e maior, orientação)
                ((x, y), (a, b), phi) = cv2.fitEllipse(contours[c])
                # debugimg = cv2.putText(debugimg, f"{c}", (int(x+a/2), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

                # área da elipse fitted
                ellipseArea = a*b*np.pi/4
                # áreas mínima e máxima entre as quais a área do contorno
                # deve estar para ser considerado uma elipse
                minArea = ellipseArea*(1 - self.AREA_TOLERANCE/100)
                maxArea = ellipseArea*(1 + self.AREA_TOLERANCE/100)

                # verifica se a área do contorno está entre as áreas mínima e máxima
                contourArea = cv2.contourArea(contours[c])
                greaterThanMinArea = contourArea > minArea
                lessThanMaxArea = contourArea < maxArea

                if greaterThanMinArea and lessThanMaxArea:
                    # adiciona elipse à lista de candidatos
                    ellipses.append((x, y, a, b, phi, c))
                    debugimg = cv2.ellipse(debugimg, (int(x), int(y)), (int(
                        a/2), int(b/2)), phi, 0, 360, (0, 255, 0), 2)
                elif not (np.isnan(x) or np.isnan(y) or np.isnan(a) or np.isnan(b) or np.isnan(phi)):
                    debugimg = cv2.ellipse(debugimg, (int(x), int(y)), (int(
                        a/2), int(b/2)), phi, 0, 360, (0, 0, 255), 2)

        # for e in ellipses:
        #   print(f"x = {e[0]:.0f} y = {e[1]:.0f} a = {e[2]/ellipses[-1][2]*500:.0f} b = {e[3]/ellipses[-1][3]*500:.0f} phi = {e[4]:.0f} c = {e[5]}")

        self.ellipseDebugger.publish(
            self.bridge.cv2_to_imgmsg(debugimg, encoding="passthrough"))
        return (ellipses)

    def ringDetector(self, imgGray, contours, ellipses):
        imgMax = np.amax(imgGray)
        imgMin = np.amin(imgGray)
        dimg = imgGray.copy()
        dimg = cv2.merge((dimg, dimg, dimg))

        ellipseCombinations = itertools.combinations(ellipses, 2)
        rings = []

        # ciclo for para determinar que pares de elipses são anéis válidos
        for index, pair in enumerate(ellipseCombinations):
            c1 = (pair[0][0], pair[0][1])
            c2 = (pair[1][0], pair[1][1])
            a = (pair[0][2], pair[1][2])
            b = (pair[0][3], pair[1][3])
            phi = (pair[0][4], pair[1][4])
            c = (pair[0][5], pair[1][5])
            # dimg = cv2.drawContours(dimg, [contours[c[0]], contours[c[1]]], -1, (0,255,0), -1)

            # se as elipses do par não são concêntricas, salta para o próximo par
            if distance.euclidean(c1, c2) >= self.CONCENTRIC_TOLERANCE:
                # print(f"[{c[0]}, {c[1]}]: concentricidade")
                continue

            # excentricidade = √(b²-a²)/b, assumindo que b≥a
            eccentricity = (sqrt(b[0]**2 - a[0]**2)/b[0],
                            sqrt(b[1]**2 - a[1]**2)/b[1])
            eccentricSimilarity = abs(eccentricity[0] - eccentricity[1])
            notSimilarlyEccentric = eccentricSimilarity > self.ECCENTRIC_SIMILARITY_TOLERANCE

            # se as elipses do par não têm excentricidades parecidas, salta para o próximo par
            if notSimilarlyEccentric:
                # print(f"[{c[0]}, {c[1]}]: similaridade excêntrica")
                continue

            # já não me lembro do que é que isto faz
            if eccentricSimilarity <= self.ECCENTRIC_TOLERANCE:
                angleDiff = 0
            else:
                angleDiff = abs(
                    atan2(sin(phi[0] - phi[1]), cos(sin(phi[0] - phi[1]))))

            anglesAreTooDifferent = angleDiff > self.ANGLE_DIFF_TOLERANCE

            if anglesAreTooDifferent:
                # print(f"[{c[0]}, {c[1]}]: orientação")
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
            # print(abs(radiiRatioA - self.SMALL_RADII_RATIO_IDEAL))
            # print(abs(radiiRatioB - self.SMALL_RADII_RATIO_IDEAL))
            smallRadiiRatioFlagB = abs(radiiRatioB - self.SMALL_RADII_RATIO_IDEAL) < self.SMALL_RADII_TOLERANCE
            isSmallRingCandidate = smallRadiiRatioFlagA and smallRadiiRatioFlagB

            ringFlags = (isOuterRingCandidate, isInnerRingCandidate, isSmallRingCandidate)
            # print(ringFlags)

            # se o rácio dos raios das elipses do anel não for 4:5, 3:5 ou 1:2, salta para o próximo par
            if sum(ringFlags) != 1:
                # print(f"[{c[0]}, {c[1]}]: candidatos")
                continue

            ringPointsList = []
            _, _, widthInPixels, _ = cv2.boundingRect(contours[c[1]])
            # distanceToObject = self.FOCAL_DISTANCE*self.WIDTH_IN_MM/(widthInPixels*0.00122)
            distanceToObject = self.FOCAL_DISTANCE*self.WIDTH_IN_MM/widthInPixels

            # valor abaixo do qual um píxel é considerado preto
            blackThreshold = self.BLACK_TOLERANCE_PERCENT * \
                (imgMax - imgMin) + imgMin + 100

            # máscara que indica os píxeis dentro do anel
            ringMask = np.zeros_like(imgGray)
            ringContours = [contours[c[0]], contours[c[1]]]
            cv2.drawContours(ringMask, ringContours, -
                             1, color=255, thickness=-1)

            # coordenadas dos pontos dentro do anel
            ringPoints = np.where(ringMask == 255)

            # valores dos píxeis dentro do anel
            ringPointsList.append(imgGray[ringPoints[0], ringPoints[1]])
            ringPointsList = np.array(ringPointsList[0])

            # número de píxeis dentro do anel abaixo do threshold de preto
            nBlackPoints = np.count_nonzero(ringPointsList < blackThreshold)
            blackPercentage = nBlackPoints/len(ringPointsList)*100

            # resultado da regressão no desmos com BLACK_TOLERANCE_PERCENT = 0.3
            # o -5 representa uma tolerância de 5% na componente dc
            desmos = (318796, 5240.44, 40.8162 - 5)

            # se o anel não for preto o suficiente, salta para o próximo par
            # desmos[0]/(distanceToObject + desmos[1]) + desmos[2]:
            # print(blackPercentage)
            if blackPercentage <= 80:
                # print(f"[{c[0]}, {c[1]}]: preto")
                continue

            # neste ponto do filtro, o anel é válido tanto como anel exterior ou anel interior
            # o anel vai ser apended ao fim da lista de candidatos a anel do alvo
            # guarda-se:
            #   - o índice do par de elipses na lista ellipseCombinations
            #   - uma flag que indica se é exterior (True) ou interior (False)
            #   - o centroide das elipses do anel
            #   - a orientação média do anel
            #   - os contornos do anel

            ringEllipsesCentroid = ((c1[0] + c2[0])/2, (c1[1] + c2[1])/2)
            meanRingOrientation = (phi[0] + phi[1])/2

            # retorna o índica da flag que está a True
            #   0 - anel exterior
            #   1 - anel interior
            #   2 - anel pequeno
            typeOfRing = [i for i, x in enumerate(ringFlags) if x][0]

            rings.append((index, typeOfRing, ringEllipsesCentroid, meanRingOrientation, ringContours))

            dimg = cv2.drawContours(dimg, [contours[c[0]], contours[c[1]]], -1, (0, 255, 0), -1)
            # print(f"tipo {typeOfRing}\tcentro ({ringEllipsesCentroid[0]:.0f}, {ringEllipsesCentroid[1]:.0f})\traios ({(a[0]+b[0])/2:.0f}, {(a[1]+b[1])/2:.0f})\trácio {(a[0]+b[0])/(a[1]+b[1]):.3f}")

        self.ringDebugger.publish(
            self.bridge.cv2_to_imgmsg(dimg, encoding="passthrough"))
        return rings

    def targetDetector(self, rings, img):
        # rings = (index, typeOfRing, ringEllipsesCentroid, meanRingOrientation, ringContours)
        debugImg = img.copy()
        debugImg = cv2.merge((debugImg, debugImg, debugImg))
        radius = int(
            max(min(-30.7692307692*self.relAlt + 265.384615385, 250), 25))
        # print(f"int(max(min(-30*{self.relAlt:.2f} + 115, 100), 25)) = {radius}")

        # só há 1 anel e não é o exterior
        if len(rings) == 1:
            # ringIsNotOuter = rings[0][1] != 0

            # if ringIsNotOuter:
            #     # print("1")
            #     x = rings[0][2][0]
            #     y = rings[0][2][1]
            #     z = 0
            #     targetCenter = Point(x, y, z)
            #     # cv2.drawContours(debugImg, rings[0][4], -1, (0, 255, 0), -1)
            #     cv2.line(debugImg, (0, int(y)), (1920, int(y)), (255, 0, 0), 2)
            #     cv2.line(debugImg, (int(x), 0), (int(x), 1080), (255, 0, 0), 2)
            #     # cv2.circle(debugImg, (640, 360), radius, (0,255,0), 2)
            #     # print("DESENHEI")
            # else:
            #     print("1 anel e é o exterior")
            #     targetCenter = Point(0, 0, -1)

            x = rings[0][2][0]
            y = rings[0][2][1]
            z = 0
            targetCenter = Point(x, y, z)
            # cv2.drawContours(debugImg, rings[0][4], -1, (0, 255, 0), -1)
            cv2.line(debugImg, (0, int(y)), (1920, int(y)), (255, 0, 0), 2)
            cv2.line(debugImg, (int(x), 0), (int(x), 1080), (255, 0, 0), 2)

        # há dois anéis e têm tipos diferentes
        elif len(rings) == 2:
            ringsAreDifferent = rings[0][1] != rings[1][1]
            ringsAreNotValid = rings[0][1] == 0 and rings[1][1] == 2 or rings[0][1] == 2 and rings[1][1] == 0

            if ringsAreDifferent and not ringsAreNotValid:
                # print("2")
                x = (rings[0][2][0] + rings[1][2][0])/2
                y = (rings[0][2][1] + rings[1][2][1])/2
                z = 0
                targetCenter = Point(x, y, z)
                # cv2.drawContours(debugImg, rings[0][4], -1, (0, 255, 0), -1)
                # cv2.drawContours(debugImg, rings[1][4], -1, (0, 255, 0), -1)
                cv2.line(debugImg, (0, int(y)), (1920, int(y)), (255, 0, 0), 2)
                cv2.line(debugImg, (int(x), 0), (int(x), 1080), (255, 0, 0), 2)
                # cv2.circle(debugImg, (640, 360), radius, (0,255,0), 2)
                # print("DESENHEI")
            else:
                print("2 anéis e têm o mesmo tipo ou são o exterior e o pequeno")
                targetCenter = Point(0, 0, -1)

        # há três anéis e têm tipos diferentes
        elif len(rings) == 3:
            ringsAreDifferent = rings[0][1] != rings[1][1] and rings[0][1] != rings[2][1] and rings[1][1] != rings[2][1]

            if ringsAreDifferent:
                # print("3")
                x = (rings[0][2][0] + rings[1][2][0] + rings[2][2][0])/3
                y = (rings[0][2][1] + rings[1][2][1] + rings[2][2][1])/3
                z = 0
                targetCenter = Point(x, y, z)
                # print(f"{x}, {y}, {z}")
                # cv2.drawContours(debugImg, rings[0][4], -1, (0, 255, 0), -1)
                # cv2.drawContours(debugImg, rings[1][4], -1, (0, 255, 0), -1)
                cv2.line(debugImg, (0, int(y)), (1920, int(y)), (255, 0, 0), 2)
                cv2.line(debugImg, (int(x), 0), (int(x), 1080), (255, 0, 0), 2)
                # cv2.circle(debugImg, (640, 360), radius, (0,255,0), 2)
                # print("DESENHEI")
            else:
                print("3 anéis e não são todos diferentes")
                targetCenter = Point(0, 0, -1)

        else:
            # print("else")
            # neste caso, onde há mais de três anéis, há dois tipos de alvos possíveis:
            #   - alvos com dois anéis
            #   - alvos com três anéis
            # portanto, é preciso gerar combinações 2-a-2 e 3-a-3 e iterar sobre ambas
            ringCombinations = itertools.combinations(rings, 2)
            # print(len([rc for rc in ringCombinations]))
            targets = []

            # este ciclo for vai determinar que pares de anéis formam o alvo de aterragem
            for ringCombo in ringCombinations:
                ringType = (ringCombo[0][1], ringCombo[1][1])
                ringCenter = (ringCombo[0][2], ringCombo[1][2])
                angles = (ringCombo[0][3], ringCombo[1][3])

                # se os anéis forem do mesmo tipo (exterior e interior), salta para o próximo par
                if ringType[0] == ringType[1]:
                    # print("3.1")
                    # cv2.drawContours(debugImg, ringCombo[0][4], -1, (0, 0, 255), -1)
                    # cv2.drawContours(debugImg, ringCombo[1][4], -1, (0, 0, 255), -1)
                    print("4+ anéis e há pelos menos 2 iguais")
                    continue

                # se os anéis não foram concêntricos, salta para o próximo par
                if distance.euclidean(ringCenter[0], ringCenter[1]) >= self.CONCENTRIC_TOLERANCE:
                    # print("3.3")
                    # cv2.drawContours(debugImg, ringCombo[0][4], -1, (0, 0, 255), -1)
                    # cv2.drawContours(debugImg, ringCombo[1][4], -1, (0, 0, 255), -1)
                    print("4+ anéis e há pelos menos 2 não concêntricos")
                    continue

                # se os anéis não tiverem a mesma orientação, salta para o próximo par
                # diferença de ângulos = tan⁻¹(sin(α-β)/cos(α-β))
                angleDiff = abs(atan2(sin(angles[0] - angles[1]), cos(sin(angles[0] - angles[1]))))

                if angleDiff > self.ANGLE_DIFF_TOLERANCE:
                    # print("3.4")
                    # cv2.drawContours(debugImg, ringCombo[0][4], -1, (0, 0, 255), -1)
                    # cv2.drawContours(debugImg, ringCombo[1][4], -1, (0, 0, 255), -1)
                    print("4+ anéis e há pelos menos 2 com orientações diferentes")
                    continue

                # print(f"{ringCombo[0][0]} {ringCombo[1][0]} APPEND")
                targets.append(ringCombo)

            ringCombinations = itertools.combinations(rings, 3)

            for ringCombo in ringCombinations:
                ringType = (ringCombo[0][1], ringCombo[1][1], ringCombo[2][1])
                ringCenter = (ringCombo[0][2],
                              ringCombo[1][2], ringCombo[2][2])
                angles = (ringCombo[0][3], ringCombo[1][3], ringCombo[2][3])

                allDifferentTypes = ringType[0] != ringType[1] and ringType[0] != ringType[2] and ringType[1] != ringType[2]

                # se os anéis forem do mesmo tipo (exterior e interior), salta para o próximo trio
                if not allDifferentTypes:
                    # print(f"{ringCombo[0][0]} {ringCombo[1][0]} {ringCombo[2][0]} há anéis do mesmo tipo")
                    # cv2.imwrite("/home/jp/IMAGE.JPG", img)
                    # cv2.drawContours(debugImg, ringCombo[0][4], -1, (0, 0, 255), -1)
                    # cv2.drawContours(debugImg, ringCombo[1][4], -1, (0, 0, 255), -1)
                    print("4+ anéis e há pelo menos 2 não concêntricos")
                    continue

                # se os anéis não foram concêntricos, salta para o próximo trio
                dist01 = distance.euclidean(ringCenter[0], ringCenter[1])
                dist02 = distance.euclidean(ringCenter[0], ringCenter[2])
                dist12 = distance.euclidean(ringCenter[1], ringCenter[2])
                concentric = dist01 >= self.CONCENTRIC_TOLERANCE and dist02 >= self.CONCENTRIC_TOLERANCE and dist12 >= self.CONCENTRIC_TOLERANCE

                if not concentric:
                    # print(f"{ringCombo[0][0]} {ringCombo[1][0]} {ringCombo[2][0]} anéis não concêntricos")
                    # cv2.drawContours(debugImg, ringCombo[0][4], -1, (0, 0, 255), -1)
                    # cv2.drawContours(debugImg, ringCombo[1][4], -1, (0, 0, 255), -1)
                    print("4+ anéis e há pelos menos 2 não concêntricos")
                    continue

                # se os anéis não tiverem a mesma orientação, salta para o próximo par
                # diferença de ângulos = tan⁻¹(sin(α-β)/cos(α-β))
                angleDiff01 = abs(atan2(sin(angles[0] - angles[1]), cos(sin(angles[0] - angles[1]))))
                angleDiff02 = abs(atan2(sin(angles[0] - angles[2]), cos(sin(angles[0] - angles[2]))))
                angleDiff12 = abs(atan2(sin(angles[1] - angles[2]), cos(sin(angles[1] - angles[2]))))
                anglesAreEqual = angleDiff01 > self.ANGLE_DIFF_TOLERANCE and angleDiff02 > self.ANGLE_DIFF_TOLERANCE and angleDiff12 > self.ANGLE_DIFF_TOLERANCE

                if not anglesAreEqual:
                    # print(f"{ringCombo[0][0]} {ringCombo[1][0]} {ringCombo[2][0]} anéis com orientações diferentes")
                    # cv2.drawContours(debugImg, ringCombo[0][4], -1, (0, 0, 255), -1)
                    # cv2.drawContours(debugImg, ringCombo[1][4], -1, (0, 0, 255), -1)
                    print("4+ anéis e há pelos menos 2 com orientações diferentes")
                    continue

                targets.append(ringCombo)

            if len(targets) >= 1:
                if len(targets) > 1:
                    # print("MAIS QUE UM ALVO DETETADO")
                    cv2.imwrite("/home/jp/IMAGE.JPG", img)

                # assume-se que o primeiro par de anéis é o correto
                targetCenterX = (targets[0][0][2][0] + targets[0][1][2][0])/2
                targetCenterY = (targets[0][0][2][1] + targets[0][1][2][1])/2

                targetCenter = Point(targetCenterX, targetCenterY, 0)
                # cv2.drawContours(debugImg, ringCombo[0][4], -1, (0, 0, 255), -1)
                # cv2.drawContours(debugImg, ringCombo[1][4], -1, (0, 0, 255), -1)
                cv2.line(debugImg, (0, int(targetCenterY)),
                         (1280, int(targetCenterY)), (255, 0, 0), 2)
                cv2.line(debugImg, (int(targetCenterX), 0),
                         (int(targetCenterX), 720), (255, 0, 0), 2)
                # cv2.circle(debugImg, (640, 360), radius, (0,255,0), 2)
                # print(f"alvo detetado ({(targetCenterX/1280)*100:.2f}, {(targetCenterY/800)*100:.2f})")
            else:
                # print("4")
                targetCenter = Point(0, 0, -1)
                # cv2.circle(debugImg, (640, 360), radius, (0,0,255), 2)

        if "targetCenter" in locals():
            if distance.euclidean((targetCenter.x, targetCenter.y), (640, 360)) > radius:
                cv2.circle(debugImg, (640, 360), radius, (0, 0, 255), 10)
            else:
                cv2.circle(debugImg, (640, 360), radius, (0, 255, 0), 10)
        else:
            cv2.circle(debugImg, (640, 360), radius, (0, 255, 0), 10)
        self.targetDebugger.publish(
            self.bridge.cv2_to_imgmsg(debugImg, encoding="passthrough"))

        return targetCenter

    def poseCallback(self, data):
        self.relAlt = data.pose.position.z  # em metros


def main():
    global processing, new_msg, msg

    rospy.init_node("target", anonymous=True)
    rate = rospy.Rate(3)
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
