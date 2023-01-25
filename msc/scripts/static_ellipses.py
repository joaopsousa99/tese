#! /usr/bin/python3
from __future__ import print_function
from cgitb import small
import itertools
import math
import cv2
import numpy as np
import rospy
from cv_bridge import (CvBridge, CvBridgeError)
from geometry_msgs.msg import Point
from msc.PrintColours import *
from msc.py_gnc_functions import *
from scipy.spatial import distance
from sensor_msgs.msg import Image
import os


class targetDetector:
  def __init__(self):
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
    self.FOCAL_DISTANCE = (1942.8336 + 1936.85487)/2 # média de fx e fy
    # self.WIDTH_IN_MM = 160
    self.WIDTH_IN_MM = 752.5
    self.OUTER_RADII_RATIO_IDEAL = 400/500
    self.INNER_RADII_RATIO_IDEAL = 150/250
    self.SMALL_RADII_RATIO_IDEAL = 10/30

  def msg_processing(self):
    # converte a mensagem ROS Image para uma imagem OpenCV
    # img = cv2.imread("/home/jp/drawio/static_ellipses/static_ellipses.jpg")
    # img = cv2.imread("/home/jp/Pictures/Screenshot from 2022-09-08 15-08-17.png")
    # img = cv2.imread("/home/jp/Pictures/focal_distance_dataset/1320mm.jpg")
    img = cv2.imread("/home/jp/IMAGE.JPG")

    # o detetor de contornos trabalha sobre imagens de apenas um canal
    # pré-processamento para remover ruído
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgGray = cv2.bilateralFilter(imgGray, 5, 175, 175)
    imgMax = np.amax(imgGray)

    # o detetor de contornos precisa de imagens binárias para funcionar bem
    threshType = cv2.THRESH_BINARY + cv2.THRESH_OTSU
    _, th = cv2.threshold(imgGray, self.BINARIZATION_THRESH, imgMax, threshType)
    cv2.imwrite("images/a.jpg", th)
    # th = cv2.adaptiveThreshold(imgGray, imgMax, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    contours, _ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    ellipses = self.ellipseFilter(contours, th)

    rings = self.ringDetector(imgGray, contours, ellipses)

    _ = self.targetDetector(rings, imgGray)

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
        ((x, y),(a, b), phi) = cv2.fitEllipse(contours[c])
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
        lessThanMaxArea  = contourArea < maxArea

        if greaterThanMinArea and lessThanMaxArea:
          # adiciona elipse à lista de candidatos
          ellipses.append((x, y, a, b, phi, c))
          debugimg = cv2.ellipse(debugimg, (int(x), int(y)), (int(a/2), int(b/2)), phi, 0, 360, (0, 255, 0), 2)
        elif not (np.isnan(x) or np.isnan(y) or np.isnan(a) or np.isnan(b) or np.isnan(phi)):
          debugimg = cv2.ellipse(debugimg, (int(x), int(y)), (int(a/2), int(b/2)), phi, 0, 360, (0, 0, 255), 2)
      
    for e in ellipses:
      # print(f"ELIPSE {e[5]}:\t[{e[0]:.0f}, {e[1]:.0f}]\t[{e[2]/ellipses[-1][2]*500:.0f}, {e[3]/ellipses[-1][3]*500:.0f}]")
      print(f"ELIPSE {e[5]}:\t[{e[0]:.0f}, {e[1]:.0f}]\t[{e[2]:.0f}, {e[3]:.0f}]")

    cv2.imwrite("images/ellipse_debugger.jpg", debugimg)
    return (ellipses)

  def ringDetector(self, imgGray, contours, ellipses):
    imgMax = np.amax(imgGray)
    imgMin = np.amin(imgGray)
    dimg = imgGray.copy()
    dimg = cv2.merge((dimg, dimg, dimg))
    dimg2 = imgGray.copy()
    dimg2 = cv2.merge((dimg2, dimg2, dimg2))

    ellipseCombinations = itertools.combinations(ellipses, 2)
    rings = []

    # ciclo for para determinar que pares de elipses são anéis válidos
    for index, pair in enumerate(ellipseCombinations):
      dimgaux = dimg2.copy()
      c1  = (pair[0][0], pair[0][1])
      c2  = (pair[1][0], pair[1][1])
      a   = (pair[0][2], pair[1][2])
      b   = (pair[0][3], pair[1][3])
      phi = (pair[0][4], pair[1][4])
      c   = (pair[0][5], pair[1][5])
      # dimg = cv2.drawContours(dimg, [contours[c[0]], contours[c[1]]], -1, (0,255,0), -1)

      # se as elipses do par não são concêntricas, salta para o próximo par
      if distance.euclidean(c1, c2) >= self.CONCENTRIC_TOLERANCE:
        # print(f"[{c[0]}, {c[1]}]: concentricidade")
        continue

      # excentricidade = √(b²-a²)/b, assumindo que b≥a
      eccentricity = (sqrt(b[0]**2 - a[0]**2)/b[0], sqrt(b[1]**2 - a[1]**2)/b[1])
      eccentricSimilarity = abs(eccentricity[0] - eccentricity[1])
      notSimilarlyEccentric = eccentricSimilarity > self.ECCENTRIC_SIMILARITY_TOLERANCE

      # se as elipses do par não têm excentricidades parecidas, salta para o próximo par
      if notSimilarlyEccentric:
        print(f"[{c[0]}, {c[1]}]: similaridade excêntrica")
        print(f"{eccentricity} {eccentricSimilarity}")
        dimgaux[ringPoints[0], ringPoints[1]] = (0,0,255)
        cv2.imwrite(f"images/ring[{c[0]}, {c[1]}].jpg", dimgaux)
        continue

      # já não me lembro do que é que isto faz
      if eccentricSimilarity <= self.ECCENTRIC_TOLERANCE:
        angleDiff = 0
      else:
        angleDiff = abs(math.atan2(math.sin(phi[0] - phi[1]), math.cos(math.sin(phi[0] - phi[1]))))

      anglesAreTooDifferent = angleDiff > self.ANGLE_DIFF_TOLERANCE

      if anglesAreTooDifferent:
        print(f"[{c[0]}, {c[1]}]: orientação")
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
        print(f"[{c[0]}, {c[1]}]: candidatos")
        continue

      ringPointsList = []
      _,_,widthInPixels,_ = cv2.boundingRect(contours[c[1]])
      # distanceToObject = self.FOCAL_DISTANCE*self.WIDTH_IN_MM/(widthInPixels*0.00122)
      distanceToObject = self.FOCAL_DISTANCE*self.WIDTH_IN_MM/widthInPixels

      # valor abaixo do qual um píxel é considerado preto
      blackThreshold = self.BLACK_TOLERANCE_PERCENT*(imgMax - imgMin) + imgMin + 100

      # máscara que indica os píxeis dentro do anel
      ringMask = np.zeros_like(imgGray)
      ringContours = [contours[c[0]], contours[c[1]]]
      cv2.drawContours(ringMask, ringContours, -1, color=255, thickness=-1)

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
      if blackPercentage <= 80:#desmos[0]/(distanceToObject + desmos[1]) + desmos[2]:
        print(blackPercentage)
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

      dimg = cv2.drawContours(dimg, [contours[c[0]], contours[c[1]]], -1, (0,255,0), -1)
      dimgaux = cv2.putText(dimgaux, f"{c}", (int(ringEllipsesCentroid[0]), int(ringEllipsesCentroid[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0))
      dimgaux = cv2.drawContours(dimgaux, [contours[c[0]], contours[c[1]]], -1, (0,255,0), -1)
      cv2.imwrite(f"images/ring{index}.jpg", dimgaux)

    cv2.imwrite("images/ring_debugger.jpg", dimg)
    return rings

  def targetDetector(self, rings, img):
    # rings = (index, typeOfRing, ringEllipsesCentroid, meanRingOrientation, ringContours)
    debugImg = img.copy()
    debugImg = cv2.merge((debugImg, debugImg, debugImg))
    # print(f"int(max(min(-30*{self.relAlt:.2f} + 115, 100), 25)) = {radius}")
    print(rings[0][1])
    # só há 1 anel e não é o exterior
    if len(rings) == 1 and rings[0][1] != 0:
      print(f"[{rings[0]}]:\t\tsó há um anel e não é o exterior")
      x = rings[0][2][0]
      y = rings[0][2][1]
      z = 0
      targetCentre = Point(x, y, z)
      # cv2.drawContours(debugImg, rings[0][4], -1, (0, 255, 0), -1)
      cv2.line(debugImg, (0,int(y)), (4608,int(y)), (255,0,0), 2)
      cv2.line(debugImg, (int(x),0), (int(x),3456), (255,0,0), 2)
      # print("DESENHEI")

    # há dois anéis e têm tipos diferentes
    elif len(rings) == 2 and rings[0][1] != rings[1][1]:
      print(f"[{rings[0][0]}, {rings[1][0]}]:\t\thá dois anéis e têm tipos diferentes")
      x = (rings[0][2][0] + rings[1][2][0])/2
      y = (rings[0][2][1] + rings[1][2][1])/2
      z = 0
      targetCentre = Point(x, y, z)
      # cv2.drawContours(debugImg, rings[0][4], -1, (0, 255, 0), -1)
      # cv2.drawContours(debugImg, rings[1][4], -1, (0, 255, 0), -1)
      cv2.line(debugImg, (0,int(y)), (4608,int(y)), (255,0,0), 2)
      cv2.line(debugImg, (int(x),0), (int(x),3456), (255,0,0), 2)
      # print("DESENHEI")
    
    # há três anéis e têm tipos diferentes
    elif len(rings) == 3 and rings[0][1] != rings[1][1] and rings[0][1] != rings[2][1] and rings[1][1] != rings[2][1]:
      print(f"[{rings[0][0]}, {rings[1][0]}, {rings[2][0]}]:\t\thá três anéis e têm tipos diferentes")
      x = (rings[0][2][0] + rings[1][2][0] + rings[2][2][0])/3
      y = (rings[0][2][1] + rings[1][2][1] + rings[2][2][1])/3
      z = 0
      targetCentre = Point(x, y, z)
      # print(f"{x}, {y}, {z}")
      # cv2.drawContours(debugImg, rings[0][4], -1, (0, 255, 0), -1)
      # cv2.drawContours(debugImg, rings[1][4], -1, (0, 255, 0), -1)
      cv2.line(debugImg, (0,int(y)), (4608,int(y)), (255,0,0), 2)
      cv2.line(debugImg, (int(x),0), (int(x),3456), (255,0,0), 2)
      # print("DESENHEI")

    else:
      debugstr = "["
      for r in rings:
        debugstr = debugstr + str(r[0]) + ", "
      print(debugstr[:-2] + f"]: há {len(rings)} anéis e têm tipos iguais")

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

        # se os anéis forem do mesmo tipo (exterior e interior), salta para o próximo par
        if ringType[0] == ringType[1]:
          print(f"{ringCombo[0][0]} {ringCombo[1][0]} anéis do mesmo tipo")
          # cv2.drawContours(debugImg, ringCombo[0][4], -1, (0, 0, 255), -1)
          # cv2.drawContours(debugImg, ringCombo[1][4], -1, (0, 0, 255), -1)
          continue

        # se os anéis não foram concêntricos, salta para o próximo par
        if distance.euclidean(ringCombo[0][2], ringCombo[1][2]) >= self.CONCENTRIC_TOLERANCE:
          print(f"{ringCombo[0][0]} {ringCombo[1][0]} anéis não concêntricos")
          # cv2.drawContours(debugImg, ringCombo[0][4], -1, (0, 0, 255), -1)
          # cv2.drawContours(debugImg, ringCombo[1][4], -1, (0, 0, 255), -1)
          continue

        # se os anéis não tiverem a mesma orientação, salta para o próximo par
        angles = (ringCombo[0][3], ringCombo[1][3])
        # diferença de ângulos = tan⁻¹(sin(α-β)/cos(α-β))
        angleDiff = abs(math.atan2(math.sin(angles[0] - angles[1]), math.cos(math.sin(angles[0] - angles[1]))))

        if angleDiff > self.ANGLE_DIFF_TOLERANCE:
          print(f"{ringCombo[0][0]} {ringCombo[1][0]} anéis com orientações diferentes")
          # cv2.drawContours(debugImg, ringCombo[0][4], -1, (0, 0, 255), -1)
          # cv2.drawContours(debugImg, ringCombo[1][4], -1, (0, 0, 255), -1)
          continue

        print(f"{ringCombo[0][0]} {ringCombo[1][0]} APPEND")
        targets.append(ringCombo)

      ringCombinations = itertools.combinations(rings, 3)

      for ringCombo in ringCombinations:
        ringType = (ringCombo[0][1], ringCombo[1][1], ringCombo[2][1])
        allDifferentTypes = ringType[0] != ringType[1] and ringType[0] != ringType[2] and ringType[1] != ringType[2]

        # se os anéis forem do mesmo tipo (exterior e interior), salta para o próximo trio
        if not allDifferentTypes:
          print(f"{ringCombo[0][0]} {ringCombo[1][0]} {ringCombo[2][0]} há anéis do mesmo tipo")
          # cv2.drawContours(debugImg, ringCombo[0][4], -1, (0, 0, 255), -1)
          # cv2.drawContours(debugImg, ringCombo[1][4], -1, (0, 0, 255), -1)
          continue

        # se os anéis não foram concêntricos, salta para o próximo trio
        dist01 = distance.euclidean(ringCombo[0][2], ringCombo[1][2])
        dist02 = distance.euclidean(ringCombo[0][2], ringCombo[2][2])
        dist12 = distance.euclidean(ringCombo[1][2], ringCombo[2][2])
        concentric = dist01 >= self.CONCENTRIC_TOLERANCE and dist02 >= self.CONCENTRIC_TOLERANCE and dist12 >= self.CONCENTRIC_TOLERANCE

        if not concentric:
          print(f"{ringCombo[0][0]} {ringCombo[1][0]} {ringCombo[2][0]} anéis não concêntricos")
          # cv2.drawContours(debugImg, ringCombo[0][4], -1, (0, 0, 255), -1)
          # cv2.drawContours(debugImg, ringCombo[1][4], -1, (0, 0, 255), -1)
          continue

        # se os anéis não tiverem a mesma orientação, salta para o próximo par
        angles = (ringCombo[0][3], ringCombo[1][3], ringCombo [2][3])
        # diferença de ângulos = tan⁻¹(sin(α-β)/cos(α-β))
        angleDiff01 = abs(math.atan2(math.sin(angles[0] - angles[1]), math.cos(math.sin(angles[0] - angles[1]))))
        angleDiff02 = abs(math.atan2(math.sin(angles[0] - angles[2]), math.cos(math.sin(angles[0] - angles[2]))))
        angleDiff12 = abs(math.atan2(math.sin(angles[1] - angles[2]), math.cos(math.sin(angles[1] - angles[2]))))

        anglesAreEqual = angleDiff01 > self.ANGLE_DIFF_TOLERANCE and angleDiff02 > self.ANGLE_DIFF_TOLERANCE and angleDiff12 > self.ANGLE_DIFF_TOLERANCE

        if not anglesAreEqual:
          print(f"{ringCombo[0][0]} {ringCombo[1][0]} {ringCombo[2][0]} anéis com orientações diferentes")
          # cv2.drawContours(debugImg, ringCombo[0][4], -1, (0, 0, 255), -1)
          # cv2.drawContours(debugImg, ringCombo[1][4], -1, (0, 0, 255), -1)
          continue

        print(f"{ringCombo[0][0]} {ringCombo[1][0]} {ringCombo[2][0]} APPEND")
        targets.append(ringCombo)

      if len(targets) >= 1:
        if len(targets) > 1:
          print("MAIS QUE UM ALVO DETETADO")

        # assume-se que o primeiro par de anéis é o correto
        targetCentreX = (targets[0][0][2][0] + targets[0][1][2][0])/2
        targetCentreY = (targets[0][0][2][1] + targets[0][1][2][1])/2
        # print(f"X {ringCombo[0][2][0]} {ringCombo[1][2][0]}")
        # print(f"Y {ringCombo[0][2][1]} {ringCombo[1][2][1]}")
        print(len(targets[0]))
        print(f"{targetCentreX} {targetCentreY}")

        targetCentre = Point(targetCentreX, targetCentreY, 0)
        # cv2.drawContours(debugImg, ringCombo[0][4], -1, (0, 0, 255), -1)
        # cv2.drawContours(debugImg, ringCombo[1][4], -1, (0, 0, 255), -1)
        cv2.line(debugImg, (0,int(targetCentreY)), (4608,int(targetCentreY)), (255,0,0),2)
        cv2.line(debugImg, (int(targetCentreX),0), (int(targetCentreX),3456), (255,0,0),2)
        # print(f"alvo detetado ({(targetCentreX/1280)*100:.2f}, {(targetCentreY/800)*100:.2f})")
      else:
        print("Não foram detetados alvos")
        targetCentre = Point(0, 0, -1)

    cv2.imwrite("images/target_debugger.jpg", debugImg)
    return targetCentre

  def poseCallback(self, data):
    self.relAlt = data.pose.pose.position.z # em metros


if __name__ == '__main__':
  os.chdir("/home/jp/catkin_ws/src/msc")
  print(os.getcwd())
  td = targetDetector()
  td.msg_processing()