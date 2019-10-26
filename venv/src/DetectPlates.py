# DetectPlates.py

import math
import os
import random
import time

import cv2
import numpy as np

import DetectChars
import PossibleChar
import PossiblePlate
import Preprocess

# module level variables ##########################################################################
PLATE_WIDTH_PADDING_FACTOR = 1.3
PLATE_HEIGHT_PADDING_FACTOR = 1.5
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_RED = (0.0, 0.0, 255.0)
SAVE_IMAGE = False
NO_ERROR_PRINT_ENABLED = False
SHOW_IMAGE = False
SHOW_TIME = False
SUPER_SPEED_MODE = True


###################################################################################################
# INPUT: imagen
# OUTPUT: Lista de PossiblePlate

def detect_plates_in_scene(img_original_scene) -> [PossiblePlate]:
    if img_original_scene is None:
        print("\n### Error: image not found ### \n\n")
        os.system("pause")
        return

    list_of_possible_plates: [PossiblePlate] = []

    height, width, num_channels = img_original_scene.shape

    # Creo las matrices vacias del tamaño de la imagen
    img_grayscale_scene = np.zeros((height, width, 1), np.uint8)
    img_thresh_scene = np.zeros((height, width, 1), np.uint8)
    img_contours = np.zeros((height, width, 3), np.uint8)

    # Proceso el INPUT y obtengo un grayScale y un Threshold.
    img_grayscale_scene, img_thresh_scene = Preprocess.preprocess(img_original_scene)

    # Guardo como jpg el grayScale y Threshold.
    if SAVE_IMAGE:
        cv2.imwrite("./output/showImage/grayscale.jpg", img_grayscale_scene)
        cv2.imwrite("./output/showImage/thresh.jpg", img_thresh_scene)

    # Muestro el grayScale y Threshold.
    if SHOW_IMAGE:
        cv2.imshow("GrayScaleImage", img_grayscale_scene)
        cv2.imshow("AdaptativeThresholdImage", img_thresh_scene)

    # Busco todos los posibles chars
    # Primero buscamos todos los contornos, y despues los filtramos por los que pueden ser chars
    # Sin comparar contra otros
    if SHOW_TIME:
        start_time_2 = time.time()
        list_of_possible_chars_in_scene = find_possible_chars_in_scene(img_thresh_scene)
        finish_time_2 = time.time() - start_time_2
        print("--- Find possible chars: %s seconds ---" % finish_time_2)
    else:
        list_of_possible_chars_in_scene: [PossibleChar] = find_possible_chars_in_scene(img_thresh_scene)

    # Busco grupos de posibles chars
    list_of_lists_of_matching_chars_in_scene = DetectChars.find_list_of_lists_of_matching_chars(
        list_of_possible_chars_in_scene)

    # Dibujo los contornos a los posibles chars
    if not SUPER_SPEED_MODE:
        for list_of_matching_chars in list_of_lists_of_matching_chars_in_scene:
            int_random_blue = random.randint(0, 255)
            int_random_green = random.randint(0, 255)
            int_random_red = random.randint(0, 255)

            contours = []

            for matching_char in list_of_matching_chars:
                contours.append(matching_char.contour)

            cv2.drawContours(img_contours, contours, -1, (int_random_blue, int_random_green, int_random_red))

            if SHOW_IMAGE:
                cv2.imshow("5- Posibles contornos", img_contours)

    for list_of_matching_chars in list_of_lists_of_matching_chars_in_scene:
        possible_plate = extract_plate(img_original_scene, list_of_matching_chars)

        if possible_plate.imgPlate is not None:
            list_of_possible_plates.append(possible_plate)

    if NO_ERROR_PRINT_ENABLED:
        print("\n" + str(len(list_of_possible_plates)) + " possible plates found" + "\n")

    for i in range(0, len(list_of_possible_plates)):
        p2f_rect_points = cv2.boxPoints(list_of_possible_plates[i].rrLocationOfPlateInScene)

        cv2.line(img_contours, tuple(p2f_rect_points[0]), tuple(p2f_rect_points[1]), SCALAR_RED, 2)
        cv2.line(img_contours, tuple(p2f_rect_points[1]), tuple(p2f_rect_points[2]), SCALAR_RED, 2)
        cv2.line(img_contours, tuple(p2f_rect_points[2]), tuple(p2f_rect_points[3]), SCALAR_RED, 2)
        cv2.line(img_contours, tuple(p2f_rect_points[3]), tuple(p2f_rect_points[0]), SCALAR_RED, 2)

        if SHOW_IMAGE:
            cv2.imshow("Posibles patentes " + str(i), img_contours)

        if SAVE_IMAGE:
            cv2.imwrite("./output/showImage/PosiblesPatentes.jpg", img_contours)

        if NO_ERROR_PRINT_ENABLED:
            print("possible plate " + str(i))
            print("\n### Plate detection complete ###\n")

    return list_of_possible_plates


# end function

######################################################
# Buscamos posibles caracteres basandonos en el tamaño
# INPUT: Imagen Threshold
# OUTPUT: Lista de PossibleChar
######################################################

def find_possible_chars_in_scene(img_thresh) -> [PossibleChar]:
    list_of_possible_chars: [PossibleChar] = []
    int_count_of_possible_chars: int = 0
    img_thresh_copy = img_thresh.copy()

    contours, npa_hierarchy = cv2.findContours(img_thresh_copy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    height, width = img_thresh.shape
    img_contours = np.zeros((height, width, 3), np.uint8)

    for i in range(0, len(contours)):
        if not SUPER_SPEED_MODE:
            cv2.drawContours(img_contours, contours, i, SCALAR_WHITE, 3)

        possible_char = PossibleChar.PossibleChar(contours[i])

        if DetectChars.check_if_possible_char(possible_char):
            int_count_of_possible_chars = int_count_of_possible_chars + 1
            list_of_possible_chars.append(possible_char)

    if NO_ERROR_PRINT_ENABLED:
        print("\n[F:Find Possible Chars in Scene] - len(contours) = " + str(len(contours)))
        print("\n[F:Find Possible Chars in Scene] - int_count_of_possible_chars = " + str(int_count_of_possible_chars))

    if SHOW_IMAGE:
        cv2.imshow("[F:Find Possible Chars in Scene] - Contours", img_contours)

    if SAVE_IMAGE:
        cv2.imwrite("./output/showImage/4-Find Possible Chars-Contours.jpg", img_contours)

    return list_of_possible_chars


##########################################################
# Recorta la imagen original y devuelve la patente cortada
# INPUT: Imagen y lista de chars a cortar
# OUTPUT: PossiblePlate con la imagen cropeada, los demas parametros en default
###########################################################
def extract_plate(img_original, list_of_matching_chars) -> PossiblePlate:
    possible_plate = PossiblePlate.PossiblePlate()

    # Ordenamos los chars de izquierda a derecha en base a la posicion X
    list_of_matching_chars.sort(key=lambda matching_char: matching_char.intCenterX)

    # Calculamos el punto central de la patente
    flt_plate_center_x = (list_of_matching_chars[0].intCenterX + list_of_matching_chars[
        len(list_of_matching_chars) - 1].intCenterX) / 2.0

    flt_plate_center_y = (list_of_matching_chars[0].intCenterY + list_of_matching_chars[
        len(list_of_matching_chars) - 1].intCenterY) / 2.0

    pt_plate_center = flt_plate_center_x, flt_plate_center_y

    # calculate plate width and height
    int_plate_width = int(
        (list_of_matching_chars[len(list_of_matching_chars) - 1].intBoundingRectX + list_of_matching_chars[
            len(list_of_matching_chars) - 1].intBoundingRectWidth - list_of_matching_chars[
             0].intBoundingRectX) * PLATE_WIDTH_PADDING_FACTOR)

    int_total_of_char_heights = 0

    for matching_char in list_of_matching_chars:
        int_total_of_char_heights = int_total_of_char_heights + matching_char.intBoundingRectHeight

    flt_average_char_height = int_total_of_char_heights / len(list_of_matching_chars)

    int_plate_height = int(flt_average_char_height * PLATE_HEIGHT_PADDING_FACTOR)

    # calculate correction angle of plate region
    flt_opposite = list_of_matching_chars[len(list_of_matching_chars) - 1].intCenterY - list_of_matching_chars[
        0].intCenterY

    flt_hypotenuse = DetectChars.distance_between_chars(list_of_matching_chars[0],
                                                        list_of_matching_chars[len(list_of_matching_chars) - 1])

    flt_correction_angle_in_rad = math.asin(flt_opposite / flt_hypotenuse)

    flt_correction_angle_in_deg = flt_correction_angle_in_rad * (180.0 / math.pi)

    # pack plate region center point, width and height, and correction angle into rotated rect member variable of plate
    possible_plate.rrLocationOfPlateInScene = (
        tuple(pt_plate_center), (int_plate_width, int_plate_height), flt_correction_angle_in_deg)

    # final steps are to perform the actual rotation

    # get the rotation matrix for our calculated correction angle
    rotation_matrix = cv2.getRotationMatrix2D(tuple(pt_plate_center), flt_correction_angle_in_deg, 1.0)

    height, width, num_channels = img_original.shape  # unpack original image width and height

    img_rotated = cv2.warpAffine(img_original, rotation_matrix, (width, height))  # rotate the entire image

    img_cropped = cv2.getRectSubPix(img_rotated, (int_plate_width, int_plate_height), tuple(pt_plate_center))

    possible_plate.imgPlate = img_cropped

    return possible_plate

# end function
