# DetectChars.py
import os

import cv2
import numpy as np
import math
import random

import Preprocess
import PossibleChar

# module level variables ##########################################################################

kNearest = cv2.ml.KNearest_create()

# constants for checkIfPossibleChar, this checks one possible char only (does not compare to another char)
MIN_PIXEL_WIDTH = 12
MIN_PIXEL_HEIGHT = 18

MIN_ASPECT_RATIO = 0.6
MAX_ASPECT_RATIO = 1.0

MIN_PIXEL_AREA = 300

# constants for comparing two chars
MIN_DIAG_SIZE_MULTIPLE_AWAY = 0.3
MAX_DIAG_SIZE_MULTIPLE_AWAY = 5.0

MAX_CHANGE_IN_AREA = 0.5

MAX_CHANGE_IN_WIDTH = 0.8
MAX_CHANGE_IN_HEIGHT = 0.2

MAX_ANGLE_BETWEEN_CHARS = 12.0

# other constants
MIN_NUMBER_OF_MATCHING_CHARS = 5

RESIZED_CHAR_IMAGE_WIDTH = 20
RESIZED_CHAR_IMAGE_HEIGHT = 30

MIN_CONTOUR_AREA = 100

SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)


###################################################################################################
def find_possible_chars_in_plate(img_grayscale, img_thresh) -> [PossibleChar]:
    list_of_possible_chars = []
    contours = []
    img_thresh_copy = img_thresh.copy()

    # Buscamos todos los contornos en la patente
    contours, npa_hierarchy = cv2.findContours(img_thresh_copy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        possible_char = PossibleChar.PossibleChar(contour)
        if check_if_possible_char(possible_char):
            list_of_possible_chars.append(possible_char)

    return list_of_possible_chars


###################################################################################################
def check_if_possible_char(possible_char: PossibleChar) -> bool:
    # Es un primer paso para ver si el contorno puede ser un caracter.
    # No estamos comparando contra otros caracteres
    if (possible_char.intBoundingRectArea > MIN_PIXEL_AREA
            and possible_char.intBoundingRectWidth > MIN_PIXEL_WIDTH
            and possible_char.intBoundingRectHeight > MIN_PIXEL_HEIGHT
            and MIN_ASPECT_RATIO < possible_char.fltAspectRatio < MAX_ASPECT_RATIO):
        return True
    else:
        return False


###################################################################################################
# Lo que hacemos en esta funcion es agrupar los char por angulo y cercania
def find_list_of_lists_of_matching_chars(list_of_possible_chars):
    list_of_lists_of_matching_chars = []

    for possible_char in list_of_possible_chars:
        list_of_matching_chars = find_list_of_matching_chars(possible_char, list_of_possible_chars)

        list_of_matching_chars.append(possible_char)

        # Queremos tener al menos un conjunto de 3 chars
        if len(list_of_matching_chars) < MIN_NUMBER_OF_MATCHING_CHARS:
            continue

        list_of_lists_of_matching_chars.append(list_of_matching_chars)

        list_of_possible_chars_with_current_matches_removed = []

        # remove the current list of matching chars from the big list so we don't use those same chars twice,
        # make sure to make a new big list for this since we don't want to change the original big list
        list_of_possible_chars_with_current_matches_removed = list(
            set(list_of_possible_chars) - set(list_of_matching_chars))

        recursive_list_of_lists_of_matching_chars = find_list_of_lists_of_matching_chars(
            list_of_possible_chars_with_current_matches_removed)

        for recursive_list_of_matching_chars in recursive_list_of_lists_of_matching_chars:
            list_of_lists_of_matching_chars.append(recursive_list_of_matching_chars)
        break

    return list_of_lists_of_matching_chars


###################################################################################################
def find_list_of_matching_chars(possible_char: PossibleChar, list_of_chars):
    # the purpose of this function is, given a possible char and a big list of possible chars,
    # find all chars in the big list that are a match for the single possible char,
    list_of_matching_chars = []

    for possible_matching_char in list_of_chars:
        # if the char we attempting to find matches for is the exact same char as the char in the big list we are
        # currently checking
        if possible_matching_char == possible_char:
            # then we should not include it in the list of matches b/c that would end up double
            # including the current char
            continue

        # compute stuff to see if chars are a match
        flt_distance_between_chars = distance_between_chars(possible_char, possible_matching_char)

        flt_angle_between_chars = angle_between_chars(possible_char, possible_matching_char)

        flt_change_in_area = float(
            abs(possible_matching_char.intBoundingRectArea - possible_char.intBoundingRectArea)) / float(
            possible_char.intBoundingRectArea)

        flt_change_in_width = float(
            abs(possible_matching_char.intBoundingRectWidth - possible_char.intBoundingRectWidth)) / float(
            possible_char.intBoundingRectWidth)

        flt_change_in_height = float(
            abs(possible_matching_char.intBoundingRectHeight - possible_char.intBoundingRectHeight)) / float(
            possible_char.intBoundingRectHeight)

        # check if chars match
        if (flt_distance_between_chars < (possible_char.fltDiagonalSize * MAX_DIAG_SIZE_MULTIPLE_AWAY) and
                flt_angle_between_chars < MAX_ANGLE_BETWEEN_CHARS and
                flt_change_in_area < MAX_CHANGE_IN_AREA and
                flt_change_in_width < MAX_CHANGE_IN_WIDTH and
                flt_change_in_height < MAX_CHANGE_IN_HEIGHT):
            list_of_matching_chars.append(possible_matching_char)

    return list_of_matching_chars


###################################################################################################
# Calculamos la distancia entre dos chars
def distance_between_chars(first_char: PossibleChar, second_char: PossibleChar) -> float:
    int_x = abs(first_char.intCenterX - second_char.intCenterX)
    int_y = abs(first_char.intCenterY - second_char.intCenterY)

    return math.sqrt((int_x ** 2) + (int_y ** 2))


###################################################################################################
#  Calculamos el angulo entre dos chars en grados
def angle_between_chars(first_char: PossibleChar, second_char: PossibleChar) -> float:
    flt_adj = float(abs(first_char.intCenterX - second_char.intCenterX))
    flt_opp = float(abs(first_char.intCenterY - second_char.intCenterY))

    if flt_adj != 0.0:
        flt_angle_in_rad = math.atan(flt_opp / flt_adj)
    else:
        flt_angle_in_rad = 1.5708

    flt_angle_in_deg = flt_angle_in_rad * (180.0 / math.pi)

    return flt_angle_in_deg
