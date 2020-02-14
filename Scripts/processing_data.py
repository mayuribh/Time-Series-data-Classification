import os
import sys
import csv
import time

import math
import numpy as np

import statistics
import matplotlib.pyplot as plt
timeout = 1.0


# Variablen für die GUI----------------------------------------------------------------------------
gesture = None
fix = 0     # added by MJ
fixations_valid_flag = False

Sample_Point = []
T = []
Gaze_last = []

Median_Cal_X = []
Median_Cal_Y = []
Median_Cal_Z = []

Median_Cal_left_X = []
Median_Cal_left_Y = []
Median_Cal_right_X = []
Median_Cal_right_Y = []

time_interval = 0
preprocessed_data_list = []

Fixation = []
number = None




TIME_MAX_GAP = 0.075
VELOCITY_THRESHOLD = 30
MAX_TIME_BETWEEN_FIXATIONS = 0.075
MAX_ANGLE_BETWEEN_FIXATIONS = 0.5
MAX_FIXATION_DURATION = 0
MIN_FIXATION_DURATION = 0.06
NOISE_REDUCTION_WINDOW_LENGTH = 3
DIRECTION_ANGLE = 0  # 0-22.5 degree
MIN_ANGLE_BETWEEN_FIXATION_GROUP = 6

#VARIABLEN FÜR DATENLOG---------------------------------------------------------------------------------------

sequence = []
sequence_list = []


file_list = []
list_from_csv = []


def loader():
    global file_list

    os.path.dirname(sys.argv[0])
    print("Location is", os.path.dirname(sys.argv[0]) + "/Model Data/relabeld Data/splitted data")

    file_list = next(os.walk(os.path.dirname(sys.argv[0]) + "/Model Data/relabeld Data/splitted data"))[2]

    file_list.sort()

    del file_list[0]

    print("\n Data List loaded")


def file_loader():

    global list_from_csv
    global actual_csv

    print("\nloading path is", os.path.dirname(sys.argv[0]) + "/Model Data/relabeld Data/splitted data/" + actual_csv)
    csv.register_dialect('myDialect1', delimiter=';')
    csv.register_dialect('myDialect2', delimiter='#')
    with open(os.path.dirname(sys.argv[0]) + "/Model Data/relabeld Data/splitted data/" + actual_csv) as f:
        try:
            reader = csv.reader(f, dialect='myDialect1')
        except:
            reader = csv.reader(f, dialect='myDialect2')

        n = -1

        for row in reader:
            n = n + 1

            list_from_csv.append(row)

    print("\n File", actual_csv, "loaded")


def sequence_generator():
    global list_from_csv
    global sequence_list

    transfer_list = []

    for i in range(len(list_from_csv)):

        if i == 0:
            transfer_list.append(list_from_csv[i])

        if list_from_csv[i-1][8] == list_from_csv[i][8]:
            transfer_list.append(list_from_csv[i])

        if list_from_csv[i-1][8] != list_from_csv[i][8]:
            sequence_list.append(transfer_list)

            transfer_list = []
        if i == len(list_from_csv)-1:

            sequence_list.append(transfer_list)

            transfer_list = []

    del sequence_list[0]

    print("\n Total Sequences generated", len(sequence_list))


def interpolation(Sample_Point):
    global Gaze_last

    velocity_calculator(Sample_Point)
    i = len(Sample_Point)
    num_interpolation = math.ceil((Sample_Point[i - 1][7] - Sample_Point[i - 2][7]) / 0.02)
    time_interval = (Sample_Point[i - 1][7] - Sample_Point[i - 2][7]) / num_interpolation
    scaling_factor = float(1.0) / num_interpolation
    num_newpoint = int(num_interpolation - 1)

    left_x = float(Sample_Point[i - 1][0] - Sample_Point[i - 2][0])
    left_y = float(Sample_Point[i - 1][1] - Sample_Point[i - 2][1])
    right_x = float(Sample_Point[i - 1][2] - Sample_Point[i - 2][2])
    right_y = float(Sample_Point[i - 1][3] - Sample_Point[i - 2][3])

    delta_x = float(Sample_Point[i - 1][4] - Sample_Point[i - 2][4])
    delta_y = float(Sample_Point[i - 1][5] - Sample_Point[i - 2][5])

    label = Sample_Point[i - 1][6]

    Gaze_last = Sample_Point[i - 1]

    del Sample_Point[i - 1]

    for n in range(0, num_newpoint):
        i = len(Sample_Point)

        interpolation_point_left_x = scaling_factor * left_x + Sample_Point[i - 1][4]
        interpolation_point_left_y = scaling_factor * left_y + Sample_Point[i - 1][5]
        interpolation_point_rigt_x = scaling_factor * right_x + Sample_Point[i - 1][4]
        interpolation_point_rigt_y = scaling_factor * right_y + Sample_Point[i - 1][5]

        interpolation_point_x = scaling_factor * delta_x + Sample_Point[i - 1][4]
        interpolation_point_y = scaling_factor * delta_y + Sample_Point[i - 1][5]


        ts = time_interval + Sample_Point[i - 1][7]
        Gaze_Position_time = [interpolation_point_left_x, interpolation_point_left_y, interpolation_point_rigt_x, interpolation_point_rigt_y, interpolation_point_x, interpolation_point_y, label, ts]
        Sample_Point.append(Gaze_Position_time)

        velocity_calculator(Sample_Point)

    Sample_Point.append(Gaze_last)


def noise_reduction(Sample_Point):
    global Median_Cal_X
    global Median_Cal_Y
    global Median_Cal_left_X
    global Median_Cal_left_Y
    global Median_Cal_right_X
    global Median_Cal_right_Y
    global Gaze_last

    i = len(Sample_Point)

    Median_Cal_left_X.append(Sample_Point[i - 3][0])
    Median_Cal_left_X.append(Sample_Point[i - 2][0])
    Median_Cal_left_X.append(Sample_Point[i - 1][0])

    Median_Cal_left_Y.append(Sample_Point[i - 3][1])
    Median_Cal_left_Y.append(Sample_Point[i - 2][1])
    Median_Cal_left_Y.append(Sample_Point[i - 1][1])

    Median_Cal_right_X.append(Sample_Point[i - 3][2])
    Median_Cal_right_X.append(Sample_Point[i - 2][2])
    Median_Cal_right_X.append(Sample_Point[i - 1][2])

    Median_Cal_right_Y.append(Sample_Point[i - 3][3])
    Median_Cal_right_Y.append(Sample_Point[i - 2][3])
    Median_Cal_right_Y.append(Sample_Point[i - 1][3])


    Median_Cal_X.append(Sample_Point[i - 3][4])
    Median_Cal_X.append(Sample_Point[i - 2][4])
    Median_Cal_X.append(Sample_Point[i - 1][4])

    Median_Cal_Y.append(Sample_Point[i - 3][5])
    Median_Cal_Y.append(Sample_Point[i - 2][5])
    Median_Cal_Y.append(Sample_Point[i - 1][5])

    median_left_x = statistics.median(Median_Cal_left_X)
    median_left_y = statistics.median(Median_Cal_left_Y)
    median_right_x = statistics.median(Median_Cal_right_X)
    median_right_y = statistics.median(Median_Cal_right_Y)

    median_x = statistics.median(Median_Cal_X)
    median_y = statistics.median(Median_Cal_Y)

    label = Sample_Point[i - 1][6]

    del Median_Cal_left_X[:], Median_Cal_left_Y[:], Median_Cal_right_X[:], Median_Cal_right_Y[:], Median_Cal_X[:], Median_Cal_Y[:],

    Gaze_Position_data = [median_left_x, median_left_y, median_right_x, median_right_y, median_x, median_y, label, Sample_Point[i - 2][7]]
    Gaze_last = Sample_Point[i - 1]

    del Sample_Point[i - 1], Sample_Point[i - 2]

    Sample_Point.append(Gaze_Position_data)
    velocity_calculator(Sample_Point)
    Sample_Point.append(Gaze_last)


def velocity_calculator(Sample_Point):

    i = len(Sample_Point)
    L1 = math.sqrt((Sample_Point[i - 2][4]) ** 2 + (Sample_Point[i - 2][5]) ** 2)
    L2 = math.sqrt((Sample_Point[i - 1][4]) ** 2 + (Sample_Point[i - 1][5]) ** 2)
    L3 = math.sqrt((Sample_Point[i - 1][4] - Sample_Point[i - 2][4]) ** 2 +
                   (Sample_Point[i - 1][5] - Sample_Point[i - 2][5]) ** 2)

    if L1 * L2 != 0:

        try:
            alpha = math.acos((L1 ** 2 + L2 ** 2 - L3 ** 2) / (2 * L1 * L2))
        except:
            alpha = 0

        alpha = alpha * 180 / 3.14

        time_interval = float(Sample_Point[i - 1][7] - Sample_Point[i - 2][7])
        velocity = alpha / time_interval
        Sample_Point[i - 1].append(velocity)


def Gesture_function(old_sequence_list):
    global preprocessed_data_list
    global name

    fixations_valid_flag = False
    Sample_Point = []
    Fixate=[]
    Fixation = []
    TIME_MAX_GAP = 0.075
    VELOCITY_THRESHOLD = 30


    for k in range(len(old_sequence_list)):
        Gaze_Position_Data = [float(old_sequence_list[k][1]), float(old_sequence_list[k][2]), float(old_sequence_list[k][3]),
                              float(old_sequence_list[k][4]), float(old_sequence_list[k][5]), float(old_sequence_list[k][6]),
                              float(old_sequence_list[k][7]), float(old_sequence_list[k][0]) / 1000]


        i = len(Sample_Point)

        if i == 0 and Gaze_Position_Data != None:
            Sample_Point.append(Gaze_Position_Data)
            Sample_Point[0].append(0)

        if i >= 1 and Gaze_Position_Data != None:

            if not (Gaze_Position_Data[5] == Sample_Point[i - 1][5] and
                    Gaze_Position_Data[6] == Sample_Point[i - 1][6])and\
                    Gaze_Position_Data[7] != Sample_Point[i - 1][7]:

                Sample_Point.append(Gaze_Position_Data)

                i = len(Sample_Point)

                if i == 2:
                    time_interval = Sample_Point[i - 1][7] - Sample_Point[i - 2][7]
                    if time_interval >= TIME_MAX_GAP:
                        interpolation(Sample_Point)
                    else:
                        velocity_calculator(Sample_Point)

                if i >= 3:

                    time_interval = Sample_Point[i - 1][7] - Sample_Point[i - 2][7]

                    if time_interval >= TIME_MAX_GAP:
                        interpolation(Sample_Point)

                    else:
                        noise_reduction(Sample_Point)
                        velocity_calculator(Sample_Point)


    # filename3 = str(number)
    #
    #
    # name1 = os.path.dirname(
    #     sys.argv[0]) + "/Model Data/preprocessed learning data/" + filename3 + ".csv"
    #
    # with open(name1, mode='a', newline='') as data_log_objective_file:
    #
    #     for i in range(0, len(Sample_Point)):
    #         data_log_demoraphic_writer = csv.writer(data_log_objective_file, delimiter=';')
    #         data_log_demoraphic_writer.writerow(Sample_Point[i])


    m = len(Sample_Point)

    if m < 21:
        return

    for i in range(m):
        if len(Sample_Point[i]) < 9:
            Sample_Point[i].append(0)

    for i in range(m):

        v = Sample_Point[i][8]

        if v < VELOCITY_THRESHOLD:
            Fixation.append(Sample_Point[i])

        else:
            Fixate.append(Fixation)
            Fixation.append(Sample_Point[i])
            Fixation = []
        if i == m - 1:
            Fixate.append(Fixation)

    first_fix_length = len(Fixate[0])       # this is for cutting the first fixation!
    cut_first_fix_number = 0

    if first_fix_length > 30:
        cut_first_fix_number = first_fix_length - 30
        del Fixate[0][:cut_first_fix_number]

    add_number = 110 - m
    add_number = add_number + cut_first_fix_number

    #print("\nadjusted length is", m)
    #print("resulting number of delta samples is", add_number)

    # normaly the outcome here are the definition of the hole sequence into fixations
    length_list = []
    reduced_list = []
    for i in range(len(Fixate)):
        length = [len(Fixate[i]), i]
        length_list.append(length)
        reduced_list.append(length)



    del reduced_list[0]
    del reduced_list[-1]

    reduced_list.sort()

    # check if the 2 longest fixations are to close together!!!
    marker_of_second_fixation = None
    new_fixation_1 = []
    new_fixation_2 = []

    for u in range(len(reduced_list) - 1):

        if not fixations_valid_flag:

            fixation_1 = Fixate[reduced_list[-1][1]]
            fixation_2 = Fixate[reduced_list[-u - 2][1]]

            fixation_1_x_list = []
            fixation_1_y_list = []
            fixation_2_x_list = []
            fixation_2_y_list = []

            for i in range(len(fixation_1)):
                x = fixation_1[i][4]
                y = fixation_1[i][5]
                fixation_1_x_list.append(x)
                fixation_1_y_list.append(y)

            for i in range(len(fixation_2)):
                x = fixation_2[i][4]
                y = fixation_2[i][5]
                fixation_2_x_list.append(x)
                fixation_2_y_list.append(y)


            fixation_1_x_mean = statistics.median(fixation_1_x_list)
            fixation_1_y_mean = statistics.median(fixation_1_y_list)
            fixation_2_x_mean = statistics.median(fixation_2_x_list)
            fixation_2_y_mean = statistics.median(fixation_2_y_list)

            distance = math.sqrt((fixation_1_x_mean - fixation_2_x_mean)**2 + (fixation_1_y_mean - fixation_2_y_mean)**2)

            if distance > 0.05:       # Values is defined iteratively: it should be higher then this for a correct fixation

                marker_of_second_fixation = -u - 2

                fixations_valid_flag = True
                if add_number < 0:

                    # samples has to be cutted
                    cut_number = - add_number
                    if cut_number / 2 < len(fixation_2):

                        for z in range(math.floor(cut_number / 2)):
                            cut_position_1 = round((len(fixation_1)) / 2)
                            cut_position_2 = round((len(fixation_2)) / 2)

                            del(fixation_1[cut_position_1])
                            del(fixation_2[cut_position_2])

                            new_fixation_1 = fixation_1
                            new_fixation_2 = fixation_2

                        if cut_number % 2 == 1:
                            cut_position_2 = round((len(fixation_2)) / 2)
                            del (fixation_2[cut_position_2])

                            new_fixation_1 = fixation_1
                            new_fixation_2 = fixation_2

                    else:   # just in case on fixation is to short
                        if len(fixation_2) > 5:
                            cut_number_2 = len(fixation_2) - 5
                            cut_number_1 = cut_number - cut_number_2
                        else:
                            cut_number_2 = 0
                            cut_number_1 = cut_number

                        for z in range(cut_number_2):
                            cut_position_2 = round((len(fixation_2)) / 2)
                            del(fixation_2[cut_position_2])

                            new_fixation_2 = fixation_2

                        for z in range(cut_number_1):
                            cut_position_1 = round((len(fixation_1)) / 2)
                            if len(fixation_1) > 5:
                                del(fixation_1[cut_position_1])

                            else:

                                if len(Fixate[reduced_list[-3][1]]) > 4:
                                    del (Fixate[reduced_list[-3][1]][-1])
                                else:
                                    if len(Fixate[reduced_list[-4][1]]) > 3:
                                        del (Fixate[reduced_list[-4][1]][-1])
                                    else:
                                        if len(Fixate[reduced_list[-5][1]]) > 2:
                                            del (Fixate[reduced_list[-5][1]][-1])
                                        else:
                                            if len(Fixate[reduced_list[-6][1]]) > 2:
                                                del (Fixate[reduced_list[-6][1]][-1])
                                            else:
                                                if len(Fixate[reduced_list[-7][1]]) > 2:
                                                    del (Fixate[reduced_list[-7][1]][-1])
                                                else:
                                                    if len(Fixate[reduced_list[-8][1]]) > 1:
                                                        del (Fixate[reduced_list[-8][1]][-1])
                                                    else:
                                                        if len(Fixate[reduced_list[-9][1]]) > 1:
                                                            del (Fixate[reduced_list[-9][1]][-1])
                                                        else:
                                                            if len(Fixate[reduced_list[-10][1]]) > 1:
                                                                del (Fixate[reduced_list[-10][1]][-1])
                                                            else:
                                                                if len(Fixate[reduced_list[-11][1]]) > 1:
                                                                    del (Fixate[reduced_list[-11][1]][-1])
                                                                else:
                                                                    if len(Fixate[reduced_list[-12][1]]) > 1:
                                                                        del (Fixate[reduced_list[-12][1]][-1])
                                                                    else:
                                                                        if len(Fixate[reduced_list[-13][1]]) > 1:
                                                                            del (Fixate[reduced_list[-13][1]][-1])
                                                                        else:
                                                                            if len(Fixate[reduced_list[-14][1]]) > 1:
                                                                                del (Fixate[reduced_list[-14][1]][-1])
                                                                            else:
                                                                                if len(Fixate[
                                                                                           reduced_list[-15][1]]) > 1:
                                                                                    del (
                                                                                    Fixate[reduced_list[-15][1]][-1])
                                                                                else:
                                                                                    if len(Fixate[reduced_list[-16][
                                                                                        1]]) > 1:
                                                                                        del (
                                                                                        Fixate[reduced_list[-16][1]][
                                                                                            -1])
                                                                                    else:
                                                                                        if len(Fixate[reduced_list[-17][
                                                                                            1]]) > 1:
                                                                                            del (Fixate[
                                                                                                reduced_list[-17][1]][
                                                                                                -1])



                        new_fixation_1 = fixation_1
                        new_fixation_2 = fixation_2

                else:
                    # samples has to be added

                    add_position_1 = round((len(fixation_1)) / 2)
                    add_position_2 = round((len(fixation_2)) / 2)

                    new_fixation_1 = fixation_1[:add_position_1]
                    new_fixation_2 = fixation_2[:add_position_2]

                    for y in range(math.floor(add_number / 2)):


                        new_fixation_1.append(fixation_1[add_position_1])
                        new_fixation_2.append(fixation_2[add_position_2])

                    if add_number % 2 == 1:
                        new_fixation_2.append(fixation_2[add_position_2])

                    for f in range(len(fixation_1[add_position_1:])):
                        new_fixation_1.append(fixation_1[add_position_1:][f])

                    for f in range(len(fixation_2[add_position_2:])):
                        new_fixation_2.append(fixation_2[add_position_2:][f])

        if u == len(reduced_list) - 2:

            if not fixations_valid_flag:  # only if we dont find 2 fitting fixations, then we dont use the distance

                marker_of_second_fixation = -2

                fixation_1 = Fixate[reduced_list[-1][1]]
                fixation_2 = Fixate[reduced_list[-2][1]]

                if add_number < 0:

                    # samples has to be cutted
                    cut_number = - add_number

                    if cut_number / 2 < len(fixation_2):
                        for z in range(math.floor(cut_number / 2)):
                            cut_position_1 = round((len(fixation_1)) / 2)
                            cut_position_2 = round((len(fixation_2)) / 2)

                            del (fixation_1[cut_position_1])
                            del (fixation_2[cut_position_2])

                            new_fixation_1 = fixation_1
                            new_fixation_2 = fixation_2

                        if cut_number % 2 == 1:
                            cut_position_2 = round((len(fixation_2)) / 2)
                            del (fixation_2[cut_position_2])

                            new_fixation_1 = fixation_1
                            new_fixation_2 = fixation_2

                    else:  # just in case on fixation is to short

                        if len(fixation_2) > 5:
                            cut_number_2 = len(fixation_2) - 5
                            cut_number_1 = cut_number - cut_number_2
                        else:
                            cut_number_2 = 0
                            cut_number_1 = cut_number

                        for z in range(cut_number_2):
                            cut_position_2 = round((len(fixation_2)) / 2)
                            del (fixation_2[cut_position_2])

                        for z in range(cut_number_1):
                            cut_position_1 = round((len(fixation_1)) / 2)
                            if len(fixation_1) > 5:
                                del (fixation_1[cut_position_1])

                            else:

                                if len(Fixate[reduced_list[-3][1]]) > 4:

                                    del (Fixate[reduced_list[-3][1]][-1])

                                else:

                                    if len(Fixate[reduced_list[-4][1]]) > 3:

                                        del (Fixate[reduced_list[-4][1]][-1])

                                    else:

                                        if len(Fixate[reduced_list[-5][1]]) > 2:

                                            del (Fixate[reduced_list[-5][1]][-1])

                                        else:

                                            if len(Fixate[reduced_list[-6][1]]) > 2:

                                                del (Fixate[reduced_list[-6][1]][-1])

                                            else:

                                                if len(Fixate[reduced_list[-7][1]]) > 2:

                                                    del (Fixate[reduced_list[-7][1]][-1])

                                                else:

                                                    if len(Fixate[reduced_list[-8][1]]) > 1:

                                                        del (Fixate[reduced_list[-8][1]][-1])

                                                    else:

                                                        if len(Fixate[reduced_list[-9][1]]) > 1:

                                                            del (Fixate[reduced_list[-9][1]][-1])

                                                        else:

                                                            if len(Fixate[reduced_list[-10][1]]) > 1:

                                                                del (Fixate[reduced_list[-10][1]][-1])

                                                            else:

                                                                if len(Fixate[reduced_list[-11][1]]) > 1:

                                                                    del (Fixate[reduced_list[-11][1]][-1])

                                                                else:

                                                                    if len(Fixate[reduced_list[-12][1]]) > 1:

                                                                        del (Fixate[reduced_list[-12][1]][-1])

                                                                    else:

                                                                        if len(Fixate[reduced_list[-13][1]]) > 1:
                                                                            del (Fixate[reduced_list[-13][1]][-1])
                else:
                    # samples has to be added

                    add_position_1 = round((len(fixation_1)) / 2)
                    add_position_2 = round((len(fixation_2)) / 2)

                    new_fixation_1 = fixation_1[:add_position_1]
                    new_fixation_2 = fixation_2[:add_position_2]
                    for y in range(math.floor(add_number / 2)):
                        new_fixation_1.append(fixation_1[add_position_1])
                        new_fixation_2.append(fixation_2[add_position_2])

                    if add_number % 2 == 1:
                        new_fixation_2 = fixation_2[:add_position_2]
                        new_fixation_2.append(fixation_2[add_position_2])

                    for f in range(len(fixation_1[add_position_1:])):
                        new_fixation_1.append(fixation_1[add_position_1:][f])

                    for f in range(len(fixation_2[add_position_2:])):
                        new_fixation_2.append(fixation_2[add_position_2:][f])

    fixations_valid_flag = False

    for i in range(len(length_list)):

        if i == 0:
            for x in range(len(Fixate[0])):
                preprocessed_data_list.append(Fixate[0][x])
        else:
            if i == reduced_list[marker_of_second_fixation][1]:
                for v in range(len(new_fixation_2)):
                    preprocessed_data_list.append(new_fixation_2[v])
            if i == reduced_list[-1][1]:
                for b in range(len(new_fixation_1)):
                    preprocessed_data_list.append(new_fixation_1[b])

            if i != reduced_list[marker_of_second_fixation][1] and i != reduced_list[-1][1]:
                for j in range(len(Fixate[i])):
                    preprocessed_data_list.append(Fixate[i][j])



    if len(preprocessed_data_list) != 110:
        print("Wrong Length detected!!!", number, add_number, len(preprocessed_data_list))
        filename3 = str(number)


        name1 = os.path.dirname(
             sys.argv[0]) + "/Model Data/preprocessed learning data/" + filename3 + ".csv"

        with open(name1, mode='a', newline='') as data_log_objective_file:

            for i in range(0, len(Sample_Point)):
                data_log_demoraphic_writer = csv.writer(data_log_objective_file, delimiter=';')
                data_log_demoraphic_writer.writerow(Sample_Point[i])
        return
    else:
        return preprocessed_data_list



def logic():
    global actual_csv
    global file_list
    global sequence_list
    global preprocessed_data_list
    global number

    filename = ""

    loader()

    for a in range(len(file_list)):
        actual_csv = file_list[a]
        file_loader()
        sequence_generator()

        for i in range(len(sequence_list)):
            old_sequence_list = sequence_list[i]
            number = i

            #print("\nSequence Number", i)
            #print("with length", len(sequence_list[i]))

            Gesture_function(old_sequence_list)
            if len(preprocessed_data_list) == 110:

                 # exporting Data for analysis
                if old_sequence_list[0][7] == "1":
                    filename = "yes_gesture_preprocessed"
                if old_sequence_list[0][7] == "2":
                    filename = "no_gesture_preprocessed"
                if old_sequence_list[0][7] == "3":
                    filename = "up_gesture_preprocessed"
                if old_sequence_list[0][7] == "4":
                    filename = "down_gesture_preprocessed"
                if old_sequence_list[0][7] == "5":
                    filename = "next_gesture_preprocessed"
                if old_sequence_list[0][7] == "6":
                    filename = "back_gesture_preprocessed"

                name1 = os.path.dirname(
                    sys.argv[0]) + "/Model Data/preprocessed learning data/" + filename + ".csv"

                with open(name1, mode='a', newline='') as data_log_objective_file:

                    data_log_demoraphic_writer = csv.writer(data_log_objective_file, delimiter=';')
                    data_log_demoraphic_writer.writerow(preprocessed_data_list)

                preprocessed_data_list = []

                #print("\nData for Relabeling written in CSV!", filename)
            else:
                pass

if __name__ == "__main__":
    logic()








