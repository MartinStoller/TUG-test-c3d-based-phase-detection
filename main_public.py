import os
import numpy as np
import utils.lab_tools as jc3d
from typing import Callable
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy import signal
from pylab import figure, text, scatter, show
from IPython.display import display
import pandas as pd

#DEFINE FUNCTIONS:
# Define local coordinate system of rectangular marker segments and get angles to global CS:
def getCsLoc_rectangle(array1, array2, array3, array4):
    # define CS
    origin = (array1 + array2 + array3 + array4) / 4
    z_ax_temp = (array1 + array2) / 2 - origin
    x_axis = (array1 + array4) / 2 - origin
    y_axis = np.cross(z_ax_temp, x_axis)
    z_axis = np.cross(x_axis, y_axis)

    alpha = []
    beta = []
    gamma = []
    cs_output = []
    for i in range(0, len(array1[:, 0])):
        x_ax_norm = x_axis[i] / np.linalg.norm(x_axis[i])
        y_ax_norm = y_axis[i] / np.linalg.norm(y_axis[i])
        z_ax_norm = z_axis[i] / np.linalg.norm(z_axis[i])

        cs = np.zeros((3, 3))
        cs[0][0] = x_ax_norm[0]
        cs[1][0] = x_ax_norm[1]
        cs[2][0] = x_ax_norm[2]
        cs[0][1] = y_ax_norm[0]
        cs[1][1] = y_ax_norm[1]
        cs[2][1] = y_ax_norm[2]
        cs[0][2] = z_ax_norm[0]
        cs[1][2] = z_ax_norm[1]
        cs[2][2] = z_ax_norm[2]

        cs_output.append(cs)

        # get angles:
        alpha.append((-1 * (np.arctan((cs_output[i][1][2] / cs_output[i][2][2])))) * 180 / np.pi)
        beta.append(np.arcsin(cs_output[i][0][2]) * 180 / np.pi)
        gamma.append((-1 * (np.arctan((cs_output[i][0][1] / cs_output[i][0][0])))) * 180 / np.pi)
    return cs_output, alpha, beta, gamma
    # alpha -> sagittal; beta -> frontal; gamma -> transversal


# Manipulate values, which are returned from from arcsin/arcos/arctan, so they are nicer to plot and easier to calculate with:
def plotableDegrees(list):
    for i in range(1, len(list)):
        if list[i] - list[i - 1] > 140:
            for a in range(i, len(list)):
                list[a] -= 180
        elif list[i] - list[i - 1] < -140:
            for a in range(i, len(list)):
                list[a] += 180


# butterworth lowpass filter:
def butterworth(array, order: int, fc: int, fs: int):
    b, a = signal.butter((order / 2), (fc / (fs / 2)), 'low')
    output = signal.filtfilt(b, a, array)
    return output


# finding plateaus/maxima/minima:
def detectPlateau(array, interval: range, threshold, fs) -> int:
    """
    We define an array and the interval for which we look at that array.
    Then we look at the derivative -> if it´s absolute value is below a certain threshold the event is triggered.
    In order to be less susceptible for 1-2 frame "jumps" in the data we look at the derivative over a small interval.
    The size of that interval is normalized on our sampling frequency (fs)
    """
    for i in interval:
        dy = np.diff(array[i:int(i + fs / 25)])
        if sum(dy) <= threshold and sum(dy) >= -threshold:
            return i


# detect SU_start
def detectSUstart(array, slopelength: int, minSlope) -> int:
    """
    SU start always starts with a pelvis anterior tilt. -> We look for that characteristic drop in the pelvis angle:
    ->find the first interval in a given array, with a given minimal slope (minslope) over a given period of time (slopelength)

    Which values for slopelength/minslope work well was found via trial and error and can be seen in the codeblock below
    were the function is used.
    """
    for i in range(0, 9999):
        dy = np.diff(array[i:i + slopelength])
        if all(n < -1 * minSlope for n in dy) == True:
            return i


# Detect first Heel off (-> Determains the beginn of walking phase (= end of standing up phase (SU_end)))
def detectHeelOff(interval: range) -> int:
    """
    To make sure random footmovements during sitting don´t trigger this event prematurely, the interval should start during
    maximal anterior tilt of the pelvis. That point can be easily detected with the detectPlateau() function (as you can see
    in the codeblock below where we actually use the function)

    The event gets triggered as soon as the heel(-> ankle markers) rise over a certain threshold compared to sitting
    reference. The size of the threshold (0.03) was determained by looking at the ankle_height graphs and seems to work well.
    """
    for i in interval:
        if ankle_height_left[i] > np.mean(ankle_height_left[0:50]) + 0.03 or ankle_height_right[i] > np.mean(
                ankle_height_right[0:50]) + 0.03:
            heelOffL = detectPlateau(ankle_height_left, interval=range(i, 0, -1), threshold=0.0025, fs=fs)
            heelOffR = detectPlateau(ankle_height_right, interval=range(i, 0, -1), threshold=0.0025, fs=fs)
            return min([heelOffL, heelOffR])


# Detect start and end of the Turn:
def detectTurn() -> int:
    """
    The start of the turn is triggered as soon as the first marker´s y-Coordinate (which should always be "Foot_Marker4")
    surpasses the y-Coordinate of the Turningpoint marker.
    The end is triggered when the last Marker´s y-Coordinate is smaller than the Turningpoint´s, which will always be
    "Foot_Marker3"
    """
    start = 0
    end = 0
    for i in range(0, len(point_data["TurningPointFloor"][:, 0])):
        if point_data["TurningPointFloor"][0, 1] <= point_data["foot_left_Marker4"][i, 1] or \
                point_data["TurningPointFloor"][0, 1] <= point_data["foot_right_Marker4"][i, 1]:
            start = i
            break
    for i in range(len(point_data["TurningPointFloor"][:, 0]) - 1, 0, -1):
        if point_data["TurningPointFloor"][i, 1] <= point_data["foot_left_Marker3"][i, 1] and \
                point_data["TurningPointFloor"][i, 1] <= point_data["foot_right_Marker3"][i, 1]:
            end = i
            break
    return [start, end]


# detect in which direction subject rotates during SD:
def hipRotationDirection() -> str:
    right = 0
    left = 0
    """
    We look for the typical values, which only occure during right/left turns.
    Example: If we have an angle of -360°  after sit-down this can only be because of 2* -180°(->right) turns.

    We need this information in order to detect the start of the sit-down movement (WB_end) properly.
    """
    for i in range(len(gamma_pelv) - 1, 0, -1):
        if gamma_pelv[i] < -300 or 125 > gamma_pelv[i] > 115 or i == 1:
            right = i
            break
    for i in range(len(gamma_pelv) - 1, 0, -1):
        if gamma_pelv[i] > 300 or -125 < gamma_pelv[i] < -115 or i == 1:
            left = i
            break
    if right > left:
        return "right"
    else:
        return "left"


# detect WB_end if the Sit-down movement includes a hip Rotation to the right:
def detectWB_endRightTurn() -> int:
    tmp = 0
    """
    The SD movement always starts with a pelvis rotation towards the chair. The exact start of this pelvis rotation is
    very hard to detect which is why we detect the ankle movement instead, which causes the pelvis rotation.

    -> Find a pelvis angle that makes it very clear the rotation movement has already started, but at the same time
    occurs realatively early during the SD phase (-> ~40° +/- the initial ~180° as we approach the chair)
        -> go from that point on backwards and detect the heelOff of the corresponding foot (pelvis rotation to the
        right must be caused by an ankle movement(-> heel off) of the left foot)
    """
    for i in range(len(gamma_pelv) - 1, 0, -1):
        if -218 > gamma_pelv[i] > -222 or 152 > gamma_pelv[i] > 148:
            tmp = i
            break
    for i in range(tmp, 0, -1):
        if ankle_height_left[i] > np.mean(ankle_height_left[0:50]) + 0.13:
            tmp = i
            break
    for i in range(tmp, 0, -1):
        if ankle_height_left[i] < np.mean(ankle_height_left[0:50]) + 0.03:
            WB_end = i
            return WB_end


# detect WB_end if the Sit-down movement includes a hip Rotation to the left:
def detectWB_endLeftTurn() -> int:
    # find the explaination at "detectWB_endRightTurn()"
    tmp = 0
    for i in range(len(gamma_pelv) - 1, 0, -1):
        if -152 < gamma_pelv[i] < -148 or 228 < gamma_pelv[i] < 232:
            tmp = i
            break
    for i in range(tmp, 0, -1):
        if ankle_height_right[i] > np.mean(ankle_height_right[0:50]) + 0.13:
            tmp = i
            break
    for i in range(tmp, 0, -1):
        if ankle_height_right[i] < np.mean(ankle_height_right[0:50]) + 0.03:
            WB_end = i
            return WB_end


# detect end of "Walk back Phase":
def detectWB_end() -> Callable:
    if hipRotationDirection() == "right":
        return detectWB_endRightTurn()
    else:
        return detectWB_endLeftTurn()


# detect end of Sit-down phase:
def detectSD_end() -> int:
    """
    angleCoefPelv is a value that combines the angular accaleration of the pelvis in all 3 dimensions. If that value is
    close to 0, the pelvis is at rest, which can only happen during sitting.
    The event is triggered when that value is close to 0 (threshold of 0.035 was found by trial and error) for a short
    time interval (1/20 sec.).
    """
    baseval = 0.035
    for i in range(WB_end, len(angleCoefPelv)):
        tmp = np.mean(angleCoefPelv[i - int(fs / 20):i])
        if tmp < baseval or i == len(angleCoefPelv) - 1:
            return i


# decode the movement from the filename:
def getMovementType() -> str:
    file_nr = int(file[16:18])
    if 0 < file_nr < 3:
        movement = "self-selected"
    if 2 < file_nr < 5:
        movement = "slow"
    if 4 < file_nr < 7:
        movement = "fast"
    if 6 < file_nr < 9:
        movement = "small steps"
    if 8 < file_nr < 11:
        movement = "hesitate"
    if 10 < file_nr:
        movement = "crouched"
    return movement


# get turning-direction at goal:
def turnGoalDirection() -> str:
    if gamma_pelv[TG_end] >= gamma_pelv[WG_end]:
        return "left"
    else:
        return "right"

#IMPORT DATA AND DETECT PHASES
directory = "tug_c3d_files"

ROOT = os.getcwd()
directory = os.path.join(ROOT, directory)

data_sum = {"subject": [], "filename": [], "movement_type": [], "turn_at_goal": [], "turn_at_sit": [],
            "tug_duration": [],
            "stand_up": [], "walk_1": [], "turn": [], "walk_2": [], "sit_down": []}

for file in os.listdir(directory):
    if file[-4:] == ".c3d" and not os.path.isdir(os.path.join(directory, file)):
        point_data, analog_data, meta_data = jc3d.load_c3d(os.path.join(directory, file))

        ######################### Define local Cs and get angles to global Cs #########################
        cs_pelvis, alpha_pelv, beta_pelv, gamma_pelv = getCsLoc_rectangle(point_data["pelvis_Marker1"],
                                                                          point_data["pelvis_Marker2"],
                                                                          point_data["pelvis_Marker3"],
                                                                          point_data["pelvis_Marker4"])

        ######################### Postprocess rawdata ########################
        frames = range(len(point_data["pelvis_Marker1"][:, 0]))

        plotableDegrees(alpha_pelv)
        plotableDegrees(beta_pelv)
        plotableDegrees(gamma_pelv)

        alpha_pelv = butterworth(alpha_pelv, order=2, fc=1, fs=200)
        beta_pelv = butterworth(beta_pelv, order=2, fc=1, fs=200)
        gamma_pelv = butterworth(gamma_pelv, order=2, fc=1, fs=200)

        ankle_height_left = butterworth(
            (point_data["foot_left_Marker1"][:, 2] + point_data["foot_left_Marker2"][:, 2]), 2, 5, 200)
        ankle_height_right = butterworth(
            (point_data["foot_right_Marker1"][:, 2] + point_data["foot_right_Marker2"][:, 2]), 2, 5, 200)

        ######################### Phase-detection ########################
        fs = 200  # sampling frequenzy(Hz)

        SU_start = detectSUstart(alpha_pelv, slopelength=int(fs / 4), minSlope=fs * 0.0001)
        SU_end = detectHeelOff(
            range(detectPlateau(alpha_pelv, range(SU_start + int(fs / 4), 9999), threshold=0.5, fs=fs), 9999))
        SU = (SU_end - SU_start) / fs

        WG_end, TG_end = detectTurn()
        WG = (WG_end - SU_end) / fs
        TG = (TG_end - WG_end) / fs

        WB_end = detectWB_end()
        WB = (WB_end - TG_end) / fs

        angleCoefPelv = butterworth((abs(np.diff(alpha_pelv)) + abs(np.diff(beta_pelv)) + abs(np.diff(gamma_pelv))),
                                    order=2, fc=1, fs=fs)
        SD_end = detectSD_end()
        SD = (SD_end - WB_end) / fs

        total = (SD_end - SU_start) / fs

        ######################### Pandas DataFrame ########################
    data_sum["subject"].append(file[8:10])
    data_sum["filename"].append(str(file))
    data_sum["movement_type"].append((getMovementType()))
    data_sum["turn_at_goal"].append(turnGoalDirection())
    data_sum["turn_at_sit"].append(hipRotationDirection())
    data_sum["tug_duration"].append(total)
    data_sum["stand_up"].append(SU)
    data_sum["walk_1"].append(WG)
    data_sum["turn"].append(TG)
    data_sum["walk_2"].append(WB)
    data_sum["sit_down"].append(SD)

#         ######################### Plot phase-detection ########################
#         a1 = plt.subplot2grid((2, 2), (0, 0))
#         a2 = plt.subplot2grid((2, 2), (0, 1))

#         # Sagittal (-> alpha):
#         a1.plot(frames, alpha_pelv, linewidth=0.8, markersize=0.00001, color='black')
#         a1.axvline(x=SU_start, ymin=0, ymax=1, linewidth=0.8, markersize=0.00001, color='red')
#         a1.axvline(x=SU_end, ymin=0, ymax=1, linewidth=0.8, markersize=0.00001, color='red')
#         a1.text((SU_start + SU_end) / 2, min(alpha_pelv) + 5, 'SU: ' + str(format(SU, '.2f')) + "s", ha='center',
#                 va='center', color='red')
#         a1.axvline(x=WG_end, ymin=0, ymax=1, linewidth=0.8, markersize=0.00001, color='red')
#         a1.text((WG_end + SU_end) / 2, min(alpha_pelv) + 5, 'WG: ' + str(format(WG, '.2f')) + "s", ha='center',
#                 va='center', color='red')
#         a1.axvline(x=TG_end, ymin=0, ymax=1, linewidth=0.8, markersize=0.00001, color='red')
#         a1.text((WG_end + TG_end) / 2, min(alpha_pelv) + 5, 'TG: ' + str(format(TG, '.2f')) + "s", ha='center',
#                 va='center', color='red')
#         a1.axvline(x=WB_end, ymin=0, ymax=1, linewidth=0.8, markersize=0.00001, color='red')
#         a1.text((WB_end + TG_end) / 2, min(alpha_pelv) + 5, 'SU: ' + str(format(WB, '.2f')) + "s", ha='center',
#                 va='center', color='red')
#         a1.axvline(x=SD_end, ymin=0, ymax=1, linewidth=0.8, markersize=0.00001, color='red')
#         a1.text((WB_end + SD_end) / 2, min(alpha_pelv) + 5, 'SD: ' + str(format(SD, '.2f')) + "s", ha='center',
#                 va='center', color='red')
#         a1.text(SU_start - fs * 2, min(alpha_pelv), 'Total: ' + str(format(total, '.2f')) + "s", ha='center',
#                 va='center', color='red')
#         a1.set_title('Sagittal movement pelvis', color='black', size=15)
#         a1.set_xlabel('Frames', color='black', size=12)
#         a1.set_ylabel('Angle [°]', color='black', size=12)

#         a2.plot(frames, gamma_pelv, linewidth=0.8, markersize=0.00001, color='black')
#         a2.axvline(x=WG_end, ymin=0, ymax=1, linewidth=0.8, markersize=0.00001, color='red')
#         a2.axvline(x=SU_end, ymin=0, ymax=1, linewidth=0.8, markersize=0.00001, color='red')
#         a2.axvline(x=SU_start, ymin=0, ymax=1, linewidth=0.8, markersize=0.00001, color='red')

#         a2.text((SU_start + SU_end) / 2, min(gamma_pelv) + 5, 'SU: ' + str(format(SU, '.2f')) + "s", ha='center',
#                 va='center', color='red')
#         a2.text((WG_end + SU_end) / 2, min(gamma_pelv) - 5, 'WG: ' + str(format(WG, '.2f')) + "s", ha='center',
#                 va='center', color='red')
#         a2.axvline(x=TG_end, ymin=0, ymax=1, linewidth=0.8, markersize=0.00001, color='red')
#         a2.text((WB_end + TG_end) / 2, min(gamma_pelv) - 5, 'WB: ' + str(format(WB, '.2f')) + "s", ha='center',
#                 va='center', color='red')
#         a2.axvline(x=WB_end, ymin=0, ymax=1, linewidth=0.8, markersize=0.00001, color='red')
#         a2.text((WG_end + TG_end) / 2, min(gamma_pelv) - 5, 'TG: ' + str(format(TG, '.2f')) + "s", ha='center',
#                 va='center', color='red')
#         a2.axvline(x=SD_end, ymin=0, ymax=1, linewidth=0.8, markersize=0.00001, color='red')
#         a2.text((WB_end + SD_end) / 2, min(gamma_pelv) + 5, 'SD: ' + str(format(SD, '.2f')) + "s", ha='center',
#                 va='center', color='red')
#         a2.set_title('Transversal movement pelvis', color='black', size=15)
#         plt.xlabel('Frames', color='black', size=12)
#         plt.ylabel('Angle [°]', color='black', size=12)

#         plt.suptitle(str(file), color='black', size=25, weight='bold')
#         plt.tight_layout()
#         plt.show()

df = pd.DataFrame(data_sum)
pd.set_option('display.max_rows', 125)
display(df)
df.to_csv(r'csv_files\pandas_dataframe.csv', index=False)