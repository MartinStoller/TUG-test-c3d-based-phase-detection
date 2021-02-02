import os
import numpy as np
import utils.lab_tools as jc3d
from typing import Callable
import matplotlib.pyplot as plt
import cs_functions_ms as csf
from IPython.display import display
import pandas as pd
import plotly.graph_objects as go
from scipy import signal
from pylab import figure, text, scatter, show
import pickle


######################### Define Methods #########################
def detectHeelOff(interval: range) -> int:
    for i in interval:
        if ankle_height_left[i] > np.mean(ankle_height_left[0:50]) + 0.03 or ankle_height_right[i] > np.mean(
                ankle_height_right[0:50]) + 0.03:
            heelOffL = csf.detectPlateau(ankle_height_left, interval=range(i, 0, -1), threshold=0.0025, fs=fs)
            heelOffR = csf.detectPlateau(ankle_height_right, interval=range(i, 0, -1), threshold=0.0025, fs=fs)
            return min([heelOffL, heelOffR])


def detectTurn() -> int:
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


def hipRotationDirection() -> str:
    right = 0
    left = 0
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


def detectWB_endRightTurn() -> int:
    tmp = 0
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


def detectWB_endLeftTurn() -> int:
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


def detectWB_end() -> Callable:
    if hipRotationDirection() == "right":
        return detectWB_endRightTurn()
    else:
        return detectWB_endLeftTurn()


def detectSD_end() -> int:
    baseval = 0.035
    for i in range(WB_end, len(angleCoefPelv)):
        tmp = np.mean(angleCoefPelv[i - int(fs / 20):i])
        if tmp < baseval or i == len(angleCoefPelv) - 1:
            return i

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

def turnGoalDirection() -> str:
    if gamma_pelv[TG_end] >= gamma_pelv[WG_end]:
        return "left"
    else:
        return "right"

######################### Import Data #########################
directory = "tug_c3d_files"

ROOT = os.getcwd()
directory = os.path.join(ROOT, directory)

for file in os.listdir(directory):
    if file[-4:] == ".c3d" and not os.path.isdir(os.path.join(directory, file)):
        point_data, analog_data, meta_data = jc3d.load_c3d(os.path.join(directory, file))

        ######################### Define local Cs and get angles to global Cs #########################
        cs_pelvis, alpha_pelv, beta_pelv, gamma_pelv = csf.getCsLoc_rectangle(point_data["pelvis_Marker1"],
                                                                              point_data["pelvis_Marker2"],
                                                                              point_data["pelvis_Marker3"],
                                                                              point_data["pelvis_Marker4"])

        ######################### Postprocess rawdata ########################
        frames = range(len(point_data["pelvis_Marker1"][:, 0]))

        csf.plotableDegrees(alpha_pelv)
        csf.plotableDegrees(beta_pelv)
        csf.plotableDegrees(gamma_pelv)

        alpha_pelv = csf.butterworth(alpha_pelv, order=2, fc=1, fs=200)
        beta_pelv = csf.butterworth(beta_pelv, order=2, fc=1, fs=200)
        gamma_pelv = csf.butterworth(gamma_pelv, order=2, fc=1, fs=200)

        ankle_height_left = csf.butterworth(
            (point_data["foot_left_Marker1"][:, 2] + point_data["foot_left_Marker2"][:, 2]), 2, 5, 200)
        ankle_height_right = csf.butterworth(
            (point_data["foot_right_Marker1"][:, 2] + point_data["foot_right_Marker2"][:, 2]), 2, 5, 200)

        ######################### Phase-detection ########################
        fs = 200  # sampling frequenzy(Hz)

        SU_start = csf.detectSUstart(alpha_pelv, slopelength=int(fs/4), minSlope=fs * 0.0001)
        SU_end = detectHeelOff(
            range(csf.detectPlateau(alpha_pelv, range(SU_start + int(fs/4), 9999), threshold=0.5, fs=fs), 9999))
        SU = (SU_end - SU_start) / fs

        WG_end, TG_end = detectTurn()
        WG = (WG_end - SU_end) / fs
        TG = (TG_end - WG_end) / fs

        WB_end = detectWB_end()
        WB = (WB_end - TG_end) / fs

        angleCoefPelv = csf.butterworth((abs(np.diff(alpha_pelv)) + abs(np.diff(beta_pelv)) + abs(np.diff(gamma_pelv))),
                                        order=2, fc=1, fs=fs)
        SD_end = detectSD_end()
        SD = (SD_end - WB_end) / fs

        total = (SD_end - SU_start) / fs

        ######################### Plot phase-detection ########################
        a1 = plt.subplot2grid((2, 2), (0, 0))
        a2 = plt.subplot2grid((2, 2), (0, 1))

        # Sagittal (-> alpha):
        a1.plot(frames, alpha_pelv, linewidth=0.8, markersize=0.00001, color='black')
        a1.axvline(x=SU_start, ymin=0, ymax=1, linewidth=0.8, markersize=0.00001, color='red')
        a1.axvline(x=SU_end, ymin=0, ymax=1, linewidth=0.8, markersize=0.00001, color='red')
        a1.text((SU_start + SU_end) / 2, min(alpha_pelv) + 5, 'SU: ' + str(format(SU, '.2f')) + "s", ha='center',
                va='center', color='red')
        a1.axvline(x=WG_end, ymin=0, ymax=1, linewidth=0.8, markersize=0.00001, color='red')
        a1.text((WG_end + SU_end) / 2, min(alpha_pelv) + 5, 'WG: ' + str(format(WG, '.2f')) + "s", ha='center',
                va='center', color='red')
        a1.axvline(x=TG_end, ymin=0, ymax=1, linewidth=0.8, markersize=0.00001, color='red')
        a1.text((WG_end + TG_end) / 2, min(alpha_pelv) + 5, 'TG: ' + str(format(TG, '.2f')) + "s", ha='center',
                va='center', color='red')
        a1.axvline(x=WB_end, ymin=0, ymax=1, linewidth=0.8, markersize=0.00001, color='red')
        a1.text((WB_end + TG_end) / 2, min(alpha_pelv) + 5, 'SU: ' + str(format(WB, '.2f')) + "s", ha='center',
                va='center', color='red')
        a1.axvline(x=SD_end, ymin=0, ymax=1, linewidth=0.8, markersize=0.00001, color='red')
        a1.text((WB_end + SD_end) / 2, min(alpha_pelv) + 5, 'SD: ' + str(format(SD, '.2f')) + "s", ha='center',
                va='center', color='red')
        a1.text(SU_start - fs * 2, min(alpha_pelv), 'Total: ' + str(format(total, '.2f')) + "s", ha='center',
                va='center', color='red')
        a1.set_title('Sagittal movement pelvis', color='black', size=15)
        a1.set_xlabel('Frames', color='black', size=12)
        a1.set_ylabel('Angle [°]', color='black', size=12)

        a2.plot(frames, gamma_pelv, linewidth=0.8, markersize=0.00001, color='black')
        a2.axvline(x=WG_end, ymin=0, ymax=1, linewidth=0.8, markersize=0.00001, color='red')
        a2.axvline(x=SU_end, ymin=0, ymax=1, linewidth=0.8, markersize=0.00001, color='red')
        a2.axvline(x=SU_start, ymin=0, ymax=1, linewidth=0.8, markersize=0.00001, color='red')

        a2.text((SU_start + SU_end) / 2, min(gamma_pelv) + 5, 'SU: ' + str(format(SU, '.2f')) + "s", ha='center',
                va='center', color='red')
        a2.text((WG_end + SU_end) / 2, min(gamma_pelv) - 5, 'WG: ' + str(format(WG, '.2f')) + "s", ha='center',
                va='center', color='red')
        a2.axvline(x=TG_end, ymin=0, ymax=1, linewidth=0.8, markersize=0.00001, color='red')
        a2.text((WB_end + TG_end) / 2, min(gamma_pelv) - 5, 'WB: ' + str(format(WB, '.2f')) + "s", ha='center',
                va='center', color='red')
        a2.axvline(x=WB_end, ymin=0, ymax=1, linewidth=0.8, markersize=0.00001, color='red')
        a2.text((WG_end + TG_end) / 2, min(gamma_pelv) - 5, 'TG: ' + str(format(TG, '.2f')) + "s", ha='center',
                va='center', color='red')
        a2.axvline(x=SD_end, ymin=0, ymax=1, linewidth=0.8, markersize=0.00001, color='red')
        a2.text((WB_end + SD_end) / 2, min(gamma_pelv) + 5, 'SD: ' + str(format(SD, '.2f')) + "s", ha='center',
                va='center', color='red')
        a2.set_title('Transversal movement pelvis', color='black', size=15)
        plt.xlabel('Frames', color='black', size=12)
        plt.ylabel('Angle [°]', color='black', size=12)

    # arr = [[SU], [WG], [TG], [WB], [SD]]
    # collabel = ("Stand up", "Walk to goal", "Turn", "Walk back", "Sit down")
    # the_table = a3.table(cellText=arr,colLabels=collabel,loc='center')
    # fig = go.Figure(data=[go.Table(header=dict(values=["Phase", "Duration [s]"]),
    #                                cells=dict(
    #                                    values=[["Stand Up", "Walk towards Goal", "Turn", "Walk back", "Sit down"],
    #                                            [SU, WG, TG, WB, SD]]))
    #                       ])
    # fig.show()

    # plt.suptitle(str(file), color='black', size=25, weight='bold')
    # plt.tight_layout()
    # plt.show()

    data_sum = {"subject": [], "filename": [], "movement_type": [], "turn_at_goal": [], "turn_at_sit": [],
            "tug_duration": [], "stand_up": [], "walk_1": [], "turn": [], "walk_2": [], "sit_down": []}

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

    df = pd.DataFrame(data_sum)
    display(df)
    ######################### Pickle export ########################
    # phases = {"Stand up": SU, "Walk towards goal": WG, "Turn:": TG, "Walk back": WB, "Sit down": SD}
    # pickle.dump(phases, open("c3d_phases", "wb"))


    #pandas table (jede zeile = eine Messung) -> filename phasenlängen
    #code kommentieren damit andere Leute wissen was los ist
    # phasendefinitionen, beschreiben von funktionen und probelemen
    # -> als csv/pickle abspeichern
    ## sanity check für 10m walk test. Verfeinern (über zusammenhänge zwischen variablen... man gibt winkelverläufe und output ist ob das plausibel ist. AUch klassifizieren was der Fehler ist oder nicht -> annomaly detection)
#( Pathologische Patienten daten -> step-detection stabiler hinbekommen. (unterstütz durch machinelearning vll))
# phasenerkennung TUG mit machinelearning
# diagnose modelle (ai basiert) auf basis von gangmustern
# modell trainieren, das synthetische daten von pathologischen gangbildern generiert, die man dann nutzt um eine pathologische klassifikationsmodell zu trainieren
