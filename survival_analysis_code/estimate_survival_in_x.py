import gzip
import os
import matplotlib.pyplot as plt
import pandas as pd
from sksurv.nonparametric import kaplan_meier_estimator
import numpy as np


def preprocess(x_axis):
    (x_key, x_transform) = x_axis
    directory = 'zip/school_data_2'
    # Use a breakpoint in the code line below to debug your script.
    result = []
    count = 0
    for filename in os.listdir(directory):
        x_i = 0
        available_energy_i = 0
        charge_pct_i = 0
        max_available_energy = 0
        x_start = 0
        days_to_failure = 0
        reached_failure_state = False
        x_curr = 0
        available_energy_arr = []
        sliding_window_size = 50

        if count > 220:
            break

        f = os.path.join(directory, filename)

        with gzip.GzipFile(f) as myZip:
            processed = []
            for line in myZip.readlines():
                d = line.decode().split(',')
                # x[len(x)-1] = x[len(x)-1].rstrip('\n')

                if d[0] == '':
                    for i in range(len(d)):
                        if d[i] == 'battery_state_of_charge_pct':
                            charge_pct_i = i
                        elif d[i] == 'battery_available_energy_wh':
                            available_energy_i = i
                        elif d[i] == x_key:
                            x_i = i
                else:
                    available_energy = int(d[available_energy_i])
                    charge_pct = float(d[charge_pct_i])

                    x_curr = int(d[x_i])
                    if x_start == 0:
                        x_start = x_curr

                    if charge_pct == 0 or available_energy == 0:
                        continue

                    available_energy_arr.append(available_energy)
                    if len(available_energy_arr) > sliding_window_size:
                        del available_energy_arr[0]
                    if len(available_energy_arr) == sliding_window_size:
                        curr_available_energy_avg = sum(available_energy_arr) / sliding_window_size
                        processed.append((x_curr, charge_pct, curr_available_energy_avg))

            processed.sort(key=lambda t: t[0])

            for (x_curr, charge_pct, available_energy) in processed:
                if max_available_energy < available_energy:
                    max_available_energy = available_energy
                    x_start = x_curr
                    reached_failure_state = False

                x_diff = x_transform(x_curr, x_start)

                available_energy_pct = (available_energy / (max_available_energy + 1)) * 100
                charge_diff = charge_pct - available_energy_pct

                if not reached_failure_state and charge_diff > 40:
                    reached_failure_state = True
                    days_to_failure = x_diff

            if days_to_failure == 0:
                days_to_failure = x_transform(x_curr, x_start)

            if days_to_failure < 0 or days_to_failure > 600:
                continue
            result.append((reached_failure_state, int(days_to_failure)))
            count = count + 1
    print(result)
    print(len(result))
    r2 = np.array(result, dtype=[('Status', '?'), ("x_label", np.int32)])
    return r2


def estimate_survival_in_days():
    def transform(x_curr, x_start):
        x = x_curr - x_start
        return int(x / (60 * 60 * 24))

    narray = preprocess(("RptTimestamp", transform))
    dframe = pd.DataFrame(narray)
    time, survival_prob = kaplan_meier_estimator(dframe["Status"], dframe["x_label"])

    plt.step(time, survival_prob, where="post")
    plt.ylabel("est. probability of survival")
    plt.xlabel("time in days")
    plt.show()


def estimate_survival_in_km():
    def transform(x_curr, x_start):
        x = x_curr - x_start
        return int(x / 1000)
    narray = preprocess(("odometer_distance_m", transform))
    dframe = pd.DataFrame(narray)
    print(dframe["Status"][dframe["Status"] == False])
    time, survival_prob = kaplan_meier_estimator(dframe["Status"], dframe["x_label"])

    plt.step(time, survival_prob, where="post")
    plt.ylabel("est. probability of survival")
    plt.xlabel("distance travelled in km")
    plt.show()
