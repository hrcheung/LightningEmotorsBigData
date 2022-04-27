import gzip
import os
import matplotlib.pyplot as plt
import pandas as pd
from sksurv.nonparametric import kaplan_meier_estimator
import numpy as np


def preprocess(x_axis, category):
    (x_key, x_transform) = x_axis
    (c_keys, category_transform) = category
    directory = 'zip/school_data_2'
    # Use a breakpoint in the code line below to debug your script.
    result = []
    data_x = []
    count = 0
    for filename in os.listdir(directory):
        x_i = 0
        category_i = {}
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
                            for key in c_keys:
                                if d[i] == key:
                                    category_i[key] = i

                else:
                    available_energy = int(d[available_energy_i])
                    charge_pct = float(d[charge_pct_i])

                    x_curr = int(d[x_i])
                    if x_start == 0:
                        x_start = x_curr

                    if charge_pct == 0 or available_energy == 0:
                        continue

                    category_values = {}
                    for key in category_i.keys():
                        category_values[key] = d[category_i[key]]

                    available_energy_arr.append(available_energy)
                    if len(available_energy_arr) > sliding_window_size:
                        del available_energy_arr[0]
                    if len(available_energy_arr) == sliding_window_size:
                        curr_available_energy_avg = sum(available_energy_arr) / sliding_window_size
                        processed.append((x_curr, charge_pct, curr_available_energy_avg, category_values))

            processed.sort(key=lambda t: t[0])

            category_arr = []
            for (x_curr, charge_pct, available_energy, category_values) in processed:
                if max_available_energy < available_energy:
                    max_available_energy = available_energy
                    x_start = x_curr
                    reached_failure_state = False

                x_diff = x_transform(x_curr, x_start)

                available_energy_pct = (available_energy / (max_available_energy + 1)) * 100
                charge_diff = charge_pct - available_energy_pct

                if not reached_failure_state and charge_diff > 20:
                    reached_failure_state = True
                    days_to_failure = x_diff

                category_arr.append(category_transform(category_values))

            if days_to_failure == 0:
                days_to_failure = x_transform(x_curr, x_start)

            if days_to_failure < 0 or days_to_failure > 600:
                continue

            if len(category_arr) == 0:
                continue

            result.append((reached_failure_state, int(days_to_failure)))
            data_x.append(int((sum(category_arr) / len(category_arr)) * 100) / 100)
            count = count + 1

    print(result)
    print(len(result))
    data_y = np.array(result, dtype=[('Status', '?'), ("x_label", np.int32)])
    data_x = np.array(data_x, dtype=[('category', np.float)])
    return data_x, data_y


def estimate_survival_hml():
    def transform(x_curr, x_start):
        x = x_curr - x_start
        return int(x / (60 * 60 * 24))

    def category_transform(obj):
        if "hvbatt_min_cell_temp_c" not in obj:
            return 0
        min_temp = obj["hvbatt_min_cell_temp_c"]
        max_temp = obj["hvbatt_max_cell_temp_c"]
        if min_temp == "" or max_temp == "":
            return 0
        min_temp = int(float(min_temp))
        max_temp = int(float(max_temp))
        # some csv's got the min and max swapped around
        if min_temp > max_temp:
            cache = min_temp
            min_temp = max_temp
            max_temp = cache
        return max_temp - min_temp

    (data_x, data_y) = preprocess(("RptTimestamp", transform),
                                  (["hvbatt_min_cell_temp_c", "hvbatt_max_cell_temp_c"], category_transform))
    dframe = pd.DataFrame(data_y)
    dcframe = pd.DataFrame(data_x)

    for (min, max) in ((0, 1), (1, 3), (3, 99)):
        time, survival_prob = kaplan_meier_estimator(
            dframe["Status"][(min < dcframe["category"]) & (dcframe["category"] < max)],
            dframe["x_label"][(min < dcframe["category"]) & (dcframe["category"] < max)])

        plt.step(time, survival_prob, where="post", label="%i - %i" % (min, max))

    plt.ylabel("est. probability of survival")
    plt.xlabel("time in days")
    plt.legend(loc="best")
    plt.show()


def estimate_survival_hml_in_km():
    def transform(x_curr, x_start):
        x = x_curr - x_start
        return int(x / 1000)

    def category_transform(obj):
        if "hvbatt_min_cell_temp_c" not in obj:
            return 0
        min_temp = obj["hvbatt_min_cell_temp_c"]
        max_temp = obj["hvbatt_max_cell_temp_c"]
        if min_temp == "" or max_temp == "":
            return 0
        min_temp = int(float(min_temp))
        max_temp = int(float(max_temp))
        # some csv's got the min and max swapped around
        if min_temp > max_temp:
            cache = min_temp
            min_temp = max_temp
            max_temp = cache
        return max_temp - min_temp

    (data_x, data_y) = preprocess(("odometer_distance_m", transform),
                                  (["hvbatt_min_cell_temp_c", "hvbatt_max_cell_temp_c"], category_transform))

    dframe = pd.DataFrame(data_y)
    dcframe = pd.DataFrame(data_x)

    for (min, max) in ((0, 1), (1, 3), (3, 5), (5, 20)):
        dstatus = dframe["Status"][(min < dcframe["category"]) & (dcframe["category"] < max)]
        print(len(dstatus))
        time, survival_prob = kaplan_meier_estimator(
            dstatus,
            dframe["x_label"][(min < dcframe["category"]) & (dcframe["category"] < max)])

        plt.step(time, survival_prob, where="post", label="avg temp diff in °C %i - %i" % (min, max))

    plt.ylabel("est. probability of survival")
    plt.xlabel("distance travelled in km")
    plt.legend(loc="best")
    plt.show()
