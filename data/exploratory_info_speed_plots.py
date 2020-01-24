import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

cwd = os.getcwd()
stations = filter(os.path.isdir, os.listdir(cwd))
sr = 48000

stations = [f for f in stations if not f.startswith('.')]

speeds = []
for station in stations:
    root_path = f"{cwd}/{station}"
    files = os.listdir(root_path)
    csv_path = [f for f in files if f.endswith('.csv')]

    for p in csv_path:
        file = pd.read_csv(root_path + '/' + p, sep=';', decimal=',', dtype=np.float32)
        speeds = file.speedInMeterPerSeconds
        speeds = speeds[(speeds > 2)]

        # if len(speeds) > 200:
        #     print(station, p)

        # if len(speeds) < 4:
        #     print(station, p)

        # if not np.alltrue(speeds >= 0):
        #     print(station, p)
        #     print('has zero value!')

        speeds_kmh = speeds * 3.6
        plt.plot(speeds_kmh);
    plt.title(f'{station} speeds');
    plt.xlabel('axleNumber')
    plt.ylabel('speed in km/h')
    plt.show()
