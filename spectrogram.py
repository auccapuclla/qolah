import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
from parseVHF import VHFparser 
from plotVHF import get_phase

FS_ORIGINAL = 40e3
DOWN_SAMPLE_FS = 2e3

def resample_by_interpolation(signal, input_fs, output_fs):
    scale = output_fs / input_fs
    # calculate new length of sample
    n = round(len(signal) * scale)
    resampled_signal = np.interp(
        np.linspace(0.0, 1.0, n, endpoint=False),  # where to interpret
        np.linspace(0.0, 1.0, len(signal), endpoint=False),  # known positions
        signal,  # known data points
    )
    return resampled_signal

filename = '2024-01-11T15:55:37.470940_laser_chip_ULN00238_laser_driver_M00435617_laser_curr_392.8mA_port_number_5.bin'
file = Path('/mnt/nas-fibre-sensing/20231115_Cintech_Heterodyne_Phasemeter/' + filename)
if len(sys.argv) > 1:
    file = Path('/mnt/nas-fibre-sensing/20231115_Cintech_Heterodyne_Phasemeter/' + sys.argv[1] + '.bin')
    filename = sys.argv[1]

parsed = VHFparser(file, skip=0)
print(f"Debug {parsed.header} : = ")
phase = get_phase(parsed)

# down_sample_phase = resample_by_interpolation(phase, FS_ORIGINAL, DOWN_SAMPLE_FS)
vel = np.diff(phase)
# vel = resample_by_interpolation(vel, DOWN_SAMPLE_FS, 200)
plt.figure(figsize=(16,6))
nfft = 1024
plt.specgram(vel, NFFT=nfft, noverlap=int(nfft*0.9), scale='dB',Fs= 200, vmin=-70)
plt.ylim(0, 30)
plt.colorbar()
plt.show()