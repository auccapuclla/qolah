# With the generation of Figure 37 in Miguel's report much better understood, 
# it is now possible to generate a array-cropped output to pass to 
# GNUPlot/matplotlib for SVG format.

import os, sys
from pathlib import Path
module_path = str(Path(__file__).parents[2])
if module_path not in sys.path:
    sys.path.append(module_path)
from matplotlib.ticker import FuncFormatter
from datetime import datetime, timedelta
from datetime import timedelta
from matplotlib import colors
from matplotlib import gridspec
from matplotlib import mlab
from matplotlib import patches
from matplotlib import pyplot as plt
from matplotlib import transforms
import math
import numpy as np
import numpy.typing as npt
from obspy.core import Stream, Trace, Stats
from obspy.imaging.spectrogram import spectrogram
from parseVHF import VHFparser  # relative import
from plotVHF import get_radius, get_spec, get_phase, plot_rad_spec
from pathlib import Path
from pynverse import inversefunc
from typing import Callable

def block_avg(my_arr: np.ndarray, N: int):
    """Returns a block average of 1D my_arr in blocks of N."""
    if N == 1:
        return my_arr
    return np.mean(my_arr.reshape(np.shape(my_arr)[0]//N, N), axis=1)

def block_avg_tail(my_arr: np.ndarray, N: int):
    """Block averages the main body, up till last block that might 
    contain less than N items. Assumes 1D."""
    # Check for 1D array
    assert my_arr.ndim == 1

    if N == 1:
        return my_arr
    if np.size(my_arr) % N == 0:
        return block_avg(my_arr, N)
    else:
        print("Chosen indices does not give nice block!")
        result = np.zeros(np.size(my_arr)//N + 1)
        result[:-1] = block_avg(my_arr[:np.size(my_arr)//N * N ], N)
        result[-1] = np.mean(my_arr[np.size(my_arr)//N * N :])
        return result

def crop_indices(t_start, t_end, sampl_freq, squash_factor, diff_delta):
    """Obtain index that contains start and end time that gives nice 
    blocks for block averaging.
    
    Args
    -----
    t_start: int
        time start in seconds. Ideally in seconds.
    t_end: int
    sampl_freq: float
        sampling frequency
    squash_factor: int
        Denotes the total compression factor that will be utilised in 
        generating spectrogram. For example, if block averaging is done 
        on N = 5 then N = 10, squash_factor = 50.
    diff_delta: int
        since np.diff gives 1 less element, specify diff_delta to offset
        end_index
    """
    i_s = t_start * sampl_freq
    i_start = int(i_s)
    if abs(i_start - i_s) > 0.5:
        i_start -= 1
    delta_seconds = t_end - t_start
    i_d = delta_seconds * sampl_freq
    i_delta = int(i_d)
    if abs(i_d - i_delta) > 0.5:
        i_delta += 1
    # + 1 to account for [,) slicing
    i_end = i_start + i_delta + squash_factor + diff_delta
    return (i_start, i_end)

def spectrogram_data_array(data, samp_rate, wlen=None, per_lap=0.9, dbscale=False, log=False):
    """Returns f, t, Sxx, ax_extent while using obspy internals (wlen and overlap).
    
    Args
    -----
    data:
    samp_rate:
    wlen:
    per_lap:
    dbscale:
    log:

    Returns
    -----
    specgram:
    freq:
    time:
    ax_extent: Bounding box to be used directly in pyplot's imshow
    """

    # Utilise Obspy 1.4 internal code, up til matplotlib.specgram, except
    # guard clauses as given within the internals are to be given explicitly

    def _nearest_pow_2(x):
        """
        Find power of two nearest to x

        >>> _nearest_pow_2(3)
        2.0
        >>> _nearest_pow_2(15)
        16.0

        :type x: float
        :param x: Number
        :rtype: int
        :return: Nearest power of 2 to x
        """
        a = math.pow(2, math.ceil(np.log2(x)))
        b = math.pow(2, math.floor(np.log2(x)))
        if abs(a - x) < abs(b - x):
            return a
        else:
            return b
            
    # enforce float for samp_rate
    samp_rate = float(samp_rate)
    # modified guard clause
    if not wlen:
        raise ValueError("wlen is to be explicit!")
    npts = len(data)
    nfft = int(_nearest_pow_2(wlen * samp_rate))

    if npts < nfft:
        msg = (f'Input signal too short ({npts} samples, window length '
               f'{wlen} seconds, nfft {nfft} samples, sampling rate '
               f'{samp_rate} Hz)')
        raise ValueError(msg)
    # modified guard clause
    mult = None
    nlap = int(nfft * float(per_lap))
    data = data - data.mean()
    specgram, freq, time = mlab.specgram(data, Fs=samp_rate, NFFT=nfft,
                                         pad_to=mult, noverlap=nlap)
    if dbscale:
        specgram = 10 * np.log10(specgram[1:, :])
    else:
        specgram = np.sqrt(specgram[1:, :])
    freq = freq[1:]

    # calculate half bin width
    halfbin_time = (time[1] - time[0]) / 2.0
    halfbin_freq = (freq[1] - freq[0]) / 2.0
    ax_extent = (time[0] - halfbin_time, time[-1] + halfbin_time,
                 freq[0] - halfbin_freq, freq[-1] + halfbin_freq)
    return specgram, freq, time, ax_extent

cond_type = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.bool_]]
def spectrogram_crop(Sxx: np.ndarray, f: np.ndarray, t: np.ndarray,
                     f_cond: cond_type, t_cond: cond_type):
    """Function to crop Sxx to within f and t bounds (in natural units)."""

    # Allow for accepting no filtering
    if f_cond == None:
        f_cond = lambda x: np.full(np.shape(x), True)
    if t_cond == None:
        t_cond = lambda x: np.full(np.shape(x), True)
    if f_cond == None and t_cond == None:
        return Sxx, f, t
    return Sxx[f_cond(f), ...][..., t_cond(t)], f[f_cond(f)], t[t_cond(t)]

def main():
    view_const = 5
    sparse_num_phase = 1
    sparse_num_velocity = 1

    file1 = Path('/scratch2/e0667612/qolah/2024-01-23T08:23:17.255959_laser_chip_ULN00238_laser_driver_M00435617_laser_curr_392.8mA_port_number_5.bin')
    parsed1 = VHFparser(file1, skip=0)
    phase1 = get_phase(parsed1)
    print(f'sample freq: {parsed1.header["sampling freq"]}')
    print(f"Debug {parsed1.header} : = ")
    phase1_avg = block_avg_tail(phase1, sparse_num_phase)
    velocity1 = np.diff(phase1_avg) * (parsed1.header["sampling freq"]/sparse_num_phase) # * 1550e-9

    # account for butchering of parsed headers
    parsed1.header["sampling freq"] /= (sparse_num_phase * sparse_num_velocity) # after the dust has settled
    t = np.arange(len(velocity1)) / parsed1.header["sampling freq"]

    shoehorn_header = Stats()
    print(f'final sample freq {parsed1.header["sampling freq"]}')
    shoehorn_header.sampling_rate = parsed1.header['sampling freq']
    shoehorn_header.starttime = parsed1.header['Time start']
    shoehorn_header.network = 'QITLAB'
    shoehorn_header.location = 'NUS'
    shoehorn_header.npts = len(velocity1)
    Sxx, f, t, ax_extent = spectrogram_data_array(velocity1, 
                                                  shoehorn_header.sampling_rate,
                                                  shoehorn_header.sampling_rate/100, 
                                                  0.98)

    # why does obspy imaging do flipud? line 183
    Sxx = np.flipud(Sxx)

    # Array crop to our needs
    # Sxx, f, t = spectrogram_crop(Sxx, f, t, lambda f: f<=30, lambda t: t<=350)
    Sxx, f, t = spectrogram_crop(Sxx, f, t, None, None)
    # account for cropped Sxx
    print(f'Sxx min: {np.min(Sxx)}, max: {np.max(Sxx)}')
    halfbin_time = (t[1] - t[0]) / 2.0
    halfbin_freq = (f[1] - f[0]) / 2.0
    spec_ax_extent = (t[0] - halfbin_time, t[-1] + halfbin_time,
                      f[0] - halfbin_freq, f[-1] + halfbin_freq)
    print(u'Δt=' + f'{halfbin_time}s '+u'Δf=' + f'{halfbin_freq}Hz')
    def lemma(x):
        return np.piecewise(x, [x<=0, x>0], [0, lambda x: np.exp(-1/x)])

    def my_sig(x):
        # print(f"my_sig called! [Debug] {x = }\n[Debug] {type(x) = }")
        result = lemma(x)/(lemma(x)+lemma(1-x))
        # print(f"[Debug] {result = }\n[Debug] {result.mask.any() = }")
        return result

    def c_forward(x):
        # print(f"c_forward with x = {x}")
        return my_sig((x-270)/120 + 0.5)+0.003*(x-1200)

    # def c_inv(x):
        # print(f"c_backward with x = {x}")
    c_inv = inversefunc(c_forward, domain=(-1200,1200))

    def trunc_cmap(cmap_name: str, minval: float = 0.0, maxval: float = 1.0, n: int = 255):
        """Cmap converts numbers between 0 and 1 to a color. Here,
        we take only a slice of matplotlib's cmap instead."""
        cm = plt.get_cmap(cmap_name)
        if minval == 0.0 and maxval == 1.0:
            return cm
        new_cmap = colors.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(n=cm.name, a=minval, b=maxval),
            cm(np.linspace(minval, maxval, n)))
        return new_cmap
    # my_norm = colors.FuncNorm((c_forward, c_inv), vmin=180, vmax=440)
    my_norm = None

    # Plot
    # 1. Take a relevant slice of the color bar so that the truncated
    # spectrogram has similar light streaks to what Miguel has done
    # 2. Specify that all values above 440 for vmax are to be mapped to the
    # upper end of cmap.
    # 3. aspect = auto to allow for set_xlim, set_ylim to not squash the ax
    fig = plt.figure()
    gs = gridspec.GridSpec(nrows=1, ncols=2, figure=fig,
        wspace=0.1,
        width_ratios=[1, .03])
    spec_ax = plt.subplot(gs[0, 0])
    colb_ax = plt.subplot(gs[0, 1])

    cmappable = spec_ax.imshow(Sxx, cmap=trunc_cmap('terrain', 0, 0.75),
                            norm=my_norm, interpolation="nearest",
                            vmax=100,
                            extent=spec_ax_extent, aspect='auto')
    spec_ax.grid(False)
    # add colorbar
    plt.colorbar(cmappable, cax=colb_ax, shrink=1.0, extend='max')
    spec_ax.tick_params(axis='both', which='major', labelsize=6)
    colb_ax.tick_params(axis='both', which='major', labelsize=6)
    
    def seconds_to_hms(seconds, _):
        dt = parsed1.header['Time start'] + timedelta(seconds=seconds)
        return dt.strftime("%H:%M")
    spec_ax.xaxis.set_major_formatter(FuncFormatter(seconds_to_hms))

    spec_ax.set_xlabel('Time (hh:mm)', fontsize=10)
    spec_ax.set_ylabel('Frequency (Hz)', fontsize=10)
    colb_ax.set_ylabel('Intensity (a.u.)', fontsize=10)
    spec_ax.set_xlim(ax_extent[0], ax_extent[1])
    spec_ax.set_ylim(ax_extent[2], 500)

    fig.set_size_inches(fig_width:=0.495*(8.3-2*0.6)*4, 0.7*fig_width)
    fig.tight_layout()
    fig.subplots_adjust(left=0.110, bottom=0.095, right=0.880, top=0.955)

    plt.show(block=True)
    # fig.savefig('traffic_fig_desired_1.png', dpi=300, format='png')
    # plt.close()

if __name__ == '__main__':
    main()
