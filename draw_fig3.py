import numpy as np
import matplotlib.pyplot as plt
import os
import math
from matplotlib import rcParams
from scipy.interpolate import interp1d
from scipy.interpolate import make_interp_spline, BSpline
from scipy.ndimage.filters import gaussian_filter1d


if __name__ == "__main__":

    jpeg2000_bpp = np.array([0.1617, 0.2,0.3,0.4,0.5,0.6])
    jpeg2000_ssim = np.array([10,10.6,11.8,12.8,13.7,14.5])
    jpeg2000_f = interp1d(jpeg2000_bpp, jpeg2000_ssim, kind='cubic')
    jpeg2000_bpp_c = np.linspace(jpeg2000_bpp.min(),jpeg2000_bpp.max(),300)
    spl = make_interp_spline(jpeg2000_bpp, jpeg2000_ssim, k=3)

    webp_bpp = np.array([0.155, 0.2128, 0.2979, 0.3830, 0.4468, 0.5000, 0.5957])
    webp_ssim = np.array([10.057, 11.0304, 12.2185, 13.3582, 13.9794, 14.5113, 15.3511])
    webp_f = interp1d(webp_bpp, webp_ssim, kind='cubic')
    webp_bpp_c = np.linspace(webp_bpp.min(),webp_bpp.max(),10)
    webp = make_interp_spline(webp_bpp, webp_ssim, k=1)

    bpg_bpp = np.array([0.11,0.2,0.3,0.4,0.5,0.6])
    bpg_ssim = np.array([10,12,13.6,14.7,15.7,16.6])
    bpg_f = interp1d(bpg_bpp, bpg_ssim, kind='cubic')
    bpg_bpp_c = np.linspace(bpg_bpp.min(),bpg_bpp.max(),10)

    johnston_bpp = np.array([0.1164, 0.1702, 0.2043, 0.3191, 0.5000, 0.5957])
    johnston_ssim = np.array([10.000, 11.3494, 12.1185, 13.8994, 16.0206, 16.8924])
    johnston_f = interp1d(johnston_bpp, johnston_ssim, kind='cubic')
    johnston_bpp_c = np.linspace(johnston_bpp.min(),johnston_bpp.max(),300)
    johnston = make_interp_spline(johnston_bpp, johnston_ssim, k=1)

    li_bpp = np.array([0.115, 0.17, 0.2,0.3,0.4,0.5,0.6])
    li_ssim = np.array([11.4267, 12.5181, 13.1103, 14.8149, 16.0206, 17.2125, 17.9352])
    li_f = interp1d(li_bpp, li_ssim, kind='cubic')
    li_bpp_c = np.linspace(li_bpp.min(),li_bpp.max(),300)
    li = make_interp_spline(li_bpp, li_ssim, k=1)

    mentzer_bpp = np.array([0.1265306, 0.1530612, 0.1795918, 0.2061224, 0.2326531, 0.2591837, 0.2857143, 0.3122449, 0.3387755, 0.3653061, 0.3918367, 0.4183673, 0.444898, 0.4714286, 0.4979592, 0.5244898, 0.5510204, 0.577551, 0.6040816])
    mentzer_ssim = np.array([11.48347906, 12.34669775, 12.99230538, 13.50357544, 13.96394502, 14.39644955, 14.79730064, 15.16675213, 15.50438298, 15.817186, 16.11380229, 16.39588369, 16.6679866, 16.93043421, 17.18405369, 17.42933708, 17.66686539, 17.89740353, 18.12132546])
    mentzer_f = interp1d(mentzer_bpp, mentzer_ssim, kind='cubic')
    mentzer_bpp_c = np.linspace(mentzer_bpp.min(),mentzer_bpp.max(),300)

    rippel_bpp = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    rippel_ssim = np.array([11.1, 13.4, 15.08,16.44,17.4,18.15])
    rippel_f = interp1d(rippel_bpp, rippel_ssim, kind='cubic')
    rippel_bpp_c = np.linspace(rippel_bpp.min(),rippel_bpp.max(),300)

    our_bpp = np.array([0.129, 0.258, 0.382, 0.51, 0.653])
    our_ssim = np.array([12.07608311, 14.68521083, 16.19788758, 17.64551079, 18.23871964])
    our_f = interp1d(our_bpp, our_ssim, kind='cubic')
    our_bpp_c = np.linspace(our_bpp.min(),0.6,300)

    rcParams['font.family'] = 'Arial'
    rcParams['font.size'] = 28
    rcParams['font.weight'] = 'light'
    rcParams['figure.figsize'] = 7, 6
    rcParams['grid.linestyle'] = 'dashed'
    lw = 3

    plt.figure(figsize=(20, 10))

    plt.plot(jpeg2000_bpp_c, jpeg2000_f(jpeg2000_bpp_c), '-', color='black', label="JPEG2000", linewidth=lw)
    plt.plot(webp_bpp_c, webp(webp_bpp_c), '-', color='m', label="WebP", linewidth=lw)
    plt.plot(bpg_bpp_c, bpg_f(bpg_bpp_c), '-', color='orange', label="BPG", linewidth=lw)
    plt.plot(johnston_bpp_c, johnston(johnston_bpp_c), '-', color='c', label="Johnston", linewidth=lw)
    plt.plot(li_bpp_c, li_f(li_bpp_c), '-', color='y', label="Li", linewidth=lw)
    plt.plot(mentzer_bpp_c, mentzer_f(mentzer_bpp_c), '-', color='g', label="Mentzer", linewidth=lw)
    plt.plot(rippel_bpp_c, rippel_f(rippel_bpp_c), '-', color='b', label="Rippel", linewidth=lw)
    plt.plot(our_bpp_c, our_f(our_bpp_c), '-', color='red', label="Proposed", linewidth=lw)

    plt.grid(axis="x")
    plt.grid(axis="y")
    plt.xlabel("bits per pixel (BPP)")
    plt.ylabel("RGB MS-SSIM (dB)")
    plt.legend()
    plt.tight_layout()
    plt.xlim((0.05,0.65))
    # plt.show()
    plt.savefig('fig3.jpg', dpi=300)
