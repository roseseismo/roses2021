"""
Array processing interface functions

.. module:: array_processing

:author:
    Jelle Assink (jelle.assink@knmi.nl)

    Acknowledgements:
    Stephen Arrowsmith (sarrowsmith@smu.edu)

:license:
    This code is distributed under the terms of the
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.en.html)
"""
from obspy import UTCDateTime, Stream
from obspy.clients.filesystem.sds import Client as sds_client
#from obspy.clients.fdsn import Client as fdsn_client
from obspy.clients.fdsn import RoutingClient
from obspy.signal.array_analysis import clibsignal, cosine_taper, get_geometry, get_timeshift
from obspy.signal.util import next_pow_2
from obspy.core import AttribDict

import os
import math
import numpy as np
import matplotlib.pyplot as plt


def get_data(data_source, inventory, starttime, endtime, margin_t=10):
    """
    """
    [(key, value)] = data_source.items()

    stream = Stream()
    for ms in stream:
        samp_rate = ms.stats.delta
        ms.trim(starttime, endtime-samp_rate)

    if key == 'local':
        cl = sds_client(value)
    elif key == 'fdsn':
        cl = RoutingClient(value)
        #cl = fdsn_client(value)
        
    for net in inventory:
        for sta in net:
            for cha in sta:
                try:
                    st = cl.get_waveforms(network=net.code,
                                          station=sta.code,
                                          location=cha.location_code,
                                          channel=cha.code,
                                          starttime=starttime-margin_t,
                                          endtime=endtime+margin_t)
                    stream += st
                except ValueError as e:
                    print(e)
                    pass

    for ms in stream:
        samp_rate = ms.stats.delta
        ms.trim(starttime, endtime-samp_rate)

    return stream

def stream2sds(stream, path_root, reclen=512, wave_format='mseed', **kwargs):
    """
    Write ObsPy Stream out to SDS data structure
    """
    elements = {}
    juldays = []
    for ms in stream:

        elements[ms.id] = ms.stats
        yj0  = ms.stats['starttime'].year * 1e3
        yj0 += ms.stats['starttime'].julday
        yj1  = ms.stats['endtime'].year * 1e3
        yj1 += ms.stats['endtime'].julday
        juldays.extend(range(int(yj0),int(yj1)+1))

    juldays = sorted(list(set(juldays)))

    for day in juldays:
        t0 = UTCDateTime(str(day))
        t1 = t0 + 24*3600

        st_day = stream.copy()

        for (ele,stats) in elements.items():
            st_ele = st_day.select(id=ele).trim(t0,t1-stats['delta'])

            for ms in st_ele:
                if ms.std() == 0:
                    if ms.stats.channel != 'lcq':
                        pass
                    if ms.stats.channel != 'vbc':
                        pass
                    else:
                        st_ele.remove(ms)

            if len(st_ele) > 0:
                out_file = '%s.D.%4d.%03d' % ( ele, t0.year, t0.julday )
            
                out_path = '%s/%s/%s/%s/%s.D' % (
                    path_root,
                    t0.year,
                    stats['network'],
                    stats['station'],
                    stats['channel']
                    )
                output = '%s/%s' % (out_path, out_file)

                if os.path.isdir(out_path) is False:
                    try:
                        os.makedirs(out_path)
                    except OSError as e:
                        print (e.args)

                print ('Writing %s file [ %s ]' % (
                    wave_format,
                    output
                    ), end=''),

                # Do not write out traces with 1 sample, or bad data
                # for such data, the standard deviation equals 0
                try:
                    st_ele.write(
                        output,
                        flush=True,
                        reclen=reclen,
                        format=wave_format
                        )
                    print (' -> OK!')
                except ValueError as e:
                    print (' -> no data, not written. [ %s ]' % e)
                    
            else:
                print (' -> No data for station %s on [%s - %s]' % (ele, t0, t1))
    return

def plotFK(st, startTime, endTime, frqlow, frqhigh,
           sll_x=-3.6, slm_x=3.6, sll_y=-3.6, slm_y=3.6, sl_s=0.18,
           beam='bartlett', prewhiten=0, coordsys='lonlat', verbose=False,
           plot=True, normalize=True, cmap='inferno_r', sl_circle=True, 
           interpolation=None, vmin=None, vmax=None, plot_normalize=False,
           sl_corr=[0.,0.]):
    '''
    Modified from Stephen Arrowsmith's ROSES 2020 class

    Computes and displays an FK plot for an ObsPy Stream object, st, given
    a start time and end time (as UTCDateTime objects) and a frequency band
    defined by frqlow and frqhigh. The slowness grid is defined as optional
    parameters (in s/km).

    This function implements code directly from ObsPy, which has been optimized,
    for simply plotting the FK spectrum

    It includes the option to normalize the data in the time window before running FK

    It also includes the option to apply a slowness correction, defined by sl_corr
    '''

    stream = st.copy()
    stream = stream.trim(startTime, endTime)
    nstat = len(stream)

    fk_methods = dict(bartlett=0, capon=1)
    
    if nstat > 0:
        if normalize:
            for ms in stream:
                ms.data = ms.data/np.max(np.abs(ms.data))

        grdpts_x = int(((slm_x - sll_x) / sl_s + 0.5) + 1)
        grdpts_y = int(((slm_y - sll_y) / sl_s + 0.5) + 1)

        geometry = get_geometry(stream, coordsys=coordsys, verbose=verbose)

        time_shift_table = get_timeshift(geometry, sll_x, sll_y,
                                        sl_s, grdpts_x, grdpts_y)
        
        fs = stream[0].stats.sampling_rate
        nsamp = stream[0].stats.npts

        # generate plan for rfftr
        nfft = next_pow_2(nsamp)
        deltaf = fs / float(nfft)
        nlow = int(frqlow / float(deltaf) + 0.5)
        nhigh = int(frqhigh / float(deltaf) + 0.5)
        nlow = max(1, nlow)  # avoid using the offset
        nhigh = min(nfft // 2 - 1, nhigh)  # avoid using nyquist
        nf = nhigh - nlow + 1  # include upper and lower frequency

        # to speed up the routine a bit we estimate all steering vectors in advance
        steer = np.empty((nf, grdpts_x, grdpts_y, nstat), dtype=np.complex128)
        clibsignal.calcSteer(nstat, grdpts_x, grdpts_y, nf, nlow,
                            deltaf, time_shift_table, steer)
        _r = np.empty((nf, nstat, nstat), dtype=np.complex128)
        ft = np.empty((nstat, nf), dtype=np.complex128)

        # 0.22 matches 0.2 of historical C bbfk.c
        tap = cosine_taper(nsamp, p=0.22)
        relpow_map = np.empty((grdpts_x, grdpts_y), dtype=np.float64)
        abspow_map = np.empty((grdpts_x, grdpts_y), dtype=np.float64)

        for i, tr in enumerate(stream):
            dat = tr.data
            dat = (dat - dat.mean()) * tap
            ft[i, :] = np.fft.rfft(dat, nfft)[nlow:nlow + nf]

        ft = np.ascontiguousarray(ft, np.complex128)
        relpow_map.fill(0.)
        abspow_map.fill(0.)

        # computing the covariances of the signal at different receivers
        dpow = 0.
        for i in range(nstat):
            for j in range(i, nstat):
                _r[:, i, j] = ft[i, :] * ft[j, :].conj()
                if i != j:
                    _r[:, j, i] = _r[:, i, j].conjugate()
                else:
                    dpow += np.abs(_r[:, i, j].sum())
        dpow *= nstat

        clibsignal.generalizedBeamformer(
            relpow_map, abspow_map, steer, _r, nstat, prewhiten,
            grdpts_x, grdpts_y, nf, dpow, fk_methods[beam])
        fisher_map = (nstat-1) * relpow_map / (1-relpow_map)

        (ix, iy) = np.unravel_index(relpow_map.argmax(), relpow_map.shape)

        # here we compute baz, slow
        slow_x = sll_x + ix * sl_s
        slow_y = sll_y + iy * sl_s

        # ---------
        slow_x = slow_x - sl_corr[0]
        slow_y = slow_y - sl_corr[1]
        #print(slow_x, slow_y)
        # ---------

        slow = np.sqrt(slow_x ** 2 + slow_y ** 2)
        if slow < 1e-8:
            slow = 1e-8
        azimut = 180 * math.atan2(slow_x, slow_y) / math.pi
        baz = azimut % -360 + 180

        if plot:
            n_frames = 3
            (fig, ax) = plt.subplots(1, n_frames, sharey=True, figsize=(8,3.5),
                                     constrained_layout=True)

            extent = extent=[sll_x, slm_x + sl_s, sll_y, slm_y + sl_s]

            # FK power
            i = 0
            H = np.flipud(np.fliplr(abspow_map.T))
            if plot_normalize:
                H = H / H.max()
            im = ax[i].imshow(H, extent=extent, origin='lower', aspect='auto',
                              cmap=cmap, interpolation=interpolation)
            plt.colorbar(im, ax=ax[i], orientation="horizontal", label='FK Power')

            # Semblance
            i += 1
            H = np.flipud(np.fliplr(relpow_map.T))
            if plot_normalize:
                H = H / H.max()
            im = ax[i].imshow(H, extent=extent, origin='lower', aspect='auto',
                              cmap=cmap, interpolation=interpolation)
            plt.colorbar(im, ax=ax[i], orientation="horizontal", label='Semblance')

            # Fisher ratio
            i += 1
            H = np.flipud(np.fliplr(fisher_map.T))
            if plot_normalize:
                H = H / H.max()
            im = ax[i].imshow(H, extent=extent, origin='lower', aspect='auto',
                              cmap=cmap, interpolation=interpolation)
            plt.colorbar(im, ax=ax[i], orientation="horizontal", label='Fisher ratio')

            for i in range(0, n_frames):
                if sl_circle:
                    angles = np.deg2rad(np.arange(0., 360, 1.))
                    slowness = dict(seismic_P=6.0, Rayleigh=3.0, infrasound=0.34)
                    for (key, radius) in slowness.items():
                        x_circle = np.sin(angles)/radius
                        y_circle = np.cos(angles)/radius
                        ax[i].plot(x_circle, y_circle, linestyle='solid', label=key, alpha=0.6)

                ax[i].plot(0, 0, 'k+')
                ax[i].plot(-slow_x, -slow_y, 'w+')
                ax[i].set_xlabel('x-slowness [s/km]')

            ax[0].set_ylabel('y-slowness [s/km]')

            baz_max = round(baz % 360., 2)
            appvel_max = round(1/slow, 2)
            title_str = (f'Peak semblance at {baz_max:.2f} deg. '
                         f'and {appvel_max:.2f} km/s '
                         f'between [ {frqlow:.2f} - {frqhigh:.2f} ] Hz')
            fig.suptitle(title_str)

            return fig, ax

        # # only flipping left-right, when using imshow to plot the matrix is takes 
        # # points top to bottom points are now starting at top-left in row major
        # return np.fliplr(relpow_map.T), baz % 360, 1. / slow

    else:
        print(f'No data present for timerange {startTime} - {endTime}')
        return