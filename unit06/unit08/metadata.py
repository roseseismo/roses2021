"""
Metadata interface functions

.. module:: metadata

:author:
    Jelle Assink (jelle.assink@knmi.nl)

:copyright:
    2021, Jelle Assink

:license:
    This code is distributed under the terms of the
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.en.html)
"""
from obspy import read_inventory
from obspy.clients.filesystem.sds import Client as sds_client
#from obspy.clients.fdsn import Client as fdsn_client
from obspy.clients.fdsn import RoutingClient
from obspy.core.inventory.inventory import Inventory
from obspy.signal.util import util_geo_km

from pandas import DataFrame
import os

class param_object(object):
    pass

def compute_offsets(cha, ref):
    (x, y) = util_geo_km(ref.longitude, ref.latitude,
                         cha.longitude, cha.latitude)
    return (x,y)

def df_to_ascii(df, fid_out, print_offset=False, verbose=False):
    """
    Write out stationtable as formatted ASCII table
    """
    if verbose:
        print ('Writing to %s' % fid_out)

    with open(fid_out, 'w') as f:   
        for _, r in df.iterrows():
            seed_id = f'{r.network}.{r.station}.{r.location}.{r.channel}'
            line  = f'{seed_id:>18s} '

            if print_offset:
                line += f'{r.deast:17.4f} {r.dnorth:17.4f} '
            line += f'{r.elevation:17.4f} {r.edepth:17.4f} '
            line += f'{r.latitude:9.6f} {r.longitude:9.6f} '
            line += f'{r.starttime:8d} {r.endtime:8d} '
            line += f'{r.gain:15.4e}\n'
            f.write(line)
    return

def get_array_reference(inv, reference='first'): 
    ref = param_object()

    if reference == 'first':
        ref.latitude = inv[0][0][0].latitude
        ref.longitude = inv[0][0][0].longitude

    elif reference == 'center':
        data = []
        for seed_id in inv.get_contents()['channels']:
            data.append(inv.get_coordinates(seed_id))
        data = DataFrame(data).drop_duplicates()
        ref.latitude = data['latitude'].mean()
        ref.longitude = data['longitude'].mean()
    return ref

def get_gain(inv, tr, starttime=None, endtime=None):
    invs = inv.select(network=tr.stats.network,
                        station=tr.stats.station,
                        location=tr.stats.location,
                        channel=tr.stats.channel,
                        starttime=starttime,
                        endtime=endtime)
    return invs[0][0][0].response.instrument_sensitivity.value

def get_metadata(data_source, network, station, location, channel,
                 starttime, endtime):
    invselect = locals()
    del(invselect['data_source'])
    [(key, value)] = data_source.items()

    if key == 'local':
        inv = read_inventory(value, 'STATIONXML')
        inv = inv.select(**invselect)

    elif key == 'fdsn':
        cl = RoutingClient(value)
        #cl = fdsn_client(value)
        inv = cl.get_stations(**invselect, level='response')

    else:
        print ('meta-data source should be "local" or "fdsn"')
        inv = []
    return inv

def inv_to_df(inv, compute_offset=False, offset_reference='first'):
    data = []

    for net in inv:
        for sta in net:
            for cha in sta:
                t0 = parse_utcdatetime(cha.start_date)
                t1 = parse_utcdatetime(cha.end_date)
                gain = cha.response.instrument_sensitivity.value
                item = dict(network=net.code,
                            station=sta.code,
                            location=cha.location_code,
                            channel=cha.code,
                            latitude=cha.latitude,
                            longitude=cha.longitude,
                            elevation=cha.elevation,
                            edepth=cha.depth,
                            starttime=t0,
                            endtime=t1,
                            gain=gain)

                if compute_offset:
                    ref = get_array_reference(inv, offset_reference)
                    (x, y) = compute_offsets(cha, ref)
                    item['deast'] = x*1e3
                    item['dnorth'] = y*1e3

                data.append(item)
    df = DataFrame(data).drop_duplicates()
    return df

def parse_utcdatetime(time):
    """
    Convenience function to format UTCDateTime in year-julday format
    """
    try:
        val = int(time.strftime('%Y%j'))
    except:
        val = -1
    return val

def select_inventory(stream, inv, starttime=None, endtime=None):
    """
    Convenience function to select inventory from active traces only
    """
    invs = Inventory()
    for tr in stream:
        invs += inv.select(network=tr.stats.network,
                           station=tr.stats.station,
                           location=tr.stats.location,
                           channel=tr.stats.channel,
                           starttime=starttime,
                           endtime=endtime)
    return invs
