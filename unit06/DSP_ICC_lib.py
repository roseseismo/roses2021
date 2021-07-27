
import numpy as np # For discretized compution and array manipulation (https://numpy.org/).
import obspy  # For processing seismological data (https://docs.obspy.org/).
from scipy import signal, fft  # For conducting signal processing (https://www.scipy.org/)
import plotly.graph_objects as go  # Plotly for interactive figure creation (https://plotly.com/python/).

"""
Description: Custom python package for this Jupyter Notebook demonstration (DSP: Introductory Crash Course).
Created: 7/24/2021
Author: David L. Guenaga (dlguenaga@utep.edu)
"""

def discrete_plot(x,y):
    """
    This function to create a plot illustrating discrete points.
    :x: array-like
    :y: array-like
    :return: class 'plotly.graph_objs._figure.Figure'
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Interpolated', line=dict(color='#6baed6', width=4)))
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Data Point',marker=dict(color='#08519c', size=6)))
    fig.update_xaxes(title_text='Time (s)')
    fig.update_yaxes(title_text='Amplitude')
    fig.update_layout(template="seaborn")
    return fig


def signal_demo(signal_name, sps = 100, t = [], n=1):
    """
    This function provides sample signals.
    :signal_name: string
    :sps: float, only used for sinusoidal signals
    :t: array-like, only used for sinusoidal signals
    :t: array-like, only used for Fourier series
    :return: array
    """
    if signal_name == 'sinewave_1a':
        y = 2 * np.sin(10 * t + 0.9)
    elif signal_name == 'cosinewave_1':
        y = 2 * np.cos(1.5 * t + 2)
    elif signal_name == 'cosinewave_2':
        y = 0.5 * np.cos(20 * t + 0.9)
    elif signal_name == 'stepfunction_1':
        y = np.concatenate([np.ones(125), np.zeros(125), np.ones(125), np.zeros(25)])
    elif signal_name == 'sawtooth_1':
        y = np.concatenate([np.zeros(100), 
            np.divide(np.add(signal.sawtooth(2 * np.pi * np.linspace(0, 1, 100)),1),2), 
            np.zeros(100)])
    elif signal_name == 'stepfunction_2':
        y = np.concatenate([np.zeros(25), 
            np.multiply(np.ones(50), -1), 
            np.multiply(np.ones(100), 2), 
            np.ones(100), np.zeros(25)])
    elif signal_name == 'fourier_series_sf':
        y = np.zeros(len(t))
        for n in np.arange(1, 2*n+1, 2):
            y = np.add(y, ((1.25/n) * np.sin(n * t)))

    return y


def fourier_demo():
    # number of signal points
    N = 500

    # sample spacing
    T = 1/1000

    x = np.linspace(0.0, N*T, N, endpoint=False)

    y = (10 * np.sin(30 * 2.0*np.pi*x) + 
         1 * np.sin(40 * 2.0*np.pi*x) +
         3 * np.sin(65 * 2.0*np.pi*x) +
         3 * np.sin(75 * 2.0*np.pi*x))

    plot_y = (10 * np.sin(30 * x) + 1 * np.sin(40 * x) + 3 * np.sin(65 * x) + 3 * np.sin(75 * x))
    yf = fft.fft(y)
    xf = fft.fftfreq(N, T)
    xf = fft.fftshift(xf)

    yplot = fft.fftshift(yf)


    fig = go.Figure()
    fig.add_trace(go.Scatter(y=plot_y, x=x % 360, mode='lines'))
    fig.update_layout(title='<b>Waveform (Time Domain)</b>')
    fig.update_xaxes(title_text='Time (s)')
    fig.update_yaxes(title_text='Amplitude')
    fig.show()

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=(1.0/N*np.abs(yplot))*2, x=xf, mode='lines'))
    fig.update_layout(title='<b> Power Spectrum (Frequency Domain)</b>', xaxis=dict(range=[0, N]))
    fig.update_xaxes(title_text='Frequency (Hz)')
    fig.update_yaxes(title_text='Amplitude')
    fig.show()


def plot_spectrogram(freq,time,power,start_timestamp,title_name='', colorscale='plasma',dB=True):
    """
    This function plots spectrogram from scipy.signal.spectrogram() results.
    :freq: array
    :time: array
    :power: array
    :start_timestamp: float
    :colorscale: string
    :dB: bool; Plot in decibels?
    """
    if dB:
        val = 10*np.log10(power)
    UTC_tt = [obspy.UTCDateTime(t) for t in np.add(time, start_timestamp)]  # Format timestamp into datetime
    # -------------------------
    # Plot Spectrogram
    trace = [go.Heatmap(x=UTC_tt,  # Datetime
                        y=freq,  # Frequency
                        z=val,  # Amplitude (usually converted into [10*log_10] dB for scale)
                        colorscale=colorscale,  # Colormap for spectrogram 
                       )]
    layout = go.Layout(yaxis=dict(title='Frequency (Hz)'),  # Create figure layout
                       xaxis=dict(title='Datetime'),
                       title='<b>Spectrogram: '+ title_name+'</b>')
    # Create figure
    fig = go.Figure(data=trace, layout=layout)
    fig.show()


def c_animation(signal1, signal2, dsp_process, fps=0.001, template='plotly'):
    """
    This function creates a animation convolving/correlating the two provided signals.
    :signal1: array-like
    :signal2: array-like
    :fps: float
    :template: string
    """
    sl1 = len(signal1)
    sl2 = len(signal2)
    signal_len = sl1 + sl2

    if dsp_process.lower() == 'correlation' or dsp_process.lower() == 'corr':
        ac = signal.correlate(signal1, signal2, mode='full')
        dsp_process = 'Correlation'
    elif dsp_process.lower() == 'convolution' or dsp_process.lower() == 'conv':
        ac = signal.convolve(signal1, signal2, mode='full')
        dsp_process = 'Convolution'
        signal2 = np.flip(signal2)
    else:
        #return "Error: Invalid DSP process. Must be 'correlation' or 'convolution'."
        pass

    gobal_absmax = np.nanmax([np.max(np.abs(signal1)), np.max(np.abs(signal2))])
    signal1 = np.divide(signal1, gobal_absmax)
    signal2 = np.divide(signal2, gobal_absmax)

    ac_n = []
    ac_max = np.nanmax(np.abs(ac))

    frames = []
    len_n = []

    for n in np.arange(signal_len):
        len_n.append(n - sl2)
        if n < len(signal2):
            ar = np.concatenate([np.ones(n+1),np.nan * np.ones(len(signal2)-(n+1))])
        ac_n.append(np.subtract(np.divide(ac[n-1],ac_max), 2.2))
        frames.append(go.Frame(data=[go.Scatter(y=signal1), 
                                     go.Scatter(y=signal2,x=np.add(np.arange(-1*sl2,0),n), mode='lines'),
                                     go.Scatter(y=[-2.2]*n,x=len_n),
                                     go.Scatter(y=ac_n, x=len_n, opacity=1, mode='lines')]
                              ))

    fig = go.Figure(
        data=[go.Scatter(y=signal1, mode='lines', fill='tozeroy', name="Signal A"),
             go.Scatter(y=signal2, x=np.arange(-1*sl2,0), mode='lines', fill='tozeroy', name="Signal B")],
        layout=go.Layout(
            xaxis=dict(range=[-1*sl2, signal_len], autorange=False),
            yaxis=dict(range=[-3.3, 1.1], autorange=False),
            title='<b>'+dsp_process+' Animation</b>',
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(label="Play", 
                              method="animate", 
                              args=[None, {"frame": {"duration": 0.001, "redraw": False}, 
                                           "fromcurrent": True, "transition": {"duration": 3000}}]), 
                         dict(label="Pause", 
                              method="animate", 
                              args=[[None],{"frame": {"duration": 0, "redraw": False}, 
                                            "mode": "immediate", "transition": {"duration": 0}}])
                        ])]
        ),
        frames=frames
    )
    fig.add_trace(go.Scatter(y=[0], mode='lines', showlegend=False))
    fig.add_trace(go.Scatter(y=[0], mode='lines', line=dict(color='purple'), fill='tonexty', name=dsp_process))
    fig.update_layout(yaxis_visible=False, xaxis_visible=False, template=template)
    fig.add_trace(go.Scatter(y=[0.9,-1.3], x=[-1* sl2 + 100, -1* sl2 + 150], showlegend=False, 
                             mode="text", text=['<b>Signal Process</b>', '<b>Resulting '+ dsp_process +'</b>']))
    fig.add_hline(y=0, line=dict(dash='dash'), opacity=0.5)
    fig.add_hline(y=-1.1)
    fig.add_hline(y=-2.2, line=dict(dash='dash'), opacity=0.5)
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = fps
    fig.show()
