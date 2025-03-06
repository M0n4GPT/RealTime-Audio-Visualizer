import numpy as np
import time, math
import pyaudio

from Renew import *
from Visual import Visualizer

class Realtime_Processing:

    def __init__(self,
        rate   = None,
        FFT_window_size_ms  = 50,
        updates_per_second  = 100,        
        n_frequency_groups   = 400,
        GUI_height    = 450,
        GUI_ratio = 2/1):

        self.n_frequency_groups = n_frequency_groups
        self.rate = rate
        self.GUI_height = GUI_height
        self.GUI_ratio = GUI_ratio

        #Read Stream
        self.PA = pyaudio.PyAudio()

        self.data_buffer = None
        if self.rate is None:
            self.rate = 44100
        self.update_n_frames = math.ceil(self.rate / updates_per_second) # Round a number upward to its nearest integer

        self.stream = self.PA.open(
            format = pyaudio.paInt16,
            channels = 1,
            rate = self.rate,
            input=True,
            frames_per_buffer = self.update_n_frames,    # reduce latency
            stream_callback=self.callback   # callback mode 
            )

        self.FFT_window_size = int(math.ceil((self.rate * FFT_window_size_ms / 1000) / 2.) * 2)# Round a number upward to its nearest odd
        self.fft  = np.ones(int(self.FFT_window_size/2), dtype=float)
        self.fft_log = np.arange(int(self.FFT_window_size/2), dtype=float) * self.rate / self.FFT_window_size

        self.data_to_buffer = math.ceil(self.FFT_window_size / self.update_n_frames)
        self.data_to_buffer = max(1,self.data_to_buffer)

        # Compute frequency bin indices using logarithmic spacing and normalization
        self.fft_log_index = np.minimum(
            np.arange(len(self.fft_log)),
            np.round(
                (-np.logspace(np.log2(len(self.fft_log)), 0, num=len(self.fft_log), base=2) + len(self.fft_log)) 
                / (len(self.fft_log) / self.n_frequency_groups)
            ).astype(int)
        )

        self.group_energies = np.zeros(self.n_frequency_groups)
        self.group_centres_f  = np.zeros(self.n_frequency_groups)
        self.g_frequency_index_log   = []
        for group_index in range(self.n_frequency_groups):
            g_frequency_index = np.where(self.fft_log_index == group_index)
            self.g_frequency_index_log.append(g_frequency_index)
            fft_log_frequencies_this_bin = self.fft_log[g_frequency_index]
            self.group_centres_f[group_index] = np.mean(fft_log_frequencies_this_bin)



        self.fft_fps = 30
        # reduce pink noise
        self.power_normalization = np.logspace(0, np.log2(self.rate / 2), len(self.fft_log), base=2)


        #Let's get started:
        self.data_buffer = data_renew(self.data_to_buffer, self.update_n_frames)


        self.visualizer = Visualizer(self)
        self.visualizer.start()


    def Realtime_FFT(self):

        latest_data = self.data_buffer.get_new_data(self.FFT_window_size)

        self.fft= np.abs(np.fft.rfft(latest_data)[1:])
        self.fft = self.fft * self.power_normalization

        for group_index in range(self.n_frequency_groups):
            self.group_energies[group_index] = np.mean(self.fft[self.g_frequency_index_log[group_index]])

        self.group_energies = np.nan_to_num(self.group_energies, copy=True) # nan->0
        self.group_energies[self.group_energies < 0] = 0

        if self.visualizer._is_running:
            self.visualizer.update()

        return self.group_energies

    def callback(self, in_data, frame_count, time_info, status):
        if self.data_buffer is not None:
            self.data_buffer.append_data(np.frombuffer(in_data, dtype=np.int16))  
        return in_data, pyaudio.paContinue