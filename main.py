from Audio import Realtime_Processing
import time

fps = 60 
last_update = time.time()

display = Realtime_Processing(
                rate   = None,               # sample rate, default 44100
                FFT_window_size_ms  = 60,  
                updates_per_second  = 1000,  
                n_frequency_groups = 80, # number of frequency groups
                GUI_height    = 650,     # GUI window height
                GUI_ratio = 24/12  # GUI window size ration     
                )
  

while True:
    if (time.time() - last_update) > (1./fps):
        last_update = time.time()
        fft_energy = display.Realtime_FFT()   #  Real-time data processing (see 'Audio' file)



