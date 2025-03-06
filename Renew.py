import numpy as np

class data_renew:

    def __init__(self, n_windows, samples_per_window, dtype = np.float32, start_value = 0):
        self.n_windows = n_windows
        self.samples_per_window = samples_per_window
        self.data = start_value * np.ones((self.n_windows, self.samples_per_window), dtype = dtype)

        self.total_samples = self.n_windows * self.samples_per_window

        self.elements_in_buffer = 0
        self.overwrite_index = 0

        self.indices = np.arange(self.n_windows, dtype=np.int32)
        self.last_window_id = np.max(self.indices)
        self.index_order = np.argsort(self.indices)

    def append_data(self, data_window):
        self.data[self.overwrite_index, :] = data_window

        self.last_window_id += 1
        self.indices[self.overwrite_index] = self.last_window_id
        self.index_order = np.argsort(self.indices)

        self.overwrite_index += 1
        self.overwrite_index = self.overwrite_index % self.n_windows

        self.elements_in_buffer += 1
        self.elements_in_buffer = min(self.n_windows, self.elements_in_buffer)

    def get_new_data(self, window_size):
        ordered_dataframe = self.data[self.index_order]
        ordered_dataframe = np.hstack(ordered_dataframe)    #Stack arrays in sequence horizontally
        return ordered_dataframe[self.total_samples - window_size:]