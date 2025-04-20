import numpy as np

# Utility function for exponential rewards
def exp_dist(x):
    return np.exp(x) - 1

def unit(x):
    if np.linalg.norm(x) == 0:
        return np.zeros_like(x)
    else:
        return x / np.linalg.norm(x)


class OnlineFrequencyAmplitudeEstimation:
    def __init__(self, n_channels, dt=0.01, ema_alpha=0.95, min_freq=None, window_size=None):
        """
        Parameters:
            n_channels (int): Number of parallel channels.
            dt (float): Time step (seconds per sample).
            ema_alpha (float): EMA smoothing factor (closer to 1 means more smoothing).
            min_freq (float, optional): If provided, window_size is set to cover 2 cycles at this frequency.
            window_size (int, optional): Override the window size. Ignored if max_freq is provided.
        """
        self.n_channels = n_channels
        self.dt = dt
        self.ema_alpha = ema_alpha

        # Determine window size: if min_freq is provided, compute it based on 2 cycles.
        if min_freq is not None:
            # Window size = number of samples covering two periods of min_freq.
            self.window_size = int(np.ceil(2 / (min_freq * dt)))
        elif window_size is not None:
            self.window_size = window_size
        else:
            # Default window size if neither is provided.
            self.window_size = 64

        # Circular buffer for binary crossing events for each channel.
        self.crossings_buffer = np.zeros((self.window_size, n_channels), dtype=int)
        # Circular buffer for raw signal samples (for amplitude estimation).
        self.signal_buffer = np.zeros((self.window_size, n_channels))

        self.buffer_index = 0
        self.crossings_count = np.zeros(n_channels, dtype=int)  # Running sum for each channel
        # Count of samples processed so far (up to window_size).
        self.sample_count = 0

        # Previous sample and its derivative sign for each channel.
        self.prev_sample = None  # Shape: (n_channels,)
        self.prev_deriv_sign = None  # Shape: (n_channels,)

        # EMA frequency estimate (Hz) for each channel.
        self.f_est = np.zeros(n_channels, dtype=float)
        # EMA amplitude estimate for each channel.
        self.a_est = np.zeros(n_channels, dtype=float)

    def update(self, new_samples):
        """
        Update the frequency and amplitude estimation for all channels with a new sample vector.

        Parameters:
            new_samples (np.array): A 1D array of new samples with shape (n_channels,).

        Returns:
            tuple: (f_est, a_est) where:
                - f_est (np.array): EMA-smoothed frequency estimate (Hz) for each channel.
                - a_est (np.array): EMA-smoothed amplitude estimate for each channel.
        """
        new_samples = np.asarray(new_samples)

        # For the very first call, initialize previous sample and store the sample.
        if self.prev_sample is None:
            self.prev_sample = new_samples.copy()
            self.signal_buffer[self.buffer_index, :] = new_samples
            self.sample_count = 1
            self.buffer_index = (self.buffer_index + 1) % self.window_size
            return self.f_est.copy(), self.a_est.copy()

        # Compute the approximate derivative for each channel.
        diff = new_samples - self.prev_sample
        current_sign = np.sign(diff)
        # If the derivative is exactly zero, retain the previous sign.
        if self.prev_deriv_sign is not None:
            zeros = (current_sign == 0)
            current_sign[zeros] = self.prev_deriv_sign[zeros]

        # Determine crossing events: a crossing is detected when the sign changes.
        if self.prev_deriv_sign is not None:
            crossing = (current_sign != self.prev_deriv_sign).astype(int)
        else:
            crossing = np.zeros(self.n_channels, dtype=int)

        # Update sample count (up to window_size).
        if self.sample_count < self.window_size:
            self.sample_count += 1

        # Update the circular buffers for both crossings and raw samples.
        # Remove the oldest crossing count and add the new one.
        self.crossings_count -= self.crossings_buffer[self.buffer_index, :]
        self.crossings_buffer[self.buffer_index, :] = crossing
        self.crossings_count += crossing

        # Store the new samples in the signal buffer.
        self.signal_buffer[self.buffer_index, :] = new_samples

        # Advance the circular buffer index.
        self.buffer_index = (self.buffer_index + 1) % self.window_size

        # Update previous sample and derivative sign.
        self.prev_sample = new_samples.copy()
        self.prev_deriv_sign = current_sign.copy()

        # Calculate effective window duration in seconds.
        effective_duration = self.sample_count * self.dt

        # Frequency estimation:
        # For a sinusoid, two derivative crossings correspond to one cycle.
        cycles_in_window = self.crossings_count / 2.0
        f_current = cycles_in_window / effective_duration

        # EMA smoothing for frequency.
        self.f_est = self.ema_alpha * self.f_est + (1 - self.ema_alpha) * f_current

        # Amplitude estimation:
        # Use the portion of the signal buffer that is filled.
        if self.sample_count < self.window_size:
            current_window = self.signal_buffer[:self.sample_count, :]
        else:
            current_window = self.signal_buffer
        # Compute amplitude as difference between max and min for each channel.
        amplitude_current = np.max(current_window, axis=0) - np.min(current_window, axis=0)

        # EMA smoothing for amplitude.
        self.a_est = self.ema_alpha * self.a_est + (1 - self.ema_alpha) * amplitude_current

        return self.f_est.copy(), self.a_est.copy()
    
    def reset(self):
        """
        Reset the internal state of the estimator.
        """
        self.crossings_buffer.fill(0)
        self.signal_buffer.fill(0)
        self.buffer_index = 0
        self.crossings_count.fill(0)
        self.sample_count = 0
        self.prev_sample = None
        self.prev_deriv_sign = None
        self.f_est.fill(0)
        self.a_est.fill(0)

    def get_estimates(self):
        """
        Get the current frequency and amplitude estimates.

        Returns:
            tuple: (f_est, a_est) where:
                - f_est (np.array): EMA-smoothed frequency estimate (Hz) for each channel.
                - a_est (np.array): EMA-smoothed amplitude estimate for each channel.
        """
        return self.f_est.copy(), self.a_est.copy()