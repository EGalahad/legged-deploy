import numpy as np
import scipy.signal as signal


class SecondOrderLowPassFilter:
    def __init__(self, cutoff: float, fs: float):
        """ Initialize the low-pass filter with cutoff frequency and sample rate. """
        self.fs = fs
        self.cutoff = cutoff
        
        # Compute the filter coefficients (second-order, low-pass)
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        self.b, self.a = signal.butter(2, normal_cutoff, btype='lowpass', analog=False)
        
        # Initialize previous inputs and outputs (n-1, n-2 terms)
        self.x1 = 0.0  # x[n-1]
        self.x2 = 0.0  # x[n-2]
        self.y1 = 0.0  # y[n-1]
        self.y2 = 0.0  # y[n-2]

    def update(self, x: np.ndarray) -> np.ndarray:
        """ Update the filter with a new input sample `x`, and return the filtered output. """
        # Apply the difference equation to get the new output value y[n]
        y = self.b[0] * x + self.b[1] * self.x1 + self.b[2] * self.x2 - self.a[1] * self.y1 - self.a[2] * self.y2
        
        # Update the state (shift the previous inputs and outputs)
        self.x2 = self.x1
        self.x1 = x
        self.y2 = self.y1
        self.y1 = y
        return y

