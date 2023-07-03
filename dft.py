from dataclasses import dataclass
from math import pi, cos, sin, sqrt, atan


@dataclass
class ComplexNumber:
    real_part: float
    imag_part: float
    freq: float
    amp: float
    phase: float


def dft(signal):
    fourier = []

    N = len(signal)

    # loop through every value (amplitude) in the signal
    for k in range(1, N):
        real_part = 0.0
        imag_part = 0.0
        # for every value, loop through the signal again, summing up the amplitudes
        for n in range(1, N):
            omega = 2 * pi * k * n / N
            # probe the signal with cosine
            real_part += signal[n] * cos(omega)
            # probe the signal with sine
            imag_part -= signal[n] * sin(omega)

        # normalize the parts
        real_part /= N
        imag_part /= N

        freq = k
        amp = sqrt(real_part ** 2 + imag_part ** 2)
        phase = atan(imag_part / real_part)
        
        fourier.append(ComplexNumber(real_part, imag_part, freq, amp, phase))

    return fourier
