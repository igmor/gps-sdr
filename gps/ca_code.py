"""GPS L1 C/A (Coarse/Acquisition) code generator.

Each satellite (PRN 1..32) broadcasts a unique 1023-chip Gold code at
1.023 Mchip/s, so the code repeats every 1 ms. A Gold code is the XOR of
two maximal-length LFSR sequences (IS-GPS-200, section 3.3.2.3):

    G1 = 1 + x^3 + x^10
    G2 = 1 + x^2 + x^3 + x^6 + x^8 + x^9 + x^10

Both registers start as all ones. The satellite-specific part is which two
G2 stages are XORed together (the "phase selector"), which is equivalent
to delaying G2 by a PRN-specific number of chips.
"""
import numpy as np

CHIP_RATE = 1.023e6
CODE_LENGTH = 1023
L1_FREQ = 1575.42e6

# PRN -> (tap a, tap b) on the G2 register, 1-indexed (IS-GPS-200 Table 3-Ia).
G2_TAPS = {
    1: (2, 6), 2: (3, 7), 3: (4, 8), 4: (5, 9), 5: (1, 9), 6: (2, 10),
    7: (1, 8), 8: (2, 9), 9: (3, 10), 10: (2, 3), 11: (3, 4), 12: (5, 6),
    13: (6, 7), 14: (7, 8), 15: (8, 9), 16: (9, 10), 17: (1, 4), 18: (2, 5),
    19: (3, 6), 20: (4, 7), 21: (5, 8), 22: (6, 9), 23: (1, 3), 24: (4, 6),
    25: (5, 7), 26: (6, 8), 27: (7, 9), 28: (8, 10), 29: (1, 6), 30: (2, 7),
    31: (3, 8), 32: (4, 9),
}


def ca_code(prn):
    """Return the 1023-chip C/A code for `prn` as an int8 array of +1/-1.

    Mapping: logic 0 -> +1, logic 1 -> -1 (so XOR becomes multiplication).
    """
    a, b = G2_TAPS[prn]
    g1 = [1] * 10
    g2 = [1] * 10
    chips = np.empty(CODE_LENGTH, dtype=np.int8)
    for i in range(CODE_LENGTH):
        g2i = g2[a - 1] ^ g2[b - 1]
        chips[i] = g1[9] ^ g2i
        fb1 = g1[2] ^ g1[9]
        fb2 = g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9]
        g1 = [fb1] + g1[:9]
        g2 = [fb2] + g2[:9]
    return (1 - 2 * chips).astype(np.int8)


def sampled_code(prn, fs, n_samples, code_phase_chips=0.0, chip_rate=CHIP_RATE):
    """C/A code sampled at `fs` for `n_samples`, starting at `code_phase_chips`.

    `chip_rate` can be adjusted for code Doppler (the code is stretched or
    compressed slightly by the satellite's relative motion).
    """
    code = ca_code(prn)
    idx = np.floor(code_phase_chips + np.arange(n_samples) * chip_rate / fs).astype(np.int64)
    return code[idx % CODE_LENGTH]
