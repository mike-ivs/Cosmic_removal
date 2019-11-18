# Mike Laverick - cosmic ray removal tool - KU Leuven - Python 3
# Email: mikelaverick@btinternet.com
import numpy as np

# Cosmic ray removal of a 1D spectrum post-pipeline.
# Takes in original flux and returns corrected-flux and flagged flux indexes
# (tech notes available at https://github.com/mike-ivs/Cosmic_removal)
#
# Usage: cosmic_removal( flux_orig ) => returns (flux_corr, flags)


def sample_local(array, ii):
    """local sample taken around flux value"""
    return (array[ii-7:ii+9])


def sample_right(array, ii):
    """right sample taken around flux-gradient"""
    return (array[ii:ii+16])


def sample_left(array, ii):
    """left sample taken around flux-gradient"""
    return (array[ii-15:ii+1])


def Q_test(sample):
    """Dixon's Q-test to determine statistical outliers. Modified range to
    exclude outlier from statistics.
    """
    return (sample[15] - sample[14])/(sample[14] - sample[0])


def gradient_calc(array, steps):
    """Calculates difference between original and rolled array"""
    return (np.roll(array, steps) - array)


def Q_test_flagging(flux, grad, iter, sorted_sample, sorted_sample1, q_value,
                    threshold, adjust, flag_array):
    """Main checks to see if a flux value is considered an outlier due to its
    sharp gradient.

    Checks if q-value (from Q-test on left/right gradient samples) exceeds
    the given Q threshold. Sanity check that outlier is the iteration point.
    Sanity check that flux is one of largest in local sample. Sanity check
    that point is not edge of non-overlapping orderself.

    If criteria + sanity checks met: flag corresponding flux point
    """
    if q_value > threshold:
        if grad[iter] == np.max(sorted_sample):
            if flux[iter + adjust] > sorted_sample1[12]:
                if np.min(sorted_sample) != 0:
                    flag_array[iter + adjust] = flag_array[iter + adjust] + 1


def array_maths(array, array2):
    """Combines left and right flag arrays to produce final set of flags for a
    given gradient. Flags single values and neighbouring values (see tech note)
    """
    return ((array*array2) + (np.roll(array, 1))*(array2) +
            (np.roll(array2, -1))*(array))


def cosmic_removal(masterflux):
    """Main routine.

    Iterates over 4 gradients of flux (see tech note) and calculates flagged
    points. Checks that point is flagged in multiple gradients and then
    corrects the flagged points using a linear interpolation of neighbouring
    values.

    Usage: cosmic_removal(flux_orig)
    returns: (flux_corr, flags)
    """
    flux = np.array(masterflux)
    reverse_flux = np.flipud(flux)

    gradlist = [gradient_calc(flux, -1), gradient_calc(flux, -2),
                np.flipud(gradient_calc(reverse_flux, -1)),
                np.flipud(gradient_calc(reverse_flux, -2))]
    flaglistL = [np.zeros(len(flux)) for x in range(0, 4)]
    flaglistR = [np.zeros(len(flux)) for x in range(0, 4)]
    adjustor = [1, 2, -1, -2]

    for i in range(flux.shape[0]):

        local_samp = sorted(sample_local(flux, i))

        for idx, item in enumerate(gradlist):
            try:

                samples_left = sorted(sample_left(item, i))
                samples_right = sorted(sample_right(item, i))

                Q_test_flagging(flux, item, i, samples_left, local_samp,
                                Q_test(samples_left), 0.4, adjustor[idx],
                                flaglistL[idx])
                Q_test_flagging(flux, item, i, samples_right, local_samp,
                                Q_test(samples_right), 0.4, adjustor[idx],
                                flaglistR[idx])

            except(IndexError):
                pass

    cosmic_flags = ((array_maths(flaglistL[0],
                     flaglistR[0]))*(array_maths(flaglistL[1], flaglistR[1])) +
                    (array_maths(flaglistL[2],
                     flaglistR[2]))*(array_maths(flaglistL[3], flaglistR[3]))
                    ).clip(max=1)

    """Blunt removal of cosmic/glitches, interpolates between the points
    flux[i-2] and flux[i+2] to correct flagged points and 'affected' region
    """
    for ii in range(cosmic_flags.shape[0]):
        if cosmic_flags[ii] == 1:
            if cosmic_flags[ii+1] == 1:
                flux[ii-1] = flux[ii-2]
                flux[ii] = flux[ii-1]
            else:
                flux[ii-1] = 0.75*flux[ii-2] + 0.25*flux[ii+2]
                flux[ii+1] = 0.75*flux[ii+2] + 0.25*flux[ii-2]
                flux[ii] = (flux[ii+2]+flux[ii-2])/2

    return (flux, cosmic_flags)
