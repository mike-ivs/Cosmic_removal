# Mike Laverick - cosmic ray removal tool - KU Leuven - mikelaverick@btinternet.com
#
#--------------------------------------------------------------------------------------------------
import numpy as np
#--------------------------------------------------------------------------------------------------

#
# Cosmic ray removal of a 1D spectrum post-pipeline.    cosmic_removal( flux_orig ) => returns (flux_corr, flags)
# Takes in original flux and returns corrected-flux and cosmic-flagged flux values
#(tech notes on method are available: mike.laverick@ster.kuleuven.be)
#

# local sample taken around flux value, left and right sample taken around flux-gradient
def sample_local(array, ii):
  return (array[ii-7:ii+9])


def sample_right(array, ii):
  return (array[ii:ii+16])


def sample_left(array, ii):
  return (array[ii-15:ii+1])


# Dixon's Q-test to determine statistical outliers. Modified range to exclude outlier from statistics
def Q_test(sample):
  return (sample[15] - sample[14])/(sample[14] - sample[0])


def gradient_calc(array,steps):
  return ( np.roll(array,steps) - array )


# Main checks to see if a flux value is considered an outlier due to its sharp gradient
#
# checks if q-value (from Q-test on left/right gradient samples) exceeds the given Q threshold, sanity check that outlier is the iteration point,
# sanity check that flux is one of largest in local sample, sanity check that point is not edge of non-overlapping order:  flag corresponding flux point
def Q_test_flagging(flux,grad,iteration,sorted_sample,sorted_sample1,q_value,threshold,adjustment,flag_array):
  if q_value > threshold:
    if grad[iteration] == np.max(sorted_sample):
      if flux[iteration + adjustment] > sorted_sample1[12]:
        if np.min(sorted_sample) != 0:
          flag_array[iteration + adjustment] =  flag_array[iteration + adjustment] +1


# Combines left and right flag arrays to produce final set of flags for a given gradient. Flags single values and neighbouring values (see tech notes)
def array_maths(array,array2):
  return ((array*array2) + (np.roll(array,1))*(array2) + (np.roll(array2,-1))*(array))


#--------------------------------------------------------------------------------------------------


# Main routine. iterates over 4 gradients of flux (see tech notes) calculates flagged points, checks that point is flagged in multiple gradients
# and then corrects the flagged points using a linear interpolation of neighbouring values
def cosmic_removal(masterflux):

  flux = np.array(masterflux)
  reverse_flux = np.flipud(flux)

  gradlist = [gradient_calc(flux,-1),gradient_calc(flux,-2), np.flipud( gradient_calc(reverse_flux,-1) ), np.flipud( gradient_calc(reverse_flux,-2) )]
  flaglistL = [np.zeros(len(flux)) for x in xrange(0,4)]
  flaglistR = [np.zeros(len(flux)) for x in xrange(0,4)]
  adjustor = [1,2,-1,-2]

  for i in xrange(flux.shape[0]):

    local_samp = sorted(sample_local(flux,i))

    for idx, item in enumerate(gradlist):
      try:

        samples_left = sorted(sample_left(item,i))
        samples_right = sorted(sample_right(item,i))

        Q_test_flagging(flux,item,i,samples_left,local_samp,Q_test(samples_left),0.4,adjustor[idx],flaglistL[idx])
        Q_test_flagging(flux,item,i,samples_right,local_samp,Q_test(samples_right),0.4,adjustor[idx],flaglistR[idx])

      except(IndexError):
        pass

  cosmic_flags = (  (array_maths(flaglistL[0],flaglistR[0]))*(array_maths(flaglistL[1],flaglistR[1])) +
                    (array_maths(flaglistL[2],flaglistR[2]))*(array_maths(flaglistL[3],flaglistR[3]))  ).clip(max=1)

  #Blunt removal of cosmic/glitches, interpolates between the points flux[i-2] and flux[i+2] to correct flagged points and 'affected' region
  for ii in range(cosmic_flags.shape[0]):
    if cosmic_flags[ii] == 1:
      if cosmic_flags[ii+1] == 1:
        flux[ii-1] = flux[ii-2]
        flux[ii] = flux[ii-1]
      else:
        flux[ii-1] = 0.75*flux[ii-2] + 0.25*flux[ii+2]
        flux[ii+1] = 0.75*flux[ii+2] + 0.25*flux[ii-2]
        flux[ii] = (flux[ii+2]+flux[ii-2])/2

  return (flux,cosmic_flags)
