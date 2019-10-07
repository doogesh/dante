import numpy as np
import healpy as hp
 
class WienerFilter(object):
 
  def __init__(self, lmax, data):
    R  = '\033[31m' # red
    P  = '\033[35m' # purple
    O  = '\033[33m' # orange
    G  = '\033[32m' # green
    W  = '\033[0m'  # white (normal)
    B  = '\033[34m' # blue
    self.Cov_N = data.load_noise_map()
    self.d_pixel = np.array(data.load_obs_map())
    self.lmax = lmax
    self.NSIDE = hp.npix2nside(self.d_pixel[0].size)
    self.NPIX = hp.nside2npix(self.NSIDE)
    # Masked mode triggered by providing a mask
    self.masked = False
    self.mask = data.load_mask()
    # Beamed mode triggered by providing a beam
    self.beamed = False
    self.beam = data.load_beam()
    # For anisotropic filter, we need to load the noise covariance components
    self.anisotropic_noise = False
    self.Cov_D_pix = data.load_modulation_anisotropic_noise()
    self.Cov_C_harmonic = data.load_correlation_anisotropic_noise()
    print(W+"###"+G+" Activating"+W+" / "+P+"Deactivating"+W+"            ###")
    if self.mask is not None:
      self.masked = True
      self.masked_idx = np.where(np.array(self.mask) == 0) # Need to convert mask from tuple to array
      print(G+"*** Mask mode triggered                  ***"+W)
    else: 
      print(P+"*** No mask provided                     ***"+W)
    if self.beam is not None:
      self.beamed = True
      self.beam_square = self.beam**2 # Required for cooling scheme
      print(G+"*** Beam mode triggered                  ***"+W)
    else: 
      print(P+"*** No beam provided                     ***"+W)
    if self.Cov_D_pix is not None:
      self.anisotropic_noise = True
      print(G+"*** Anisotropic noise detected           ***"+W)
    else:
      print(P+"*** White noise provided                 ***"+W)

    self.initializer = W+"""
    W E L C O M E ... """+B+"""

     _______________________________________________________        _        _______________________________________________________                                                                
    |"""+W+"""o"""+B+"""|/\*\*\///\*\*\///\*\*\///\*\*\///\*\*\///\*\*\///\/"""+W+"""o"""+B+"""/       / \       \\"""+W+"""o"""+B+"""\/\*\*\///\*\*\///\*\*\///\*\*\///\*\*\///\*\*\///\|"""+W+"""o"""+B+"""|
    |"""+W+"""o"""+B+"""|\///\*\*\///\*\*\///\*\*\///\*\*\///\*\*\///\*\*\/"""+W+"""o"""+B+"""/       / """+O+"""_"""+B+""" \       \\"""+W+"""o"""+B+"""\///\*\*\///\*\*\///\*\*\///\*\*\///\*\*\///\*\*\/|"""+W+"""o"""+B+"""|
    |"""+W+"""o"""+B+"""|/\*\*\///\*\*\///\*\*\///\*\*\///\*\*\///\*\*\///|"""+W+"""o"""+B+"""|      / """+O+"""(_)"""+B+""" \      |"""+W+"""o"""+B+"""|\*\*\///\*\*\///\*\*\///\*\*\///\*\*\///\*\*\///\|"""+W+"""o"""+B+"""|
    |"""+W+"""o"""+B+"""|============================"""+W+"""*"""+B+"""====================|"""+W+"""o"""+B+"""|     /       \     |"""+W+"""o"""+B+"""|===================="""+W+"""*"""+B+"""============================|"""+W+"""o"""+B+"""|
    |"""+W+"""o"""+B+"""|=========================="""+W+"""*****"""+B+"""==================|"""+W+"""o"""+B+"""|    /   """+W+""""""+W+"""***"""+B+""""""+B+"""   \    |"""+W+"""o"""+B+"""|=================="""+W+"""*****"""+B+"""==========================|"""+W+"""o"""+B+"""|
    |"""+W+"""o"""+B+"""|==========================="""+W+"""***"""+B+"""===================|"""+W+"""o"""+B+"""|   /"""+O+""" D A N T E """+B+"""\   |"""+W+"""o"""+B+"""|==================="""+W+"""***"""+B+"""===========================|"""+W+"""o"""+B+"""|
    |"""+W+"""o"""+B+"""|\///\*\*\///\*\*\///\*\*\///\*\*\///\*\*\///\*\*\|"""+W+"""o"""+B+"""|  /     """+W+"""***"""+B+"""     \  |"""+W+"""o"""+B+"""|///\*\*\///\*\*\///\*\*\///\*\*\///\*\*\///\*\*\/|"""+W+"""o"""+B+"""|
    |"""+W+"""o"""+B+"""|/\*\*\///\*\*\///\*\*\///\*\*\///\*\*\///\*\*\///\\"""+W+"""o"""+B+"""\ /_ _ _ _ _ _ _ _\ /"""+W+"""o"""+B+"""/\*\*\///\*\*\///\*\*\///\*\*\///\*\*\///\*\*\///\|"""+W+"""o"""+B+"""|
    |"""+W+"""o"""+B+"""|\///\*\*\///\*\*\///\*\*\///\*\*\///\*\*\///\*\*\/\\"""+W+"""o"""+B+"""\                 /"""+W+"""o"""+B+"""/\///\*\*\///\*\*\///\*\*\///\*\*\///\*\*\///\*\*\/|"""+W+"""o"""+B+"""|
    !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!
    """+O+"""
    (c) Doogesh Kodi Ramanah, 2017-2019
        Guilhem Lavaux, 2017-2019
    """+W

  # By default, convergence criterion is Cauchy, with tolerance 10^-3
  def run_classic_filter(self, data, convergence='norm', precision=10**-3):
    raise NotImplementedError("run_classic_filter is not implemented")
