### This observations module just loads data files (.fits) of interest
### Its sole purpose is to load the Cov_N and respective CMB data maps

import healpy as hp
import numpy as np

__all__ = ["Observations"]

UNSEEN_PLANCK = -1.6382041955959526e+30

class Observations(object):

  def __init__(self, map_name, noise_name=None, mask_name=None, beam_name=None, anisotropic_noise_name=None):
    self.map_name = map_name
    self.noise_name = noise_name
    self.mask_name = mask_name
    self.beam_name = beam_name
    self.anisotropic_noise_name = anisotropic_noise_name
    self._obs = None
    self._noise = None
    self._mask = None
    self._beam = None
    self._anisotropic_noise = None

  def load_obs_map(self):
    return self._obs if self._obs is not None else hp.read_map(self.map_name, field=(0,1,2))

  def load_mask(self):
    return self._mask if self.mask_name is None else hp.read_map(self.mask_name, field=(0,1,2))

  def load_beam(self):
    return self._beam if self.beam_name is None else np.load(self.beam_name)['beam']

  def load_noise_map(self):
    return self._noise if self.noise_name is None else np.load(self.noise_name)['Cov_N']

  def load_modulation_anisotropic_noise(self):
    return self._anisotropic_noise if self.anisotropic_noise_name is None else np.load(self.anisotropic_noise_name)['Cov_D_pix']

  def load_correlation_anisotropic_noise(self):
    return self._anisotropic_noise if self.anisotropic_noise_name is None else np.load(self.anisotropic_noise_name)['Cov_C_harmonic']

