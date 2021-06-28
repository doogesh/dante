import numpy as np
import healpy as hp
from dual_messenger_tools import * 
from mock import CR_reference_gen
from wiener import WienerFilter
from scipy.interpolate import interp1d
from time import time as wall_time
import tqdm

__all__ = ["DualMessenger"]

class DualMessenger(WienerFilter):

  def __init__(self, lmax, observations): 
    '''
    Initialize the Dual Messenger algorithm
    '''
    super(DualMessenger, self).__init__(lmax, observations)

  def compute_v_trunks(self, l_start):
    """
    Compute initial truncation of signal covariance according to choice of l_start
    """
    Cov_S_interp = interp1d(np.arange(self.Cov_S.shape[0]), self.Cov_S, axis=0, bounds_error=False, fill_value=0)
    v_trunk = np.zeros(3)
    v_trunk_combined = Cov_S_interp(l_start)
    V = v_trunk_combined[0:2,0:2]
    evalues_V,_ = np.linalg.eig(V)
    v_trunk[0] = evalues_V[0]
    v_trunk[1] = evalues_V[1]
    v_trunk[2] = v_trunk_combined[2,2]
   
    return v_trunk

  def compute_v_trunks_pure_filter(self, l_start):
    """
    Compute initial truncation of signal covariance according to choice of l_start
    Pure filter case -> Trivial
    """
    Cov_S_interp = interp1d(np.arange(self.Cov_S.shape[0]), self.Cov_S, axis=0, bounds_error=False, fill_value=0)
    v_trunk = Cov_S_interp(l_start)
    return v_trunk

  def compute_inv_N_bar_EB_only(self):

    d_pix = self.d_pixel.copy()

    Cov_N_diag = np.zeros([3, d_pix[0].size])
    for k in range(3):
      Cov_N_diag[k,:] = self.Cov_N[:,k,k]

    alpha = np.min(Cov_N_diag[np.where(Cov_N_diag!=0)])*1.0 # Re-scale T slightly to avoid trouble

    self.alpha = alpha 
    self.inv_alpha = 1./alpha
    self.Cov_T = np.ones(3)*alpha
    self.inv_T = 1/self.Cov_T
    print("alpha =")
    print(alpha)
    print("### Computation of Cov_T terminated     ###")  

    Cov_N_bar = Cov_N_diag - self.alpha
    inv_N_bar_ = 1./Cov_N_bar
    inv_N_bar_[np.where(Cov_N_bar==0)] = 0

    inv_N_bar = np.zeros([d_pix[0].size, 3, 3])
    for k in range(3):
      inv_N_bar[:,k,k] = inv_N_bar_[k,:]

    self.inv_N_bar = inv_N_bar

    inv_N_bar_plus_inv_T = inv_N_bar_ + self.inv_alpha
    inv_inv_N_bar_plus_inv_T_ = 1./inv_N_bar_plus_inv_T
    inv_inv_N_bar_plus_inv_T_[np.where(inv_N_bar_plus_inv_T==0)] = 0

    inv_inv_N_bar_plus_inv_T = np.zeros([d_pix[0].size, 3, 3])

    print(inv_inv_N_bar_plus_inv_T_.shape)
    for k in range(3):
      inv_inv_N_bar_plus_inv_T[:,k,k] = inv_inv_N_bar_plus_inv_T_[k,:]
    
    self.inv_inv_N_bar_plus_inv_T = inv_inv_N_bar_plus_inv_T

  def compute_inv_N_bar(self):
    """
    New & more general formalism for dealing with masks
    Output: Cov_T, inv_T, inv_N_bar, inv_inv_N_bar_plus_inv_T
    """
    d_pix = self.d_pixel.copy()
    sigma_IQU = np.zeros((3,d_pix[0].size)) 
    inv_sigma_IQU = np.zeros((3,d_pix[0].size)) 
    cholesky_matrix = np.zeros([d_pix[0].size, 3, 3])

    for i in range(d_pix[0].size):
      sigma_IQU[:,i] = np.sqrt(np.diag(self.Cov_N[i,:,:]))
      inv_sigma_IQU[:,i] = 1./sigma_IQU[:,i]

    for i in range(d_pix[0].size):
      cholesky_matrix[i,:,:] = np.matrix(np.diag(inv_sigma_IQU[:,i]))*np.matrix(self.Cov_N[i,:,:])*np.matrix(np.diag(inv_sigma_IQU[:,i]))

    # Here we set all sigma_IQU for all masked pixels to zero, ensuring no contamination of data
    # from unobserved (masked) regions in the analysis
    if self.masked:
      inv_sigma_IQU[self.masked_idx] = 0
 
    # Diagonalize Cov_N -> sole purpose: Compute Cov_T (evectors not relevant)
    Cov_N_diag = np.zeros([3, d_pix[0].size])
    for i in range(d_pix[0].size):
      Z = self.Cov_N[i,:,:]
      evalues_Z, _ = np.linalg.eigh(Z)
      Cov_N_diag[0,i] = evalues_Z[0] 
      Cov_N_diag[1,i] = evalues_Z[1] 
      Cov_N_diag[2,i] = evalues_Z[2]

    alpha = np.min(Cov_N_diag[np.where(Cov_N_diag!=0)])*.98 # Re-scale T slightly to avoid trouble

    self.alpha = alpha 
    self.inv_alpha = 1./alpha
    self.Cov_T = np.ones(3)*alpha
    self.inv_T = 1/self.Cov_T
    print("alpha =")
    print(alpha)
    print("### Computation of Cov_T terminated     ###")  

    if self.compute_chi2: 
      inv_N = np.zeros([d_pix[0].size,3,3])

    # Compute inv_N_bar 
    inv_N_bar = np.zeros([d_pix[0].size, 3, 3])
    inv_Sigma_Q_dag = np.zeros([d_pix[0].size, 3, 3])
    Q_inv_Sigma = np.zeros([d_pix[0].size, 3, 3])
    inter_delta_term = np.zeros([3, 3])
    for idx in range(d_pix[0].size):
      alpha_i = inv_sigma_IQU[0,idx]
      beta_i = inv_sigma_IQU[1,idx]
      gamma_i = inv_sigma_IQU[2,idx]
      Y = cholesky_matrix[idx,:,:]
      evalues_Y, evectors_Y = np.linalg.eigh(Y)
      delta1 = evalues_Y[0] 
      delta2 = evalues_Y[1] 
      delta3 = evalues_Y[2]
      # Construct Q_dag using eigenvectors
      a = evectors_Y[0,0]
      b = evectors_Y[0,1]
      c = evectors_Y[0,2]
      d = evectors_Y[1,0]
      e = evectors_Y[1,1]
      f = evectors_Y[1,2]
      g = evectors_Y[2,0]
      h = evectors_Y[2,1]
      i = evectors_Y[2,2]
      inv_Sigma_Q_dag[idx,0,0] = alpha_i*a
      inv_Sigma_Q_dag[idx,1,1] = beta_i*e
      inv_Sigma_Q_dag[idx,2,2] = gamma_i*i
      inv_Sigma_Q_dag[idx,0,1] = alpha_i*b
      inv_Sigma_Q_dag[idx,0,2] = alpha_i*c
      inv_Sigma_Q_dag[idx,1,0] = beta_i*d
      inv_Sigma_Q_dag[idx,1,2] = beta_i*f
      inv_Sigma_Q_dag[idx,2,0] = gamma_i*g
      inv_Sigma_Q_dag[idx,2,1] = gamma_i*h
      Q_inv_Sigma[idx,:,0] = inv_Sigma_Q_dag[idx,0,:]
      Q_inv_Sigma[idx,:,1] = inv_Sigma_Q_dag[idx,1,:]
      Q_inv_Sigma[idx,:,2] = inv_Sigma_Q_dag[idx,2,:]
      inter_delta_term[0,0] = delta1 - (alpha*(alpha_i**2*a**2 + beta_i**2*d**2 + gamma_i**2*g**2))
      inter_delta_term[1,1] = delta2 - (alpha*(alpha_i**2*b**2 + beta_i**2*e**2 + gamma_i**2*h**2))
      inter_delta_term[2,2] = delta3 - (alpha*(alpha_i**2*c**2 + beta_i**2*f**2 + gamma_i**2*i**2))
      inter_delta_term[0,1] = -alpha*(alpha_i**2*a*b + beta_i**2*d*e + gamma_i**2*g*h)
      inter_delta_term[0,2] = -alpha*(alpha_i**2*a*c + beta_i**2*d*f + gamma_i**2*g*i)
      inter_delta_term[1,2] = -alpha*(alpha_i**2*b*c + beta_i**2*e*f + gamma_i**2*h*i)
      inter_delta_term[1,0] = inter_delta_term[0,1]
      inter_delta_term[2,0] = inter_delta_term[0,2]
      inter_delta_term[2,1] = inter_delta_term[1,2]
      inv_inter_delta_term = np.linalg.inv(inter_delta_term)
      inv_N_bar[idx,:,:] = np.matrix(inv_Sigma_Q_dag[idx,:,:])*np.matrix(inv_inter_delta_term)*np.matrix(Q_inv_Sigma[idx,:,:])
      if self.compute_chi2: 
        inv_delta_diag = np.zeros((3,3))
        inv_delta_diag[0,0] = 1./delta1
        inv_delta_diag[1,1] = 1./delta2
        inv_delta_diag[2,2] = 1./delta3
        inv_N[idx,:,:] = np.matrix(inv_Sigma_Q_dag[idx,:,:])*np.matrix(inv_delta_diag)*np.matrix(Q_inv_Sigma[idx,:,:])
    print("### Computation of inv_N_bar terminated ###")

    if self.compute_chi2: 
      self.inv_N = inv_N
      print("### Computation of inv_N terminated    ###")

    # No need for idx_null0 now
    self.inv_N_bar = inv_N_bar # shape [npix,3,3]
    inv_N_bar_plus_inv_T = inv_N_bar.copy()
    for k in range(3):
      inv_N_bar_plus_inv_T[:,k,k] += self.inv_alpha
    inv_inv_N_bar_plus_inv_T = np.zeros([d_pix[0].size, 3, 3])
    for k in range(d_pix[0].size):
      inv_inv_N_bar_plus_inv_T[k,:,:] = np.linalg.inv(inv_N_bar_plus_inv_T[k,:,:])
    self.inv_inv_N_bar_plus_inv_T = inv_inv_N_bar_plus_inv_T

  def compute_inv_D_pix(self):
    """
    New & more general formalism for dealing with masks
    Compute inverse of Cov_D_pix
    """
    sigma_IQU = np.zeros((3,self.NPIX)) 
    inv_sigma_IQU = np.zeros((3,self.NPIX)) 
    cholesky_matrix = np.zeros((self.NPIX, 3, 3))
    inv_D_pix = np.zeros((self.NPIX, 3, 3))

    for i in range(self.NPIX):
      sigma_IQU[:,i] = np.sqrt(np.diag(self.Cov_D_pix[i,:,:]))
      inv_sigma_IQU[:,i] = 1./sigma_IQU[:,i]
      cholesky_matrix[i,:,:] = np.matrix(np.diag(inv_sigma_IQU[:,i]))*np.matrix(self.Cov_D_pix[i,:,:])*np.matrix(np.diag(inv_sigma_IQU[:,i]))

    # Here we set all sigma_IQU for all masked pixels to zero, ensuring no contamination of data
    # from unobserved (masked) regions in the analysis
    if self.masked:
      inv_sigma_IQU[self.masked_idx] = 0

    inv_cholesky_matrix = np.linalg.inv(cholesky_matrix)

    for idx in range(self.NPIX):
      inv_D_pix[idx,:,:] = np.matrix(np.diag(inv_sigma_IQU[:,idx]))*np.matrix(inv_cholesky_matrix[idx,:,:])*np.matrix(np.diag(inv_sigma_IQU[:,idx]))
    self.inv_D_pix = inv_D_pix.copy()
    np.savez("inv_D_pix_masked", inv_D_pix=inv_D_pix)

    inv_D_pix_square = np.zeros((self.NPIX, 3, 3))
    # Compute inv_D_square BLOCK FORM
    for m in range(self.NPIX):
      inv_D_pix_square[m,:,:] = np.matrix(inv_D_pix[m,:,:])*np.matrix(inv_D_pix[m,:,:])
    self.inv_D_pix_square = inv_D_pix_square.copy()

  def diagonalize_D_pix(self):
    """
    Diagonalize Cov_D_pix and compute omega_constant and a constant associated term (for convenience later)
    """
    print("Diagonalizing Cov_D_pix")
    inv_D_pix_square_diag = np.zeros((3,self.NPIX))
    # To compute omega, we use an orthonormal decomposition to diagonalize inv_D_pix_square
    for m in tqdm.tqdm(range(self.NPIX)):
      Y = self.inv_D_pix_square[m,:,:]
      evalues_Y, evectors_Y = np.linalg.eigh(Y)
      inv_D_pix_square_diag[0,m] = evalues_Y[0]
      inv_D_pix_square_diag[1,m] = evalues_Y[1]
      inv_D_pix_square_diag[2,m] = evalues_Y[2]

    inv_omega_constant = np.max(inv_D_pix_square_diag)
    self.omega_constant = 1./inv_omega_constant
    self.constant_D_square_term = -1*self.inv_D_pix_square*self.omega_constant
    for k in range(3):
      self.constant_D_square_term[:,k,k] += 1.0

  def diagonalize_Cov_S_ell(self, Cov_S_mod_ell):
    """
    Diagonalize Cov_S_ell and compute S_bar_ell
    Cov_S_mod_ell is a (3,3) array
    """
    Cov_S_diag_ell = np.zeros(3)
    S_bar_diag_ell = np.zeros(3)
    inv_S_bar_diag_ell = np.zeros(3)
    matrix_P_sub = np.zeros([2,2])
    operator_P_ell = np.zeros([3,3])

    # Construct diagonalized Cov_S
    Y = Cov_S_mod_ell[0:2,0:2]
    evalues_Y, evectors_Y = np.linalg.eigh(Y)
    Cov_S_diag_ell[0] = evalues_Y[0]
    Cov_S_diag_ell[1] = evalues_Y[1]
    Cov_S_diag_ell[2] = Cov_S_mod_ell[2,2]

    # Construct P using eigenvectors
    matrix_P_sub[:,0] = evectors_Y[:,0]
    matrix_P_sub[:,1] = evectors_Y[:,1]
    operator_P_ell[0:2,0:2] = matrix_P_sub[:,:]
    operator_P_ell[2,2] = 1.

    return Cov_S_diag_ell, operator_P_ell

  def diagonalize_Cov_S(self, lmax):
    """
    Call diagonalize_Cov_S_ell to diagonalize Cov_S
    """
    Cov_S_diag = np.zeros((lmax+1,3))
    operator_P = np.zeros((lmax+1,3,3))
    
    for g in range(lmax + 1):
      Cov_S_diag_ell, operator_P_ell = self.diagonalize_Cov_S_ell(self.Cov_S[g,:,:]) 
      Cov_S_diag[g,:] = Cov_S_diag_ell
      operator_P[g,0:2,0:2] = operator_P_ell[:2,:2]
      operator_P[g,2,2] = 1.
    self.Cov_S_diag =  Cov_S_diag
    idx_null1 = np.where(Cov_S_diag == 0.)
    self.inv_S_diag = 1./Cov_S_diag
    self.inv_S_diag[idx_null1] = 0.
    self.operator_P = operator_P.copy()

  def compute_S_bar(self, v_trunk):
    """
    Construct S_bar; Subtract off v_trunks from diagonalized Cov_S
    """
    Cov_S_bar = self.Cov_S_diag - v_trunk
    idx_null = np.where(Cov_S_bar <= 0.)
    Cov_S_bar[idx_null] = 0.
    inv_S_bar = 1./Cov_S_bar
    inv_S_bar[idx_null] = 0.   
    self.Cov_S_bar =  Cov_S_bar
    self.inv_S_bar = inv_S_bar
 
  def compute_S_bar_pure_filter(self, v_trunk, lmax):
    """
    Construct S_bar for pure filter case -> Straightforward since no diagonalization required
    """    
    Cov_S_bar = self.Cov_S - v_trunk
    idx_null = np.where(Cov_S_bar <= 0.)
    Cov_S_bar[idx_null] = 0.
    inv_S_bar = 1./Cov_S_bar
    inv_S_bar[idx_null] = 0.
    self.Cov_S_bar =  Cov_S_bar.copy()
    self.inv_S_bar = inv_S_bar.copy()
    idx_null1 = np.where(self.Cov_S == 0.)
    self.inv_S_diag = 1./self.Cov_S
    self.inv_S_diag[idx_null1] = 0.

  def _compute_signal(self, t):
    """
    Compute the signal s (in harmonic) from messenger field t
    """  
    t_harmonic = np.array(hp.map2alm(t, lmax=self.lmax, pol=True, iter=0))
    if self.beamed:
      t_harmonic = numba_almxfl_vec(t_harmonic, self.beam, self.lmax)

    if self.EB_only:
      alm = numba_almxfl_vec(t_harmonic, self.s_coeff, self.lmax)
    else:
      P_t = numba_almxfl_block(t_harmonic, self.operator_P, self.lmax)
      alm_inter = numba_almxfl_vec(P_t, self.s_coeff, self.lmax)
      alm = numba_almxfl_block(alm_inter, self.operator_P, self.lmax)
    if self.jacobi_correction:
      alm = jacobi_corrector(alm, self.lmax, self.NSIDE, self.operator_P, self.s_coeff, self.inv_S_bar_plus_v, self.alpha, self.inv_norm_coeff, beam_jacobi=self.beam) 
    return alm

  def _compute_messenger_field(self, s):
    """
    Compute messenger field t (in pixel), given s and data
    """     
    if self.beamed:
      s = numba_almxfl_vec(np.array(s), self.beam, self.lmax)
    s_pixel = hp.alm2map(tuple(s), nside=self.NSIDE, lmax=self.lmax, pol=True, verbose=False) 
    t = numba_array_manipulation_TypeD(self.inv_inv_N_bar_plus_inv_T, self.inv_alpha, np.array(s_pixel), self.inv_N_bar, self.d_pixel) 
    if self.masked: 
      t[self.masked_idx] = np.array(s_pixel)[self.masked_idx]
    return t

  def run_one_iteration(self, s):
    """
    Do a complete iteration of the dual messenger algorithm
    """
    t = self._compute_messenger_field(s)
    s = self._compute_signal(t) # alms      
    return s, t

  def _compute_1st_messenger_field_anisotropic(self, s):
    """
    Compute 1st messenger field t (in pixel), given s and data
    """     
    if self.beamed:
      s = numba_almxfl_vec(np.array(s), self.beam, self.lmax)

    s_pixel = hp.alm2map(tuple(s), nside=self.NSIDE, lmax=self.lmax, pol=True, verbose=False)
    inv_D_s_pixel = numba_array_manipulation_TypeA(self.inv_D_pix, s_pixel)
    inv_D_s_harmonic = hp.map2alm(inv_D_s_pixel, lmax=self.lmax, pol=True, iter=0)
    inv_C_inv_D_s_harmonic = numba_almxfl_vec(inv_D_s_harmonic, self.inv_C_harmonic, self.lmax)
    inv_C_inv_D_s_pixel = hp.alm2map(tuple(inv_C_inv_D_s_harmonic), nside=self.NSIDE, lmax=self.lmax, pol=True, verbose=False)*self.inv_norm_coeff
    if self.jacobi_correction:
      inv_C_inv_D_s_pixel = generic_jacobi_corrector(inv_C_inv_D_s_pixel, self.Cov_C_harmonic_hack, self.inv_C_harmonic, self.lmax, self.NSIDE)
    Y_dag_inv_C_inv_D_s_pixel = hp.map2alm(inv_C_inv_D_s_pixel, lmax=self.lmax, pol=True, iter=0)*self.norm_coeff
    first_term_harmonic = numba_almxfl_vec(Y_dag_inv_C_inv_D_s_pixel, self.C_minus_phi, self.lmax)
    first_term_pixel = hp.alm2map(tuple(first_term_harmonic), nside=self.NSIDE, lmax=self.lmax, pol=True, verbose=False)

    return numba_array_addition(first_term_pixel, self.phi_inv_D_d_pix)

  def _compute_2nd_messenger_field_anisotropic(self, s, t):
    """
    Compute 2nd messenger field v (in pixel), given s and t
    """
    if self.jacobi_correction:
      t = np.array(hp.map2alm(t, lmax=self.lmax, pol=True, iter=0))
      t = hp.alm2map(tuple(t), nside=self.NSIDE, lmax=self.lmax, pol=True, verbose=False)*self.inv_norm_coeff
      t = generic_jacobi_corrector_DKR(t, self.lmax, self.NSIDE, self.norm_coeff, self.inv_norm_coeff)
      inv_D_t_pixel = numba_array_manipulation_TypeA(self.inv_D_pix, t)
      Y_dag_inv_D_t_pixel = hp.map2alm(inv_D_t_pixel, lmax=self.lmax, iter=0, pol=True)*self.norm_coeff
      omega_inv_D_t_pixel = hp.alm2map(tuple(Y_dag_inv_D_t_pixel), nside=self.NSIDE, lmax=self.lmax, pol=True, verbose=False)*self.omega_constant
    else:
      omega_inv_D_t_pixel = self.omega_constant*(numba_array_manipulation_TypeA(self.inv_D_pix, t))

    if self.beamed:
      s = numba_almxfl_vec(np.array(s), self.beam, self.lmax)

    s_pixel = hp.alm2map(tuple(s), nside=self.NSIDE, lmax=self.lmax, pol=True, verbose=False)

    if self.jacobi_correction:
      inv_D_s_pixel = numba_array_manipulation_TypeA(self.inv_D_pix, s_pixel)
      inv_D_s_pixel = np.array(hp.map2alm(inv_D_s_pixel, lmax=self.lmax, pol=True, iter=0))
      inv_D_s_pixel = hp.alm2map(tuple(inv_D_s_pixel), nside=self.NSIDE, lmax=self.lmax, pol=True, verbose=False)*self.inv_norm_coeff
      inv_D_s_pixel = generic_jacobi_corrector_DKR(inv_D_s_pixel, self.lmax, self.NSIDE, self.norm_coeff, self.inv_norm_coeff)
      inv_D_Y_Y_dag_inv_D_s_pixel = numba_array_manipulation_TypeA(self.inv_D_pix, inv_D_s_pixel)
      Y_dag_inv_D_Y_Y_dag_inv_D_s_pixel = hp.map2alm(inv_D_Y_Y_dag_inv_D_s_pixel, lmax=self.lmax, iter=0, pol=True)*self.norm_coeff
      Y_Y_dag_inv_D_Y_Y_dag_inv_D_s_pixel = hp.alm2map(tuple(Y_dag_inv_D_Y_Y_dag_inv_D_s_pixel), nside=self.NSIDE, lmax=self.lmax, pol=True, verbose=False)*self.omega_constant
      second_term_pixel = s_pixel - Y_Y_dag_inv_D_Y_Y_dag_inv_D_s_pixel
      #second_term_pixel = s_pixel - hp.alm2map(hp.map2alm(numba_array_manipulation_TypeA(self.inv_D_pix, generic_jacobi_corrector_DKR(numba_array_manipulation_TypeA(self.inv_D_pix, s_pixel), self.lmax, self.NSIDE, self.norm_coeff, self.inv_norm_coeff)), lmax=self.lmax, iter=0, pol=True), nside=self.NSIDE, verbose=False, pol=True)*self.norm_coeff*self.omega_constant
    else:
      second_term_pixel = numba_array_manipulation_TypeA(self.constant_D_square_term, s_pixel)

    return numba_array_addition(omega_inv_D_t_pixel, second_term_pixel)

  def _compute_signal_anisotropic(self, v):
    """
    Compute the signal s (in harmonic) from 2nd messenger field v
    """

    if self.jacobi_correction:
      v = np.array(hp.map2alm(v, lmax=self.lmax, pol=True, iter=0))
      v = hp.alm2map(tuple(v), nside=self.NSIDE, lmax=self.lmax, pol=True, verbose=False)*self.inv_norm_coeff
      v = generic_jacobi_corrector_DKR(v, self.lmax, self.NSIDE, self.norm_coeff, self.inv_norm_coeff)
      v_harmonic = np.array(hp.map2alm(v, lmax=self.lmax, pol=True, iter=0))*self.norm_coeff
    else:
      v_harmonic = np.array(hp.map2alm(v, lmax=self.lmax, pol=True, iter=0))

    if self.beamed:
      v_harmonic = numba_almxfl_vec(v_harmonic, self.beam, self.lmax)

    P_v = numba_almxfl_block(v_harmonic, self.operator_P, self.lmax)

    alm_inter = numba_almxfl_vec(P_v, self.s_coeff, self.lmax) # s_coeff -> [lmax+1,3], P_v -> typical alm 
    alm = numba_almxfl_block(alm_inter, self.operator_P, self.lmax)

    if self.jacobi_correction:    
      alm = jacobi_corrector_anisotropic(alm, self.lmax, self.NSIDE, self.operator_P, self.s_coeff, self.inv_S_bar_plus_v, self.phi_constant*self.omega_constant, self.norm_coeff, self.inv_norm_coeff, beam_jacobi=self.beam)

    return alm

  def run_one_iteration_anisotropic(self, s):
    """
    Do a complete iteration of the dual messenger algorithm
    """
    t = self._compute_1st_messenger_field_anisotropic(s)
    v = self._compute_2nd_messenger_field_anisotropic(s,t)
    s = self._compute_signal_anisotropic(v)
    return s

  def _compute_signal_anisotropic_pure_filter(self, v):
    """
    Compute the signal s (in harmonic) from 2nd messenger field v
    """ 
    if self.jacobi_correction:
      v = np.array(hp.map2alm(v, lmax=self.lmax, pol=True, iter=0))
      v = hp.alm2map(tuple(v), nside=self.NSIDE, lmax=self.lmax, pol=True, verbose=False)*self.inv_norm_coeff
      v = generic_jacobi_corrector_DKR(v, self.lmax, self.NSIDE, self.norm_coeff, self.inv_norm_coeff)
      v_harmonic = np.array(hp.map2alm(v, lmax=self.lmax, pol=True, iter=0))*self.norm_coeff
    else:
      v_harmonic = np.array(hp.map2alm(v, lmax=self.lmax, pol=True, iter=0))

    if self.beamed:
      v_harmonic = numba_almxfl_vec(v_harmonic, self.beam, self.lmax)
    
    alm = numba_almxfl_vec(v_harmonic, self.s_coeff, self.lmax)
    if self.jacobi_correction:    
      alm = jacobi_corrector_anisotropic_pure(alm, self.lmax, self.NSIDE, self.s_coeff, self.inv_S_bar_plus_v, self.phi_constant*self.omega_constant, self.norm_coeff, self.inv_norm_coeff, beam_jacobi=self.beam)
    return alm

  def run_one_iteration_anisotropic_pure_filter(self, s):
    """
    Do a complete iteration of the dual messenger algorithm, with pure E/B filter
    """
    t = self._compute_1st_messenger_field_anisotropic(s)
    v = self._compute_2nd_messenger_field_anisotropic(s,t)
    s = self._compute_signal_anisotropic_pure_filter(v)
    return s

  def compute_norm(self, s):
    """ 
    Compute norm of the solution 's'
    """
    return np.linalg.norm(s)

  def compute_chi_squared(self, s):
    """
    Compute chi2 of the solution 's'
    """
    d_pix = self.d_pixel.copy()
    s_pix = hp.alm2map(tuple(s), nside=self.NSIDE, lmax=self.lmax, pol=True, verbose=False)
    d_minus_s = d_pix - s_pix
    inv_N_times_d_minus_s = numba_array_manipulation_TypeA(self.inv_N, d_minus_s)
    first_term_chi2 = numba_array_multiplication(d_minus_s, inv_N_times_d_minus_s).sum()
    inv_S_times_alm = numba_almxfl_block(s, self.inv_S, self.lmax)
    second_term_chi2 = numba_array_multiplication_harmonic(s.conj(), inv_S_times_alm).sum()
    chi2 = np.abs(first_term_chi2 + second_term_chi2)

    return chi2

  def compute_chi_squared_anisotropic(self, s):
    """
    Compute chi2 of the solution 's'
    """
    d_pix = self.d_pixel.copy()
    s_pix = hp.alm2map(tuple(s), nside=self.NSIDE, lmax=self.lmax, pol=True, verbose=False)
    d_minus_s = d_pix - s_pix

    inv_D_times_d_minus_s = numba_array_manipulation_TypeA(self.inv_D_pix, d_minus_s)

    inter_A = np.array(hp.map2alm(inv_D_times_d_minus_s, lmax=self.lmax, pol=True, iter=0))
    inter_B = numba_almxfl_vec(inter_A, self.inv_C_harmonic, self.lmax)
    inter_C = hp.alm2map(tuple(inter_B), nside=self.NSIDE, lmax=self.lmax, pol=True, verbose=False)*self.inv_norm_coeff
    if self.jacobi_correction:
      inter_C = generic_jacobi_corrector(inter_C, self.Cov_C_harmonic_hack, self.inv_C_harmonic, self.lmax, self.NSIDE)
    inv_N_times_d_minus_s = numba_array_manipulation_TypeA(self.inv_D_pix, inter_C)
    first_term_chi2 = numba_array_multiplication(d_minus_s, inv_N_times_d_minus_s).sum()

    if self.pure_E or self.pure_B:
      inv_S_times_alm = numba_almxfl_vec(s, self.inv_S, self.lmax)
    else:
      inv_S_times_alm = numba_almxfl_block(s, self.inv_S, self.lmax)
    second_term_chi2 = numba_array_multiplication_harmonic(s.conj(), inv_S_times_alm).sum()

    return np.abs(first_term_chi2 + second_term_chi2)

  def compute_chi_squared_anisotropic_cholesky(self, s):
    """
    Compute chi2 of the solution 's'
    """
    d_pix = self.d_pixel.copy()
    s_pix = hp.alm2map(tuple(s), nside=self.NSIDE, lmax=self.lmax, pol=True, verbose=False)
    d_minus_s = d_pix - s_pix

    inv_D_times_d_minus_s = numba_array_manipulation_TypeA(self.inv_D_pix, d_minus_s)
    Y_inv_inv_D_times_d_minus_s = np.array(hp.map2alm(inv_D_times_d_minus_s, lmax=self.lmax, pol=True)) ### FIXME: , use_pixel_weights=True 
    inv_C_sqrt_Y_inv_inv_D_times_d_minus_s = numba_almxfl_vec(Y_inv_inv_D_times_d_minus_s, np.sqrt(self.inv_C_harmonic), self.lmax)
    first_term_chi2 = numba_array_multiplication_harmonic(inv_C_sqrt_Y_inv_inv_D_times_d_minus_s.conj(), inv_C_sqrt_Y_inv_inv_D_times_d_minus_s).sum()

    if self.pure_E or self.pure_B:
      inv_S_times_alm = numba_almxfl_vec(s, self.inv_S, self.lmax)
    else:
      inv_S_times_alm = numba_almxfl_block(s, self.inv_S, self.lmax)
    second_term_chi2 = numba_array_multiplication_harmonic(s.conj(), inv_S_times_alm).sum()
    chi2 = np.abs(first_term_chi2 + second_term_chi2)

    return chi2

  def compute_residual_error_anisotropic(self, s):
    """
    Compute (PCG) residual error |Ax-y|/|y| of the solution 's' -> Cov_S is block-diagonal, so includes basis transformations
    """
    
    P_a = numba_almxfl_block(s, self.operator_P, self.lmax)
    inv_S_sqrt_P_a = numba_almxfl_vec(P_a, np.sqrt(self.inv_S_diag), self.lmax)

    def A_op(vec):

      ###inv_D_P_Cov_S_sqrt_vec_pix = numba_array_manipulation_TypeA(self.inv_D_pix, hp.alm2map(tuple(numba_almxfl_block(numba_almxfl_vec(vec, np.sqrt(self.Cov_S_diag), self.lmax), self.operator_P, self.lmax)), nside=self.NSIDE, lmax=self.lmax, pol=True, verbose=False))

      ###inter_action = numba_almxfl_vec(numba_almxfl_block(np.array(hp.map2alm(numba_array_manipulation_TypeA(self.inv_D_pix, hp.alm2map(tuple(numba_almxfl_vec(np.array(hp.map2alm(inv_D_P_Cov_S_sqrt_vec_pix, lmax=self.lmax, pol=True)), self.inv_C_harmonic, self.lmax)), nside=self.NSIDE, lmax=self.lmax, pol=True, verbose=False)), lmax=self.lmax, pol=True, iter=0)), self.operator_P, self.lmax), np.sqrt(self.Cov_S_diag), self.lmax)

      inter_A = numba_almxfl_vec(vec, np.sqrt(self.Cov_S_diag), self.lmax)
      inter_B = numba_almxfl_block(inter_A, self.operator_P, self.lmax)
      inter_C = hp.alm2map(tuple(inter_B), nside=self.NSIDE, lmax=self.lmax, pol=True, verbose=False)
      inv_D_P_Cov_S_sqrt_vec_pix = numba_array_manipulation_TypeA(self.inv_D_pix, inter_C)
      inter_D = np.array(hp.map2alm(inv_D_P_Cov_S_sqrt_vec_pix, lmax=self.lmax, pol=True, iter=0))
      inter_E = numba_almxfl_vec(inter_D, self.inv_C_harmonic, self.lmax)
      inter_F = hp.alm2map(tuple(inter_E), nside=self.NSIDE, lmax=self.lmax, pol=True, verbose=False)*self.inv_norm_coeff
      if self.jacobi_correction:
        inter_F = generic_jacobi_corrector(inter_F, self.Cov_C_harmonic_hack, self.inv_C_harmonic, self.lmax, self.NSIDE)
      inter_G = numba_array_manipulation_TypeA(self.inv_D_pix, inter_F)
      inter_H = np.array(hp.map2alm(inter_G, lmax=self.lmax, pol=True, iter=0))*self.norm_coeff
      inter_I = numba_almxfl_block(inter_H, self.operator_P, self.lmax)
      inter_action = numba_almxfl_vec(inter_I, np.sqrt(self.Cov_S_diag), self.lmax)

      return vec + inter_action

    return np.linalg.norm(A_op(inv_S_sqrt_P_a) - self.y_vec)/self.norm_y

  def compute_residual_error_anisotropic_ell(self, s):
    """
    Compute (PCG) residual error |Ax-y|/|y| of the solution 's' -> Cov_S is block-diagonal, so includes basis transformations
    """
    
    P_a = numba_almxfl_block(s, self.operator_P, self.lmax)
    inv_S_sqrt_P_a = numba_almxfl_vec(P_a, np.sqrt(self.inv_S_diag), self.lmax)

    def A_op(vec):

      inter_A = numba_almxfl_vec(vec, np.sqrt(self.Cov_S_diag), self.lmax)
      inter_B = numba_almxfl_block(inter_A, self.operator_P, self.lmax)
      inter_C = hp.alm2map(tuple(inter_B), nside=self.NSIDE, lmax=self.lmax, pol=True, verbose=False)
      inv_D_P_Cov_S_sqrt_vec_pix = numba_array_manipulation_TypeA(self.inv_D_pix, inter_C)
      inter_D = np.array(hp.map2alm(inv_D_P_Cov_S_sqrt_vec_pix, lmax=self.lmax, pol=True, iter=0))
      inter_E = numba_almxfl_vec(inter_D, self.inv_C_harmonic, self.lmax)
      inter_F = hp.alm2map(tuple(inter_E), nside=self.NSIDE, lmax=self.lmax, pol=True, verbose=False)*self.inv_norm_coeff
      if self.jacobi_correction:
        inter_F = generic_jacobi_corrector(inter_F, self.Cov_C_harmonic_hack, self.inv_C_harmonic, self.lmax, self.NSIDE)
      inter_G = numba_array_manipulation_TypeA(self.inv_D_pix, inter_F)
      inter_H = np.array(hp.map2alm(inter_G, lmax=self.lmax, pol=True, iter=0))*self.norm_coeff
      inter_I = numba_almxfl_block(inter_H, self.operator_P, self.lmax)
      inter_action = numba_almxfl_vec(inter_I, np.sqrt(self.Cov_S_diag), self.lmax)

      return vec + inter_action

    return np.sqrt( hp.alm2cl(A_op(inv_S_sqrt_P_a) - self.y_vec)/hp.alm2cl(self.y_vec) )

  def compute_residual_error_anisotropic_pure(self, s):
    """
    Compute (PCG) residual error |Ax-y|/|y| of the solution 's' -> Cov_S is fully diagonal, so does not include basis transformations
    """
    inv_S_sqrt_a = numba_almxfl_vec(s, np.sqrt(self.inv_S), self.lmax)

    def A_op(vec):

      inter_A = numba_almxfl_vec(vec, np.sqrt(self.Cov_S_error), self.lmax)
      inter_C = hp.alm2map(tuple(inter_A), nside=self.NSIDE, lmax=self.lmax, pol=True, verbose=False)
      inv_D_Cov_S_sqrt_vec_pix = numba_array_manipulation_TypeA(self.inv_D_pix, inter_C)
      inter_D = np.array(hp.map2alm(inv_D_Cov_S_sqrt_vec_pix, lmax=self.lmax, pol=True, iter=0))
      inter_E = numba_almxfl_vec(inter_D, self.inv_C_harmonic, self.lmax)
      inter_F = hp.alm2map(tuple(inter_E), nside=self.NSIDE, lmax=self.lmax, pol=True, verbose=False)*self.inv_norm_coeff
      if self.jacobi_correction:
        inter_F = generic_jacobi_corrector(inter_F, self.Cov_C_harmonic_hack, self.inv_C_harmonic, self.lmax, self.NSIDE)
      inter_G = numba_array_manipulation_TypeA(self.inv_D_pix, inter_F)
      inter_H = np.array(hp.map2alm(inter_G, lmax=self.lmax, pol=True, iter=0))*self.norm_coeff
      inter_action = numba_almxfl_vec(inter_H, np.sqrt(self.Cov_S_error), self.lmax)

      return vec + inter_action

    inter = A_op(inv_S_sqrt_a) - self.y_vec

    if self.pure_E:
      out = np.linalg.norm(inter[1,:])/np.linalg.norm(self.y_vec[1,:])
    if self.pure_B:
      out = np.linalg.norm(inter[2,:])/np.linalg.norm(self.y_vec[2,:])

    return out

  def compute_residual_error_anisotropic_pure_ell(self, s):
    """
    Compute (PCG) residual error |Ax-y|/|y| of the solution 's' -> Cov_S is fully diagonal, so does not include basis transformations
    """
    inv_S_sqrt_a = numba_almxfl_vec(s, np.sqrt(self.inv_S), self.lmax)

    def A_op(vec):

      inter_A = numba_almxfl_vec(vec, np.sqrt(self.Cov_S_error), self.lmax)
      inter_C = hp.alm2map(tuple(inter_A), nside=self.NSIDE, lmax=self.lmax, pol=True, verbose=False)
      inv_D_Cov_S_sqrt_vec_pix = numba_array_manipulation_TypeA(self.inv_D_pix, inter_C)
      inter_D = np.array(hp.map2alm(inv_D_Cov_S_sqrt_vec_pix, lmax=self.lmax, pol=True, iter=0))
      inter_E = numba_almxfl_vec(inter_D, self.inv_C_harmonic, self.lmax)
      inter_F = hp.alm2map(tuple(inter_E), nside=self.NSIDE, lmax=self.lmax, pol=True, verbose=False)*self.inv_norm_coeff
      if self.jacobi_correction:
        inter_F = generic_jacobi_corrector(inter_F, self.Cov_C_harmonic_hack, self.inv_C_harmonic, self.lmax, self.NSIDE)
      inter_G = numba_array_manipulation_TypeA(self.inv_D_pix, inter_F)
      inter_H = np.array(hp.map2alm(inter_G, lmax=self.lmax, pol=True, iter=0))*self.norm_coeff
      inter_action = numba_almxfl_vec(inter_H, np.sqrt(self.Cov_S_error), self.lmax)

      return vec + inter_action

    return np.sqrt( hp.alm2cl(A_op(inv_S_sqrt_a) - self.y_vec)/hp.alm2cl(self.y_vec) )

  ### Keep the option of using chi2 as convergence criterion.
  def check_convergence(self, s, state, convergence, precision):
    if convergence == 'norm':
      if state is None: # First iteration
        return False, s, 1.0
      last_conv = self.compute_norm(s-state)/self.compute_norm(state) # computes normalized Cauchy criterion |s_new - s_old|/|s_old|
      converged = last_conv < precision
      print("Delta Norm = %lg" % last_conv)
      return converged, s, last_conv

    # Eventually to add other options such as 'chi2' and 'ref'
    # 'ref' takes in a reference solution , and you specify a tolerance of your final solution w.r.t this reference
    '''
    elif convergence == 'chi2':
      chi2 = self.compute_chi_squared(s)
      if state is None:
        return False, s, chi2 
      converged = np.abs(chi2-state) < precision
      print("Delta Chi2 = %lg" % (chi2-state))
      return converged, s, chi2

    elif convergence =='ref':
      ref,prec = precision
      N = self.compute_norm(s-ref,ref)
      return N < prec, None
    ''' 

############################################################
##                     CLASSIC FILTER                     ##
############################################################

  def run_classic_filter(self, convergence='norm', precision=10**-4, cooling_step=2.5, l_start=100, relaxed_convergence_threshold=False, store_steps=False, jacobi_correction=False, constrained_realizations=False, compute_chi2=False, compute_residual=False, EB_only=False, CAMB_Cov_S_fname="pol_data_boost_totCls.dat", Cov_S_provided=None, noise_amplitude_CR=None, Cov_N_CR=None):
    """
    Run the dual messenger algorithm to the given precision
    ***This is the main function that calls all preliminaries & initializes all constant coefficients***
    """
    W  = '\033[0m'  # white (normal)
    G  = '\033[32m' # green
    P  = '\033[35m' # purple
    O  = '\033[33m' # orange
    B  = '\033[34m' # blue
    start1 = wall_time()

    if self.anisotropic_noise:
      raise ValueError("Wrong (classic) filter chosen!")

    self.EB_only = EB_only
    if self.EB_only:
      print(G+"*** E/B only mode activated               ***"+W)
    else:
      print(P+"*** E/B only mode deactivated             ***"+W)

    if Cov_S_provided is not None:
      self.Cov_S = np.load(Cov_S_provided)["Cov_S"]
    else:
      _, self.Cov_S = DKR_read_camb_cl(CAMB_Cov_S_fname, self.lmax, self.EB_only)

    if self.EB_only:
      Cov_S_diag = np.zeros((self.lmax+1,3))
      for i in range(3):
        Cov_S_diag[:,i] = self.Cov_S[:,i,i]
      self.Cov_S_diag = Cov_S_diag.copy()
      idx_null1 = np.where(self.Cov_S_diag == 0.)
      self.inv_S_diag = 1./Cov_S_diag
      self.inv_S_diag[idx_null1] = 0.
      self.operator_P = np.zeros((self.lmax+1,3,3))
      for k in range(3):
        self.operator_P[:,k,k] = 1.0
    else:
      self.diagonalize_Cov_S(self.lmax)

    self.jacobi_correction = jacobi_correction
    if self.jacobi_correction:
      print(G+"*** Jacobi corrector activated           ***"+W)
    else:
      print(P+"*** Jacobi corrector deactivated         ***"+W)

    if relaxed_convergence_threshold:
      print(G+"*** Relaxed convergence activated        ***"+W)
    else:
      print(P+"*** Relaxed convergence deactivated      ***"+W)

    if constrained_realizations:
      print(G+"*** Constrained realizations activated   ***"+W)
      # Generate reference signal and data maps
      # IMPORTANT: Ensure consistency between the noise amplitude used in mock generation and CR generation
      self.noise_amplitude_CR = noise_amplitude_CR
      if Cov_N_CR is not None:
        self.Cov_N_CR = np.load(Cov_N_CR)["Cov_N"]
      if Cov_S_provided is not None:
        self.Cov_S_CR = self.Cov_S.copy()
      else:
        self.Cov_S_CR = None
      alm_ref, s_ref, d_ref = CR_reference_gen(self.NSIDE, self.lmax, self.noise_amplitude_CR, self.Cov_N_CR, self.EB_only, self.Cov_S_CR, self.beam) 
      self.d_pixel -= d_ref
    else:
      print(P+"*** Constrained realizations deactivated ***"+W)

    self.compute_chi2 = compute_chi2
    if compute_chi2: 
      self.inv_S = compute_inv_2x2(self.Cov_S, self.lmax)
      print(G+"*** chi2 evaluation activated            ***"+W)

    if compute_residual: 
      #self.y_norm = 
      print(G+"*** Residual error evaluation activated  ***"+W)

    print(self.initializer)
    print(G+"$$$          CLASSIC FILTER          $$$"+W) 
    print(B+"$$$ cooling step = %4.1f $$$" % cooling_step, "precision = %4.1e $$$" % precision, "l_start = %4.1f" % l_start, " $$$"+W) 
    print("### Beginning pre-computations          ###")

    if convergence not in ('norm', 'ref', 'chi2'): 
      raise ValueError("Invalid convergence value")
     
    lmax = self.lmax
    self.nside = hp.npix2nside(self.d_pixel[0].size)
    self.norm_coeff = (12*self.nside**2)/(4*np.pi) 
    self.inv_norm_coeff = 1/self.norm_coeff

    d_pix = self.d_pixel.copy()

    if self.EB_only:
      self.compute_inv_N_bar_EB_only()
    else:
      self.compute_inv_N_bar()

    inv_norm_coeff_times_T = self.inv_norm_coeff*self.Cov_T
    self.inv_N_bar_d = numba_array_manipulation_TypeA(self.inv_N_bar, d_pix)
    s = new_s(lmax, self.NPIX)
 
    # Saving all intermediate solutions. Set store_steps = False if this is not desired.
    full_steps = [] 
    # Saving all info of interest to us in record_actions
    record_actions = []
    last_iteration = False 
    # Initialize empty list to store fractional differences for convergence test (residual errors) & chi2
    diff_conv = []
    residual_conv = []
    chi2 = []
    # Initialize empty list to store epsilon
    epsilon_list = []
    state = None # state is s_old, i.e. solution at previous step
    
    # l_start -> Choice for initial truncation of signal covariance
    if self.EB_only:
      v_trunk_ = self.compute_v_trunks_pure_filter(l_start)
      v_trunk = np.zeros((3))
      for i in range(3):
        v_trunk[i] = v_trunk_[i,i]
    else:
      v_trunk = self.compute_v_trunks(l_start)
    '''
    if self.beamed:
      zeta = (self.beam_square*inv_norm_coeff_times_T) + v_trunk
    else:
      zeta = inv_norm_coeff_times_T + v_trunk
    '''
    zeta = inv_norm_coeff_times_T + v_trunk

    # For relaxed convergence threshold mode, always start at 10^-4 
    precision_mod = precision
    if relaxed_convergence_threshold: 
      if precision_mod >= 10**-4:
        relaxed_convergence_threshold = False
        print(O+"For relaxed convergence mode, final required precision must be below 10^-4"+W)
        print(O+"Deactivating relaxed convergence mode"+W)
      else:
        precision_mod = 10**-4
        eps_threshold_step = compute_threshold_step(precision, zeta, inv_norm_coeff_times_T, cooling_step) 
        print("eps_threshold_step =")
        print(eps_threshold_step)
    end1 = wall_time()
  
    print("### Precomputations over                ###")
    print("Execution time, in seconds,")
    print(end1 - start1)
    start2 = wall_time()

    while True:

      print("v_trunk (mu) =")
      print(v_trunk)
      self.compute_S_bar(v_trunk)
      if self.beamed:
        self.Cov_S_bar_plus_v_trunk = numba_array_addition(self.Cov_S_bar, v_trunk)
        self.s_coeff = compute_s_coeff_beam(inv_norm_coeff_times_T, self.Cov_S_bar_plus_v_trunk, self.beam_square, self.operator_P) 
      else:
        self.s_coeff = compute_s_coeff(self.Cov_S_bar, v_trunk, zeta)
        
      # Precompute the following if Jacobi corrector is activated
      if self.jacobi_correction:      
        self.inv_S_bar_plus_v = compute_inv_S_bar_plus_v(self.Cov_S_bar, v_trunk)
        idx_v_trunk = np.where(np.array(v_trunk) == 0)
        # Fixing singularity whenever v_trunk = 0 for T/E/B for Jacobi corrector
        self.inv_S_bar_plus_v[:,idx_v_trunk] = self.inv_S_diag[:,idx_v_trunk]

      i_iter = 0
      converged = False
      time_for_action = wall_time()
      while not converged:
        s, t = self.run_one_iteration(s)
        converged, state, last_conv = self.check_convergence(s, state, convergence, precision_mod)
        epsilon_list.append(precision_mod)
        diff_conv.append(last_conv)
        i_iter += 1
        if compute_chi2:
          chi2.append(self.compute_chi_squared(s))
        #if compute_residual:
          #residual_conv.append(self.compute_residual_error(s)) ### TODO: This should be a different function
        if store_steps:
          full_steps.append(s)

      time_for_action -= wall_time()   
      record_actions.append((i_iter,time_for_action))

      if last_iteration:
        converged = True
        break

      # Cooling scheme for zeta
      zeta /= cooling_step 
      # Compute corresponding v_trunk
      '''
      if self.beamed:
        v_trunk = zeta - (self.beam_square*inv_norm_coeff_times_T)
      else:
        v_trunk = zeta - inv_norm_coeff_times_T
      '''
      v_trunk = zeta - inv_norm_coeff_times_T
      
      cond_v_trunk = np.where(v_trunk < 0)
      v_trunk[cond_v_trunk] = 0.
      zeta[cond_v_trunk] = inv_norm_coeff_times_T[cond_v_trunk]
      ### More sophisticated cooling scheme for epsilon 
      if relaxed_convergence_threshold:
        precision_mod /= eps_threshold_step 
		
      # Set v_trunk to zero if all v_trunks go below a certain threshold
      if (zeta[0] == inv_norm_coeff_times_T[0]) and (zeta[1] == inv_norm_coeff_times_T[1]) and (zeta[2] == inv_norm_coeff_times_T[2]):
        v_trunk = np.zeros(3)
        last_iteration = True
        precision_mod = precision # This is significant only for relaxed convergence mode

    # WF solution stored last, but this is a duplicate of penultimate occupant in the list
    if store_steps:
      full_steps.append(s)
         
    self.last_actions = record_actions
    self.last_solution = s
    self.all_steps = full_steps
    s_wf = hp.alm2map(tuple(s), nside=self.NSIDE, lmax=self.lmax, pol=True, verbose=False)

    end2 = wall_time()
    execution_time = end2 - start2

    if constrained_realizations:
      s += alm_ref
      s_wf = hp.alm2map(tuple(s), nside=self.NSIDE, lmax=self.lmax, pol=True, verbose=False)

    print(G+"*** Dante execution complete!           ***"+W)
    print("Execution time for main loop, in seconds, is")
    print(execution_time)
    print("Corresponding to the following number of iterations:")
    print(len(diff_conv))

    return s, s_wf, d_pix, full_steps, diff_conv, residual_conv, epsilon_list, chi2


############################################################
##                   ANISOTROPIC FILTER                   ##
############################################################

  def run_anistropic_filter(self, pure_E=False, pure_B=False, convergence='norm', precision=10**-4, cooling_step=2.5, l_start=100, relaxed_convergence_threshold=False, store_steps=False, jacobi_correction=False, constrained_realizations=False, compute_chi2=False, compute_residual=False):
    """
    Run the dual messenger algorithm to the given precision
    ***This is the main function that calls all preliminaries & initializes all constant coefficients***
    """
    W  = '\033[0m'  # white (normal)
    G  = '\033[32m' # green
    P  = '\033[35m' # purple
    O  = '\033[33m' # orange
    B  = '\033[34m' # blue
    start1 = wall_time()
    _, self.Cov_S = DKR_read_camb_cl("pol_data_boost_totCls.dat", self.lmax)

    if pure_E:
      print(G+"*** Pure E filter activated              ***"+W)
      _, self.Cov_S = DKR_read_camb_cl_pure_E("pol_data_boost_totCls.dat", self.lmax)
    if pure_B:
      print(G+"*** Pure B filter activated              ***"+W)
      _, self.Cov_S = DKR_read_camb_cl_pure_B("pol_data_boost_totCls.dat", self.lmax)


    self.jacobi_correction = jacobi_correction
    if self.jacobi_correction:
      print(G+"*** Jacobi corrector activated           ***"+W)
    else:
      print(P+"*** Jacobi corrector deactivated         ***"+W)

    if relaxed_convergence_threshold:
      print(G+"*** Relaxed convergence activated        ***"+W)
    else:
      print(P+"*** Relaxed convergence deactivated      ***"+W)

    if constrained_realizations:
      print(G+"*** Constrained realizations activated   ***"+W)
      # Generate reference signal and data maps
      # IMPORTANT: CR_reference_gen uses the default value for noise amplitude per pixel (ensure consistency)
      alm_ref, s_ref, d_ref = CR_reference_gen(self.NSIDE, self.lmax, self.beam) 
      self.d_pixel -= d_ref
    else:
      print(P+"*** Constrained realizations deactivated ***"+W)

    self.compute_chi2 = compute_chi2
    if compute_chi2: 
      self.inv_S = compute_inv_2x2(self.Cov_S, self.lmax)
      self.pure_E = False
      self.pure_B = False
      print(G+"*** chi2 evaluation activated            ***"+W)

    self.compute_residual = compute_residual
    if compute_residual: 
      print(G+"*** Residual error evaluation activated  ***"+W)

    print(self.initializer)
    print(G+"$$$          ANISOTROPIC FILTER          $$$"+W) 
    print(B+"$$$ cooling step = %4.1f $$$" % cooling_step, "precision = %4.1e $$$" % precision, "l_start = %4.1f" % l_start, " $$$"+W) 
    print("### Beginning pre-computations          ###")

    if convergence not in ('norm', 'ref', 'chi2'): 
      raise ValueError("Invalid convergence value")
     
    lmax = self.lmax
    self.norm_coeff = (12*self.NSIDE**2)/(4*np.pi) 
    self.inv_norm_coeff = 1/self.norm_coeff

    d_pix = self.d_pixel.copy()

    # We need to encode mask in Cov_D_pix
    self.compute_inv_D_pix()
    self.diagonalize_D_pix()
    self.diagonalize_Cov_S(self.lmax)
    ###self.inv_C_harmonic = np.linalg.inv(self.Cov_C_harmonic)
    ### Temporary hack until mock generation is amended
    self.Cov_C_harmonic_hack = np.zeros((lmax+1,3))
    for k in range(3):
      self.Cov_C_harmonic_hack[:,k] = self.Cov_C_harmonic[:,k,k]
    self.inv_C_harmonic = 1./self.Cov_C_harmonic_hack
    self.phi_constant = np.min(self.Cov_C_harmonic_hack[np.where(self.Cov_C_harmonic_hack != 0)]) # This constraint no longer required if fully-diagonal
    print("phi is %f" % self.phi_constant)
    print("omega is %.9f" % self.omega_constant)
    
    self.inv_D_d_pix = numba_array_manipulation_TypeA(self.inv_D_pix, d_pix)
    self.C_minus_phi = self.Cov_C_harmonic_hack - self.phi_constant
    inv_D_d_harmonic = np.array(hp.map2alm(self.inv_D_d_pix, lmax=self.lmax, pol=True, iter=0))
    inv_C_inv_D_d_harmonic = numba_almxfl_vec(inv_D_d_harmonic, self.inv_C_harmonic, self.lmax)
    inv_C_inv_D_d_pixel = hp.alm2map(tuple(inv_C_inv_D_d_harmonic), nside=self.NSIDE, lmax=self.lmax, pol=True, verbose=False)*self.inv_norm_coeff

    if self.jacobi_correction:
      inv_C_inv_D_d_pixel = generic_jacobi_corrector(inv_C_inv_D_d_pixel, self.Cov_C_harmonic_hack, self.inv_C_harmonic, self.lmax, self.NSIDE)
    Y_dag_inv_C_inv_D_d_harmonic = np.array(hp.map2alm(inv_C_inv_D_d_pixel, lmax=self.lmax, pol=True, iter=0))*self.norm_coeff
    self.phi_inv_D_d_pix = hp.alm2map(tuple(Y_dag_inv_C_inv_D_d_harmonic), nside=self.NSIDE, lmax=self.lmax, pol=True, verbose=False)*self.phi_constant

    if compute_residual:
      #self.y_vec = numba_almxfl_vec( numba_almxfl_block(np.array(hp.map2alm(numba_array_manipulation_TypeA(self.inv_D_pix, hp.alm2map(tuple(numba_almxfl_vec(np.array(hp.map2alm(self.inv_D_d_pix, lmax=self.lmax, pol=True)), self.inv_C_harmonic, self.lmax)), nside=self.NSIDE, lmax=self.lmax, pol=True, verbose=False)), lmax=self.lmax, pol=True, iter=0)), self.operator_P, self.lmax), np.sqrt(self.Cov_S_diag), self.lmax)
      
      ### FIXME: Make more compact --> Nested functions
      inter_A = np.array(hp.map2alm(self.inv_D_d_pix, lmax=self.lmax, pol=True, iter=0))
      inter_B = numba_almxfl_vec(inter_A, self.inv_C_harmonic, self.lmax)
      inter_C = hp.alm2map(tuple(inter_B), nside=self.NSIDE, lmax=self.lmax, pol=True, verbose=False)*self.inv_norm_coeff
      if self.jacobi_correction:
        inter_C = generic_jacobi_corrector(inter_C, self.Cov_C_harmonic_hack, self.inv_C_harmonic, self.lmax, self.NSIDE)
      inter_D = numba_array_manipulation_TypeA(self.inv_D_pix, inter_C)
      inter_E = np.array(hp.map2alm(inter_D, lmax=self.lmax, pol=True, iter=0))*self.norm_coeff
      inter_F = numba_almxfl_block(inter_E, self.operator_P, self.lmax)
      self.y_vec = numba_almxfl_vec(inter_F, np.sqrt(self.Cov_S_diag), self.lmax)
      self.norm_y = np.linalg.norm(self.y_vec)

    s = new_s(lmax, self.NPIX)
 
    # Saving all intermediate solutions. Set store_steps = False if this is not desired.
    full_steps = [] 
    # Saving all info of interest to us in record_actions
    record_actions = []
    last_iteration = False 
    # Initialize empty list to store fractional differences for convergence test (residual errors) & chi2
    diff_conv = []
    residual_conv = []
    chi2 = []
    # Initialize empty list to store epsilon
    epsilon_list = []
    state = None # state is s_old, i.e. solution at previous step
    
    # l_start -> Choice for initial truncation of signal covariance 
    v_trunk = self.compute_v_trunks(l_start)
    # This is important only if we use naive implementation with numerically large values in Cov_S
    if pure_E:
      v_trunk[0] = 0.
      v_trunk[2] = 0.
    if pure_B:
      v_trunk[1] = 0.
    '''
    if self.beamed:
      zeta = (self.beam_square*cooling_constant) + v_trunk
    else:
      zeta = cooling_constant + v_trunk
    '''
    cooling_constant = np.ones(3)*self.phi_constant*self.omega_constant
    zeta = cooling_constant + v_trunk
    print("zeta is")
    print(zeta)

    # For relaxed convergence threshold mode, always start at 10^-4 
    precision_mod = precision
    if relaxed_convergence_threshold: 
      if precision_mod >= 10**-4:
        relaxed_convergence_threshold = False
        print(O+"For relaxed convergence mode, final required precision must be below 10^-4"+W)
        print(O+"Deactivating relaxed convergence mode"+W)
      else:
        precision_mod = 10**-4
        if pure_E:
          eps_threshold_step = compute_threshold_step_pure_E(precision, zeta, cooling_constant, cooling_step) 
        else:
          eps_threshold_step = compute_threshold_step(precision, zeta, cooling_constant, cooling_step) 
        print("eps_threshold_step =")
        print(eps_threshold_step)
    end1 = wall_time()      
  
    print("### Precomputations over                ###")
    print("Execution time, in seconds,")
    print(end1 - start1)
    start2 = wall_time()

    residual_error_ell = []
    while True:

      print("v_trunk (mu) =")
      print(v_trunk)

      self.compute_S_bar(v_trunk)

      if self.beamed:
        self.Cov_S_bar_plus_v_trunk = numba_array_addition(self.Cov_S_bar, v_trunk)
        self.s_coeff = compute_s_coeff_beam(zeta, self.Cov_S_bar_plus_v_trunk, self.beam_square, self.operator_P) 
      else:
        self.s_coeff = compute_s_coeff(self.Cov_S_bar, v_trunk, zeta) 

      # Precompute the following if Jacobi corrector is activated
      if self.jacobi_correction:      
        self.inv_S_bar_plus_v = compute_inv_S_bar_plus_v(self.Cov_S_bar, v_trunk)
        idx_v_trunk = np.where(np.array(v_trunk) == 0)
        # Fixing singularity whenever v_trunk = 0 for T/E/B for Jacobi corrector
        self.inv_S_bar_plus_v[:,idx_v_trunk] = self.inv_S_diag[:,idx_v_trunk]

      i_iter = 0
      converged = False
      time_for_action = wall_time()
      while not converged:
        s = self.run_one_iteration_anisotropic(s)
        converged, state, last_conv = self.check_convergence(s, state, convergence, precision_mod)
        epsilon_list.append(precision_mod)
        diff_conv.append(last_conv)
        i_iter += 1

        if compute_chi2: 
          chi2.append(self.compute_chi_squared_anisotropic_cholesky(s))
        if compute_residual:
          residual_conv.append(self.compute_residual_error_anisotropic(s))   
          if last_iteration:
            residual_error_ell = self.compute_residual_error_anisotropic_ell(s)
        if store_steps:
          full_steps.append(s)

      time_for_action -= wall_time()   
      record_actions.append((i_iter,time_for_action))

      if last_iteration:
        converged = True
        break

      # Cooling scheme for zeta
      zeta /= cooling_step 
      # Compute corresponding v_trunk
      '''
      if self.beamed:
        v_trunk = zeta - (self.beam_square*cooling_constant)
      else:
        v_trunk = zeta - cooling_constant
      '''
      v_trunk = zeta - cooling_constant
      
      cond_v_trunk = np.where(v_trunk < 0)
      v_trunk[cond_v_trunk] = 0.
      zeta[cond_v_trunk] = cooling_constant[cond_v_trunk]
      ### More sophisticated cooling scheme for epsilon 
      if relaxed_convergence_threshold:
        precision_mod /= eps_threshold_step 
		
      # Set v_trunk to zero if all v_trunks go below a certain threshold
      if (zeta[0] == cooling_constant[0]) and (zeta[1] == cooling_constant[1]) and (zeta[2] == cooling_constant[2]):
        v_trunk = np.zeros(3)
        last_iteration = True
        precision_mod = precision # This is significant only for relaxed convergence mode

    # WF solution stored last, but this is a duplicate of penultimate occupant in the list
    if store_steps:
      full_steps.append(s)
         
    self.last_actions = record_actions
    self.last_solution = s
    self.all_steps = full_steps
    s_wf = hp.alm2map(tuple(s), nside=self.NSIDE, lmax=self.lmax, pol=True, verbose=False)

    end2 = wall_time()
    execution_time = end2 - start2

    if constrained_realizations:
      s += alm_ref
      s_wf = hp.alm2map(tuple(s), nside=self.NSIDE, lmax=self.lmax, pol=True, verbose=False)

    print(G+"*** Dante execution complete!           ***"+W)
    print("Execution time for main loop, in seconds, is")
    print(execution_time)
    print("Corresponding to the following number of iterations:")
    print(len(diff_conv))

    return s, s_wf, d_pix, full_steps, diff_conv, residual_conv, epsilon_list, chi2, residual_error_ell

############################################################
##             ANISOTROPIC FILTER + PURE E/B              ##
############################################################

  def run_anistropic_pure_filter(self, pure_filter='pure_B', convergence='norm', precision=10**-4, cooling_step=2.5, l_start=100, relaxed_convergence_threshold=False, store_steps=False, jacobi_correction=False, constrained_realizations=False, compute_chi2=False, compute_residual=False):
    """
    Run the dual messenger algorithm to the given precision
    ***This is the main function that calls all preliminaries & initializes all constant coefficients***
    """
    W  = '\033[0m'  # white (normal)
    G  = '\033[32m' # green
    P  = '\033[35m' # purple
    O  = '\033[33m' # orange
    B  = '\033[34m' # blue
    start1 = wall_time()

    if pure_filter not in ('pure_E', 'pure_B'): 
      raise ValueError("Invalid pure filter value")
    if pure_filter == 'pure_B':
      _, self.Cov_S = DKR_residual_error_read_pure_EB_camb_cl("pol_data_boost_totCls.dat", self.lmax, pure_filter) #DKR_read_pure_EB_camb_cl("pol_data_boost_totCls.dat", self.lmax, pure_filter) ###FIXME
      _, self.Cov_S_error = DKR_residual_error_read_pure_EB_camb_cl("pol_data_boost_totCls.dat", self.lmax, pure_filter)
      self.pure_B = True
      self.pure_E = False
      print(G+"*** Pure B filter activated              ***"+W)
    if pure_filter == 'pure_E':
      _, self.Cov_S = DKR_residual_error_read_pure_EB_camb_cl("pol_data_boost_totCls.dat", self.lmax, pure_filter) #DKR_read_pure_EB_camb_cl("pol_data_boost_totCls.dat", self.lmax, pure_filter) ###FIXME
      _, self.Cov_S_error = DKR_residual_error_read_pure_EB_camb_cl("pol_data_boost_totCls.dat", self.lmax, pure_filter)
      self.pure_E = True
      self.pure_B = False
      print(G+"*** Pure E filter activated              ***"+W)

    self.jacobi_correction = jacobi_correction
    if self.jacobi_correction:
      print(G+"*** Jacobi corrector activated           ***"+W)
    else:
      print(P+"*** Jacobi corrector deactivated         ***"+W)

    if relaxed_convergence_threshold:
      print(G+"*** Relaxed convergence activated        ***"+W)
    else:
      print(P+"*** Relaxed convergence deactivated      ***"+W)

    if constrained_realizations:
      print(G+"*** Constrained realizations activated   ***"+W)
      # Generate reference signal and data maps
      # IMPORTANT: CR_reference_gen uses the default value for noise amplitude per pixel (ensure consistency)
      alm_ref, s_ref, d_ref = CR_reference_gen(self.NSIDE, self.lmax, self.beam) 
      self.d_pixel -= d_ref
    else:
      print(P+"*** Constrained realizations deactivated ***"+W)

    self.compute_chi2 = compute_chi2
    if compute_chi2:
      self.inv_S = 1./self.Cov_S
      self.inv_S[np.where(self.Cov_S==0)] = 0.
      print(G+"*** chi2 evaluation activated            ***"+W)

    if compute_residual: 
      print(G+"*** residual error evaluation activated  ***"+W)

    print(self.initializer)
    print(G+"$$$          PURE E/B FILTER            $$$"+W) 
    print(B+"$$$ cooling step = %4.1f $$$" % cooling_step, "precision = %4.1e $$$" % precision, "l_start = %4.1f" % l_start, " $$$"+W) 
    print("### Beginning pre-computations          ###")

    if convergence not in ('norm', 'ref', 'chi2'): 
      raise ValueError("Invalid convergence value")
     
    lmax = self.lmax
    self.norm_coeff = (12*self.NSIDE**2)/(4*np.pi) 
    self.inv_norm_coeff = 1/self.norm_coeff

    d_pix = self.d_pixel.copy()

    # We need to encode mask in Cov_D_pix
    self.compute_inv_D_pix()

    self.diagonalize_D_pix()
    ###self.inv_C_harmonic = np.linalg.inv(self.Cov_C_harmonic)
    ### Temporary hack until mock generation is amended
    self.Cov_C_harmonic_hack = np.zeros((lmax+1,3))
    for k in range(3):
      self.Cov_C_harmonic_hack[:,k] = self.Cov_C_harmonic[:,k,k]
    self.inv_C_harmonic = 1./self.Cov_C_harmonic_hack
    self.phi_constant = np.min(self.Cov_C_harmonic_hack[np.where(self.Cov_C_harmonic_hack != 0)]) # This constraint no longer required if fully-diagonal
    print("phi is %f" % self.phi_constant)
    print("omega is %.9f" % self.omega_constant)
    
    self.inv_D_d_pix = numba_array_manipulation_TypeA(self.inv_D_pix, d_pix)
    self.C_minus_phi = self.Cov_C_harmonic_hack - self.phi_constant

    inv_D_d_harmonic = np.array(hp.map2alm(self.inv_D_d_pix, lmax=self.lmax, pol=True, iter=0))
    inv_C_inv_D_d_harmonic = numba_almxfl_vec(inv_D_d_harmonic, self.inv_C_harmonic, self.lmax)
    inv_C_inv_D_d_pixel = hp.alm2map(tuple(inv_C_inv_D_d_harmonic), nside=self.NSIDE, lmax=self.lmax, pol=True, verbose=False)*self.inv_norm_coeff

    if self.jacobi_correction:
      inv_C_inv_D_d_pixel = generic_jacobi_corrector(inv_C_inv_D_d_pixel, self.Cov_C_harmonic_hack, self.inv_C_harmonic, self.lmax, self.NSIDE)
    Y_dag_inv_C_inv_D_d_harmonic = np.array(hp.map2alm(inv_C_inv_D_d_pixel, lmax=self.lmax, pol=True, iter=0))*self.norm_coeff
    self.phi_inv_D_d_pix = hp.alm2map(tuple(Y_dag_inv_C_inv_D_d_harmonic), nside=self.NSIDE, lmax=self.lmax, pol=True, verbose=False)*self.phi_constant

    if compute_residual:
      #self.y_vec = numba_almxfl_vec(np.array(hp.map2alm(numba_array_manipulation_TypeA(self.inv_D_pix, hp.alm2map(tuple(numba_almxfl_vec(np.array(hp.map2alm(self.inv_D_d_pix, lmax=self.lmax, pol=True)), self.inv_C_harmonic, self.lmax)), nside=self.NSIDE, lmax=self.lmax, pol=True, verbose=False)), lmax=self.lmax, pol=True, iter=0)), np.sqrt(self.Cov_S_error), self.lmax)
      
      ### FIXME: Make more compact --> Nested functions
      inter_A = np.array(hp.map2alm(self.inv_D_d_pix, lmax=self.lmax, pol=True, iter=0))
      inter_B = numba_almxfl_vec(inter_A, self.inv_C_harmonic, self.lmax)
      inter_C = hp.alm2map(tuple(inter_B), nside=self.NSIDE, lmax=self.lmax, pol=True, verbose=False)*self.inv_norm_coeff
      if self.jacobi_correction:
        inter_C = generic_jacobi_corrector(inter_C, self.Cov_C_harmonic_hack, self.inv_C_harmonic, self.lmax, self.NSIDE)
      inter_D = numba_array_manipulation_TypeA(self.inv_D_pix, inter_C)
      inter_E = np.array(hp.map2alm(inter_D, lmax=self.lmax, pol=True, iter=0))*self.norm_coeff
      self.y_vec = numba_almxfl_vec(inter_E, np.sqrt(self.Cov_S_error), self.lmax)
      self.norm_y = np.linalg.norm(self.y_vec)

    s = new_s(lmax, self.NPIX)
 
    # Saving all intermediate solutions. Set store_steps = False if this is not desired.
    full_steps = [] 
    # Saving all info of interest to us in record_actions
    record_actions = []
    last_iteration = False 
    # Initialize empty list to store fractional differences for convergence test (residual errors) & chi2
    diff_conv = []
    residual_conv = []
    chi2 = []
    # Initialize empty list to store epsilon
    epsilon_list = []
    state = None # state is s_old, i.e. solution at previous step
    
    # l_start -> Choice for initial truncation of signal covariance 
    v_trunk = self.compute_v_trunks_pure_filter(l_start)
    '''
    if self.beamed:
      zeta = (self.beam_square*cooling_constant) + v_trunk
    else:
      zeta = cooling_constant + v_trunk
    '''

    # This is important only we if use naive implementation with numerically large values in Cov_S
    if self.pure_E:
      v_trunk[0] = 0.
      v_trunk[2] = 0.
    if self.pure_B:
      v_trunk[1] = 0.

    cooling_constant = np.ones(3)*self.phi_constant*self.omega_constant
    zeta = cooling_constant + v_trunk
    print("zeta is")
    print(zeta)

    # For relaxed convergence threshold mode, always start at 10^-4 
    precision_mod = precision
    if relaxed_convergence_threshold: 
      if precision_mod >= 10**-4:
        relaxed_convergence_threshold = False
        print(O+"For relaxed convergence mode, final required precision must be below 10^-4"+W)
        print(O+"Deactivating relaxed convergence mode"+W)
      else:
        precision_mod = 10**-4
        if self.pure_E:
          eps_threshold_step = compute_threshold_step_pure_E(precision, zeta, cooling_constant, cooling_step) 
        else:
          eps_threshold_step = compute_threshold_step(precision, zeta, cooling_constant, cooling_step) 
        print("eps_threshold_step =")
        print(eps_threshold_step)
    end1 = wall_time()
  
    print("### Precomputations over                ###")
    print("Execution time, in seconds,")
    print(end1 - start1)
    start2 = wall_time()

    while True:

      print("v_trunk (mu) =")
      print(v_trunk)
      self.compute_S_bar_pure_filter(v_trunk, lmax)

      if self.beamed:
        self.Cov_S_bar_plus_v_trunk = numba_array_addition(self.Cov_S_bar, v_trunk)
        self.s_coeff = compute_s_coeff_beam_pure(zeta, self.Cov_S_bar_plus_v_trunk, self.beam_square) 
      else:
        self.idx_v_trunk = np.where(np.array(v_trunk) == 0)
        inv_v_trunk = 1./v_trunk
        inv_v_trunk[self.idx_v_trunk] = 0.
        self.inv_S_bar_plus_v = self.inv_S_bar*(1./(self.inv_S_bar + inv_v_trunk))*inv_v_trunk
        # Fixing singularity whenever v_trunk = 0 for T/E/B for Jacobi corrector
        self.inv_S_bar_plus_v[:,self.idx_v_trunk] = self.inv_S_diag[:,self.idx_v_trunk]
        self.s_coeff = 1./((self.inv_S_bar_plus_v*self.phi_constant*self.omega_constant) + 1.0)

      i_iter = 0
      converged = False
      time_for_action = wall_time()
      while not converged:
        s = self.run_one_iteration_anisotropic_pure_filter(s)
        converged, state, last_conv = self.check_convergence(s, state, convergence, precision_mod)
        epsilon_list.append(precision_mod)
        diff_conv.append(last_conv)
        i_iter += 1

        if compute_chi2:
          chi2.append(self.compute_chi_squared_anisotropic_cholesky(s))
        if compute_residual:
          residual_conv.append(self.compute_residual_error_anisotropic_pure(s))
          if last_iteration:
            residual_error_ell = self.compute_residual_error_anisotropic_pure_ell(s)
        if store_steps:
          full_steps.append(s)

      time_for_action -= wall_time()
      record_actions.append((i_iter,time_for_action))

      if last_iteration:
        converged = True
        break

      # Cooling scheme for zeta
      zeta /= cooling_step 
      # Compute corresponding v_trunk
      '''
      if self.beamed:
        v_trunk = zeta - (self.beam_square*cooling_constant)
      else:
        v_trunk = zeta - cooling_constant
      '''
      v_trunk = zeta - cooling_constant
      
      cond_v_trunk = np.where(v_trunk < 0)
      v_trunk[cond_v_trunk] = 0.
      zeta[cond_v_trunk] = cooling_constant[cond_v_trunk]
      ### More sophisticated cooling scheme for epsilon 
      if relaxed_convergence_threshold:
        precision_mod /= eps_threshold_step 
		
      # Set v_trunk to zero if all v_trunks go below a certain threshold
      if (zeta[0] == cooling_constant[0]) and (zeta[1] == cooling_constant[1]) and (zeta[2] == cooling_constant[2]):
        v_trunk = np.zeros(3)
        last_iteration = True
        precision_mod = precision # This is significant only for relaxed convergence mode

    # WF solution stored last, but this is a duplicate of penultimate occupant in the list
    if store_steps:
      full_steps.append(s)

    self.last_actions = record_actions
    self.last_solution = s
    self.all_steps = full_steps
    s_wf = hp.alm2map(tuple(s), nside=self.NSIDE, lmax=self.lmax, pol=True, verbose=False)

    end2 = wall_time()
    execution_time = end2 - start2

    if constrained_realizations:
      s += alm_ref
      s_wf = hp.alm2map(tuple(s), nside=self.NSIDE, lmax=self.lmax, pol=True, verbose=False)

    if self.pure_E:
      print(G+"*** Dante (Pure E) execution complete!  ***"+W)
    if self.pure_B:
      print(G+"*** Dante (Pure B) execution complete!  ***"+W)

    print("Execution time for main loop, in seconds, is")
    print(execution_time)
    print("Corresponding to the following number of iterations:")
    print(len(diff_conv))

    return s, s_wf, d_pix, full_steps, diff_conv, residual_conv, epsilon_list, chi2, residual_error_ell

