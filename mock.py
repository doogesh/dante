import numpy as np
import healpy as hp
from dual_messenger_tools import *
from astropy.io import fits
import tqdm

def _build_mask(npix, nside, lmax, maskname):
  """
  Builds a map based on choice of npix/nside by upgrading/degrading accordingly
  """
  # Load a particular mask
  f = fits.open(maskname)
  d = f[1].data
  # Re-order mask to have usual "ring" format
  m_T = hp.reorder(d['TMASK'], n2r = True)
  m_P = hp.reorder(d['PMASK'], n2r = True)
  # Initialize mask
  mask = np.zeros([3, npix])
  # Upgrade/degrade mask if ever mask is of different resolution than required
  if npix != m_T.size:
    print("### Upgrading/degrading mask to match choice of nside ###")
    m_T = hp.ud_grade(m_T, nside_out=nside)
    m_P = hp.ud_grade(m_P, nside_out=nside)

  # Assign the temperature & polarization components of mask 
  mask[0,:] = m_T
  mask[1:,:] = m_P
  return mask

def _extract_beam(lmax, beamname):
  """
  Extracts beam from fits file -> Shape [lmax+1,3]
  """
  # Load a particular beam 
  f = fits.open(beamname)
  beam_mod = f[2].data
  beam = np.zeros((lmax+1, 3))
  # Assign temperature & polarization components accordingly
  beam[:,0] = beam_mod['INT_BEAM'][:lmax+1] # T
  beam[:,1] = beam_mod['POL_BEAM'][:lmax+1] # Q
  beam[:,2] = beam[:,1]                     # U
  return beam  

def _mock_noise_covariance_generator(npix, noise_amplitude, EB_only):
  """
  Generates noise according to a given Cov_N, with default noise amplitude being 100 muK
  IMPORTANT: For CR generation, ensure that default value of noise amplitude is consistent with mock map/noise
  """
  print("### Generating mock noise covariance ###")
  n_I = np.zeros(npix)
  n_Q = np.zeros(npix)
  n_U = np.zeros(npix)
  noise_IQU = np.zeros((3, npix)) 
  Cov_N = np.zeros((npix, 3, 3))
  # This will now vary on the grid -> This is how we can implement inhomogeneous noise for IQU maps
  sigma_IQU = np.zeros((3,npix))
  if EB_only:
    sigma_IQU[1:,:] = noise_amplitude # noise amplitude (per pixel) in muK
  else:
    sigma_IQU[:,:] = noise_amplitude # noise amplitude (per pixel) in muK

  print("### Noise amplitude per pixel = %3.1f muK ###" % noise_amplitude)
  # No T/P masks here --> To be loaded in observations module

  Cov_N_triangular = np.zeros((npix, 3, 3)) # Lower triangular form, i.e. L
  # Compute elements of L, set values of alpha, beta, gamma s.t. constraints are obeyed
  alpha = 0.4
  beta = 0.7
  gamma = 0.3

  a = 1.
  b = np.sqrt(1 - alpha**2)
  c = np.sqrt( (1 - alpha**2 - beta**2 - gamma**2 + (2*alpha*beta*gamma))/(1 - alpha**2) )
  d = alpha
  e = beta
  f = (gamma - (alpha*beta))/np.sqrt(1 - alpha**2)

  # Construct Cov_N
  for i in range(npix):
    Cov_N[i,0,0] = a**2*sigma_IQU[0,i]**2
    Cov_N[i,0,1] = a*d*sigma_IQU[0,i]*sigma_IQU[1,i] 
    Cov_N[i,0,2] = a*e*sigma_IQU[0,i]*sigma_IQU[2,i]
    Cov_N[i,1,0] = a*d*sigma_IQU[0,i]*sigma_IQU[1,i] 
    Cov_N[i,1,1] = (b**2 + d**2)*sigma_IQU[1,i]**2
    Cov_N[i,1,2] = ((e*d) + (b*f))*sigma_IQU[1,i]*sigma_IQU[2,i]
    Cov_N[i,2,0] = a*e*sigma_IQU[2,i]*sigma_IQU[0,i]
    Cov_N[i,2,1] = ((e*d) + (b*f))*sigma_IQU[2,i]*sigma_IQU[1,i]
    Cov_N[i,2,2] = (e**2 + f**2 + c**2)*sigma_IQU[2,i]**2

  # Construct L (lower triangular)
  Cov_N_triangular[:,0,0] = a
  Cov_N_triangular[:,1,1] = b
  Cov_N_triangular[:,2,2] = c
  Cov_N_triangular[:,1,0] = d
  Cov_N_triangular[:,2,0] = e
  Cov_N_triangular[:,2,1] = f

  # Generate noise according to Cov_N
  for k in range(npix):
    vec = sigma_IQU[:,k]*np.dot(Cov_N_triangular[k], np.random.randn(3))
    n_I[k] = vec[0]
    n_Q[k] = vec[1]
    n_U[k] = vec[2]
  noise_IQU[0,:] = n_I
  noise_IQU[1,:] = n_Q
  noise_IQU[2,:] = n_U

  return noise_IQU, Cov_N

def construct_input_D_pix(nside, lmax, norm_coeff):
  """
  Builds the input/reference D_pix
  """
  npix = hp.nside2npix(nside)

  # Initialize D_square_pix (covariance) & D_pix (sigma)
  D_square_pix = np.zeros((npix,3,3))
  D_square_pix_diag = np.zeros((npix,3))
  operator_Q = np.zeros((npix,3,3))
  operator_Q_dag = np.zeros((npix,3,3))
  D_pix = np.zeros((npix,3,3))

  # Load relevant fits file
  f0 = fits.open("./HFI_SkyMap_100_2048_R2.02_full.fits")
  noise_covariance = f0[1].data
  # Due to symmetricity, we need only 6 components out of 9
  Cov_II = hp.reorder(noise_covariance['II_COV'],n2r=True)
  Cov_QQ = hp.reorder(noise_covariance['QQ_COV'],n2r=True)
  Cov_UU = hp.reorder(noise_covariance['UU_COV'],n2r=True)
  Cov_IQ = hp.reorder(noise_covariance['IQ_COV'],n2r=True)
  Cov_IU = hp.reorder(noise_covariance['IU_COV'],n2r=True)
  Cov_QU = hp.reorder(noise_covariance['QU_COV'],n2r=True)

  # Downgrade/upgrade map to desired resolution (choice of npix)
  if npix != Cov_II.size:
    print("### Upgrading/degrading map to match choice of nside ###")
    Cov_II = hp.ud_grade(Cov_II, nside_out=nside, power=-2)
    Cov_QQ = hp.ud_grade(Cov_QQ, nside_out=nside, power=-2)
    Cov_UU = hp.ud_grade(Cov_UU, nside_out=nside, power=-2)
    Cov_IQ = hp.ud_grade(Cov_IQ, nside_out=nside, power=-2)
    Cov_IU = hp.ud_grade(Cov_IU, nside_out=nside, power=-2)
    Cov_QU = hp.ud_grade(Cov_QU, nside_out=nside, power=-2)

  # Assign elements of D_pix 
  D_square_pix[:,0,0] = Cov_II
  D_square_pix[:,1,1] = Cov_QQ
  D_square_pix[:,2,2] = Cov_UU
  D_square_pix[:,0,1] = Cov_IQ
  D_square_pix[:,1,0] = Cov_IQ
  D_square_pix[:,0,2] = Cov_IU
  D_square_pix[:,2,0] = Cov_IU
  D_square_pix[:,1,2] = Cov_QU
  D_square_pix[:,2,1] = Cov_QU

  sigma_IQU = np.zeros((3,npix))
  inv_sigma_IQU = np.zeros((3,npix))
  cholesky_matrix = np.zeros((npix, 3, 3))

  for i in range(npix):
    sigma_IQU[:,i] = np.sqrt(np.diag(D_square_pix[i,:,:]))
    inv_sigma_IQU[:,i] = 1./sigma_IQU[:,i]
    cholesky_matrix[i,:,:] = np.matrix(np.diag(inv_sigma_IQU[:,i]))*np.matrix(D_square_pix[i,:,:])*np.matrix(np.diag(inv_sigma_IQU[:,i]))

  # Normalize D_square_pix
  sigma_IQU /= sigma_IQU.max()

  for i in range(npix):
    D_square_pix[i,:,:] = np.matrix(np.diag(sigma_IQU[:,i]))*np.matrix(cholesky_matrix[i,:,:])*np.matrix(np.diag(sigma_IQU[:,i]))

  print("Diagonalizing D_square_pix")
  # To compute D_pix, we use an orthonormal decomposition to diagonalize D_square_pix
  for m in tqdm.tqdm(range(npix)):
    Y = D_square_pix[m,:,:]
    evalues_Y, evectors_Y = np.linalg.eigh(Y)
    D_square_pix_diag[m,0] = evalues_Y[0] 
    D_square_pix_diag[m,1] = evalues_Y[1] 
    D_square_pix_diag[m,2] = evalues_Y[2] 
    # Construct transpose of Q --> Q_dag
    operator_Q[m,:,0] = evectors_Y[0,:]
    operator_Q[m,:,1] = evectors_Y[1,:]
    operator_Q[m,:,2] = evectors_Y[2,:]
    # Construct Q using eigenvectors
    operator_Q_dag[m,:,0] = evectors_Y[:,0]
    operator_Q_dag[m,:,1] = evectors_Y[:,1]
    operator_Q_dag[m,:,2] = evectors_Y[:,2]

  # Compute D_pix_diag
  D_pix_diag = np.sqrt(D_square_pix_diag)
  print("Constructing D_pix")
  # Transform/Rotate back to original basis
  for k in tqdm.tqdm(range(npix)):
    D_pix[k,:,:] = np.matrix(operator_Q_dag[k,:,:])*np.matrix(np.diag(D_pix_diag[k,:]))*np.matrix(operator_Q[k,:,:])

  ### Introduce band limit in D_pix
  Y_inv_D_II = np.array(hp.map2alm(D_pix[:,0,0], lmax, pol=False, use_pixel_weights=True))
  Y_inv_D_QQ = np.array(hp.map2alm(D_pix[:,1,1], lmax, pol=False, use_pixel_weights=True))
  Y_inv_D_UU = np.array(hp.map2alm(D_pix[:,2,2], lmax, pol=False, use_pixel_weights=True))
  Y_inv_D_IQ = np.array(hp.map2alm(D_pix[:,0,1], lmax, pol=False, use_pixel_weights=True))
  Y_inv_D_QU = np.array(hp.map2alm(D_pix[:,1,2], lmax, pol=False, use_pixel_weights=True))
  Y_inv_D_IU = np.array(hp.map2alm(D_pix[:,0,2], lmax, pol=False, use_pixel_weights=True))

  D_pix[:,0,0] = hp.alm2map(Y_inv_D_II, nside, lmax, pol=False, verbose=False)
  D_pix[:,1,1] = hp.alm2map(Y_inv_D_QQ, nside, lmax, pol=False, verbose=False)
  D_pix[:,2,2] = hp.alm2map(Y_inv_D_UU, nside, lmax, pol=False, verbose=False)
  D_pix[:,0,1] = hp.alm2map(Y_inv_D_IQ, nside, lmax, pol=False, verbose=False)
  D_pix[:,1,2] = hp.alm2map(Y_inv_D_QU, nside, lmax, pol=False, verbose=False)
  D_pix[:,0,2] = hp.alm2map(Y_inv_D_IU, nside, lmax, pol=False, verbose=False)
  D_pix[:,1,0] = hp.alm2map(Y_inv_D_IQ, nside, lmax, pol=False, verbose=False)
  D_pix[:,2,1] = hp.alm2map(Y_inv_D_QU, nside, lmax, pol=False, verbose=False)
  D_pix[:,2,0] = hp.alm2map(Y_inv_D_IU, nside, lmax, pol=False, verbose=False)

  return D_pix

def _mock_anisotropic_noise_covariance_generator(nside, lmax, noise_amplitude):
  """
  Generates anistropic (modulated correlated) noise realizations, corresponding to 
  a given choice of C_lm and D_x, where Cov_N = sqrt(D_x) C_lm sqrt(D_x)
  *** For mock test purposes only ***
  N = number of MC realizations, lmax & nside -> usual definitions
  """
  npix = hp.nside2npix(nside)
  # Average noise amplitude per pixel (in muK) - Cov_D_pix is dimensionless (keeping units in C_harmonic)
  sigma_pixel = noise_amplitude
  # knee frequency (ell) -> Integer
  f_knee = 10
  # Normalization of YY_dag -> Consistent with Dante convention
  norm_coeff = npix/(4*np.pi)

  D_pix = construct_input_D_pix(nside, lmax, norm_coeff)

  print("Noise amplitude is %f" % noise_amplitude)

  C_harmonic = np.zeros((lmax+1,3,3))

  # Construct C_harmonic -> diagonal in spherical harmonic space
  alpha_knee = 1.5
  for k in range(lmax + 1):
      for a in range(3):
          C_harmonic[k,a,a] = (sigma_pixel**2/norm_coeff)*( 1 + (f_knee/(k+1))**alpha_knee )

  print("Generating noise realization")
  # Manual generation below
  '''
  # Generate noise realization via a Cholesky decomposition 
  alm_T = np.zeros(hp.Alm.getsize(lmax), dtype='complex128')
  alm_E = np.zeros(hp.Alm.getsize(lmax), dtype='complex128')
  alm_B = np.zeros(hp.Alm.getsize(lmax), dtype='complex128')
  for l in range(lmax + 1):
    C          = np.linalg.cholesky(C_harmonic[l,:,:])
    idx        = hp.Alm.getidx(lmax, l, 0)
    vec        = np.dot(C, np.random.randn(3))
    alm_T[idx] = vec[0]
    alm_E[idx] = vec[1]
    alm_B[idx] = vec[2]
    for m in range(1, l+1):
      idx        = hp.Alm.getidx(lmax, l, m)
      vec        = np.dot(C, (np.random.randn(3) + 1.0j*np.random.randn(3))/np.sqrt(2.0))
      alm_T[idx] = vec[0]
      alm_E[idx] = vec[1]
      alm_B[idx] = vec[2]
  sqrt_C_noise_alms = np.array([alm_T, alm_E, alm_B])
  '''
  cl      = np.zeros([6, lmax+1])
  cl[0,:] = C_harmonic[:,0,0] #TT
  cl[1,:] = C_harmonic[:,1,1] #EE
  cl[2,:] = C_harmonic[:,2,2] #BB
  sqrt_C_noise_alms = hp.synalm(tuple(cl), new=True)

  sqrt_C_noise_pixel = hp.alm2map(tuple(sqrt_C_noise_alms), nside, lmax, pol=True, verbose=False)
  noise_sim = numba_array_manipulation_TypeA(D_pix, np.array(sqrt_C_noise_pixel))

  return noise_sim, D_pix, C_harmonic

def mock_gen(nside=128, lmax=256, masked=False, beamed=False, anisotropic_noise=False, noise_amplitude=100, EB_only=False, maskname=None, beamname=None):
  """
  Simulates a polarized CMB map
  IMPORTANT: Changing nside & lmax will respectively require rebuilding of mask & beam, respectively
  """
  print("### Generating mock CMB data ###")
  npix = hp.nside2npix(nside)

  # Read the cls from CAMB (.dat)
  # DKR -> usual format, GL -> more clever format
  cls, Cov_S = DKR_read_camb_cl("pol_data_boost_totCls.dat", lmax, EB_only)

  # Generate a polarized CMB map from CAMB spectra
  alms = hp.synalm(tuple(cls), new=True) # "new" format for cls ordering
  # Manual generation below
  '''
  ####################
  alm_T = np.zeros(hp.Alm.getsize(lmax), dtype='complex128')
  alm_E = np.zeros(hp.Alm.getsize(lmax), dtype='complex128')
  alm_B = np.zeros(hp.Alm.getsize(lmax), dtype='complex128')
  for l in range(2,lmax + 1):
    C          = np.linalg.cholesky(Cov_S[l,:,:])
    idx        = hp.Alm.getidx(lmax, l, 0)
    vec        = np.dot(C, np.random.randn(3))
    alm_T[idx] = vec[0]
    alm_E[idx] = vec[1]
    alm_B[idx] = vec[2]
    for m in range(1, l+1):
      idx        = hp.Alm.getidx(lmax, l, m)
      vec        = np.dot(C, (np.random.randn(3) + 1.0j*np.random.randn(3))/np.sqrt(2.0))
      alm_T[idx] = vec[0]
      alm_E[idx] = vec[1]
      alm_B[idx] = vec[2]
  alms = [alm_T, alm_E, alm_B]
  ####################
  '''
  simulated_map = hp.alm2map(tuple(alms), nside, lmax, pol=True, verbose=False)
  print("### True polarized CMB map generated successfully ###")
  hp.write_map("true_map.fits", simulated_map, overwrite=True)
  alms = np.array(alms)

  if beamed:
    beam = _extract_beam(lmax, beamname)
    np.savez("beam_T_P.npz", beam=beam) 
    print("### Beam saved successfully ###")
    alms = numba_almxfl_vec(alms, beam, lmax)
    beamed_simulated_map = hp.alm2map(tuple(alms), nside)
    print("### True beamed polarized CMB map generated successfully ###")
    hp.write_map("true_beamed_map.fits", beamed_simulated_map, overwrite=True)

  hp.write_alm("true_map_alms_T.fits", alms[0,:], overwrite=True)
  hp.write_alm("true_map_alms_E.fits", alms[1,:], overwrite=True)
  hp.write_alm("true_map_alms_B.fits", alms[2,:], overwrite=True)

  # Generate mock map and noise covariance
  if anisotropic_noise:
    print("### Generating anisotropic noise ###")
    noise_IQU, Cov_D_pix, Cov_C_harmonic = _mock_anisotropic_noise_covariance_generator(nside, lmax, noise_amplitude)
    # Save anisotropic noise covariance
    np.savez("anisotropic_noise_covariance.npz", Cov_D_pix=Cov_D_pix, Cov_C_harmonic=Cov_C_harmonic) 
  else:
    print("### Generating white noise ###")
    noise_IQU, Cov_N = _mock_noise_covariance_generator(npix, noise_amplitude, EB_only)
    # Save white noise covariance
    np.savez("noise_covariance.npz", Cov_N=Cov_N) 
  if beamed:
    mock_map = beamed_simulated_map + noise_IQU
    print("### Mock beamed polarized CMB map generated successfully ###")
  else:
    mock_map = simulated_map + noise_IQU
    print("### Mock polarized CMB map generated successfully ###")

  # Save simulated map
  hp.write_map("mock_map.fits", mock_map, overwrite=True)

  print("### Mock data saved successfully ###")

  ### We build & save the mask here but load it in the observations module
  if masked:
    mask = _build_mask(npix, nside, lmax, maskname)
    hp.write_map("mask_T_P.fits", mask, overwrite=True)
    print("### Mask saved successfully ###")

def CR_reference_gen(nside, lmax, beam=None):
  """
  Generates a reference CMB map and data for (constrained realizations) CR purposes (stand-alone function)
  """
  npix = hp.nside2npix(nside)
  print("### Generating reference maps for CR purposes ###")
  cls, _ = DKR_read_camb_cl("pol_data_boost_totCls.dat", lmax)  

  alms_ref = hp.synalm(tuple(cls), new=True)
  alms_ref_unbeamed = np.array(alms_ref).copy()
  if beam is not None:
    alms_ref = numba_almxfl_vec(np.array(alms_ref), beam, lmax)
    print("### Beam included ###")

  reference_signal = hp.alm2map(alms_ref, nside)
  print("### Reference (prior) polarized CMB map generated successfully ###")

  # Generate reference data
  noise_IQU, _ = _mock_noise_covariance_generator(npix)
  reference_data = reference_signal + noise_IQU
  print("### Reference CMB data generated successfully ###")

  return alms_ref_unbeamed, reference_signal, reference_data

def simulate_data_fixed_input_alms(input_map="true_map.fits", new_noise_amplitude=500, nside=256, lmax=256):
  """
  This is a stand-alone function that anchors the input alm's and generates a mock data set for noise level desired. 
  NOTE: This is applicable only if nside/lmax are also fixed. 
  """
  npix = hp.nside2npix(nside)
  # Generate mock map and noise covariance
  noise_IQU, Cov_N = _mock_noise_covariance_generator(npix, new_noise_amplitude)
  simulated_map = hp.read_map(input_map, field=(0,1,2))
  mock_map = simulated_map + noise_IQU
  print("### Mock polarized CMB map (for fixed alms) generated successfully ###")

  # Save simulated map and noise covariances
  hp.write_map("mock_map_test.fits", mock_map)
  np.savez("noise_covariance_test.npz", Cov_N=Cov_N) 
  print("### Mock data (for fixed alms) saved successfully ###")

def simulate_anisotropic_data_fixed_input_alms(input_map="true_map.fits", new_noise_amplitude=500, nside=256, lmax=256):
  """
  [ANISOTROPIC NOISE COVARIANCE]
  This is a stand-alone function that anchors the input alm's and generates a mock data set for noise level desired. 
  NOTE: This is applicable only if nside/lmax are also fixed. 
  """
  npix = hp.nside2npix(nside)
  # Generate mock map and noise covariance
  noise_IQU, Cov_D_pix, Cov_C_harmonic = _mock_anisotropic_noise_covariance_generator(nside, lmax, new_noise_amplitude)
  simulated_map = hp.read_map(input_map, field=(0,1,2))
  mock_map = simulated_map + noise_IQU
  print("### Mock polarized CMB map (for fixed alms) generated successfully ###")

  # Save simulated map and noise covariances
  hp.write_map("mock_map_test.fits", mock_map, overwrite=True)
  np.savez("anisotropic_noise_covariance_test.npz", Cov_D_pix=Cov_D_pix, Cov_C_harmonic=Cov_C_harmonic)
  print("### Mock data (for fixed alms) saved successfully ###")
