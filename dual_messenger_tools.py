import numpy as np
import healpy as hp
import numba as nb
from numba import vectorize, int64, float64, complex128, jit #, njit, prange

__all__ = ["new_s", "_sum", "_numba_sum", "DKR_read_camb_cl", "DKR_read_camb_cl_pure_E", "DKR_read_camb_cl_pure_B", "DKR_read_pure_EB_camb_cl", "DKR_residual_error_read_pure_EB_camb_cl","compute_inv_2x2", "numba_almxfl", "numba_almxfl_vec", "numba_almxfl_block", "numba_array_addition", "numba_array_subtraction", "numba_array_multiplication", "numba_array_manipulation_TypeA", "numba_array_manipulation_TypeB", "numba_array_manipulation_TypeA_mod", "compute_s_coeff", "compute_s_coeff_beam", "compute_s_coeff_beam_pure", "numba_array_sub", "numba_array_manipulation_TypeD", "jacobi_corrector", "jacobi_corrector_anisotropic", "jacobi_corrector_anisotropic_pure", "generic_jacobi_corrector", "generic_jacobi_corrector_GL", "generic_jacobi_corrector_DKR", "compute_inv_S_bar_plus_v", "compute_threshold_step", "compute_threshold_step_pure_E", "numba_array_multiplication_harmonic", "numba_array_addition_harmonic"] # "numba_array_manipulation_TypeC"

def new_s(lmax, npix):
  """
  Generate an empty alm 
  """
  s_pix = np.zeros((3,npix))  
  return hp.map2alm(s_pix, lmax, pol=True, use_pixel_weights=True)

def _sum(l):
  """
  Sum all elements of a 1d array
  """
  a = l[0]
  for e in l[1:]:
    a += e
  return a

def compute_threshold_step(precision, zeta, inv_norm_coeff_times_T, cooling_step):
  """
  Compute the number of steps required to bring relaxed epsilon to final desired precision
  """
  inv_coeff_T = inv_norm_coeff_times_T.copy()
  z = zeta.copy()
  i_eps = 0
  while z[0] > inv_coeff_T[0]: # Since temperature is the dominant component
    z[0] /= cooling_step
    i_eps += 1
  return (10**-4/precision)**(1./i_eps)

def compute_threshold_step_pure_E(precision, zeta, inv_norm_coeff_times_T, cooling_step):
  """
  Compute the number of steps required to bring relaxed epsilon to final desired precision
  """
  inv_coeff_T = inv_norm_coeff_times_T.copy()
  z = zeta.copy()
  i_eps = 0
  while z[1] > inv_coeff_T[1]: # Since temperature is the dominant component
    z[1] /= cooling_step
    i_eps += 1
  return (10**-4/precision)**(1./i_eps)

@jit(float64(float64[:]), nopython=True)
def _numba_sum(l):
  """
  Sum all elements of a 1d array using numba
  """
  a = l[0]
  for e in l[1:]:
    a += e
  return a

def GL_read_camb_cl(fname, lmax):
  """
  Read a .dat file from CAMB and return the cls
  """
  f = np.float64
  data_type = [('C_TT',f),('C_EE',f),('C_BB',f),('C_TE',f)]
 
  a = np.genfromtxt(fname, dtype=([('l','i')] + data_type))
   
  # Preferable to set lmax elsewhere for consistency
  #lmax = a['l'].max()
   
  cls = np.zeros(lmax+1, dtype=data_type)
   
  for n,_ in data_type:
    l = a['l']
    wl = l*(l+1)/(2*np.pi)
    # Save the cls as a columned-numpy array, e.g. cls['C_TE'] to access C_TE
    cls[n][a['l']] = a[n]/wl
     
  return cls

def DKR_read_camb_cl(fname, lmax):
  """
  Read a .dat file from CAMB and return the cls
  cls -> different format to GL version
  """
  camb_data = np.loadtxt(fname)
  llp1     = np.arange(lmax + 1)*(1+np.arange(lmax + 1)) / (2.0*np.pi)
  cls       = np.zeros([6, lmax+1])
  cls[0,2:] = camb_data[:lmax-1,1] / llp1[2:] #TT
  cls[1,2:] = camb_data[:lmax-1,2] / llp1[2:] #EE
  cls[2,2:] = camb_data[:lmax-1,3] / llp1[2:] #BB
  cls[3,2:] = camb_data[:lmax-1,4] / llp1[2:] #TE

  # Construct Cov_S (correct array structure)
  Cov_S = np.zeros([lmax + 1, 3, 3])

  for ell in range(lmax + 1):
    Cov_S[ell,0,0] = cls[0,ell]
    Cov_S[ell,1,1] = cls[1,ell]
    Cov_S[ell,2,2] = cls[2,ell]
    Cov_S[ell,0,1] = cls[3,ell]
    Cov_S[ell,1,0] = cls[3,ell]
     
  return cls, Cov_S

def DKR_read_camb_cl_pure_E(fname, lmax):
  """
  Read a .dat file from CAMB and return the cls
  cls -> different format to GL version
  """
  camb_data = np.loadtxt(fname)
  llp1     = np.arange(lmax + 1)*(1+np.arange(lmax + 1)) / (2.0*np.pi)
  cls       = np.zeros([6, lmax+1])
  cls[0,2:] = camb_data[:lmax-1,1] / llp1[2:] #TT
  cls[1,2:] = camb_data[:lmax-1,2] / llp1[2:] #EE
  cls[2,2:] = camb_data[:lmax-1,3] / llp1[2:] #BB
  cls[3,2:] = camb_data[:lmax-1,4] / llp1[2:] #TE

  # Construct Cov_S (correct array structure)
  Cov_S = np.zeros([lmax + 1, 3, 3])

  for ell in range(lmax + 1):
    Cov_S[ell,0,0] = 10**16*cls[0,ell]
    Cov_S[ell,1,1] = cls[1,ell]
    Cov_S[ell,2,2] = 10**16*cls[2,ell]
    Cov_S[ell,0,1] = np.sqrt(10**16)*cls[3,ell]
    Cov_S[ell,1,0] = np.sqrt(10**16)*cls[3,ell]
     
  return cls, Cov_S

def DKR_read_camb_cl_pure_B(fname, lmax):
  """
  Read a .dat file from CAMB and return the cls
  cls -> different format to GL version
  """
  camb_data = np.loadtxt(fname)
  llp1     = np.arange(lmax + 1)*(1+np.arange(lmax + 1)) / (2.0*np.pi)
  cls       = np.zeros([6, lmax+1])
  cls[0,2:] = camb_data[:lmax-1,1] / llp1[2:] #TT
  cls[1,2:] = camb_data[:lmax-1,2] / llp1[2:] #EE
  cls[2,2:] = camb_data[:lmax-1,3] / llp1[2:] #BB
  cls[3,2:] = camb_data[:lmax-1,4] / llp1[2:] #TE

  # Construct Cov_S (correct array structure)
  Cov_S = np.zeros([lmax + 1, 3, 3])

  for ell in range(lmax + 1):
    Cov_S[ell,0,0] = cls[0,ell]
    Cov_S[ell,1,1] = 10**16*cls[1,ell]
    Cov_S[ell,2,2] = cls[2,ell]
    Cov_S[ell,0,1] = np.sqrt(10**16)*cls[3,ell]
    Cov_S[ell,1,0] = np.sqrt(10**16)*cls[3,ell]
     
  return cls, Cov_S

def DKR_read_pure_EB_camb_cl(fname, lmax, pure_filter):
  """
  Read a .dat file from CAMB and return the cls
  cls -> different format to GL version
  For pure E/B prescription, Cov_S -> fully diagonal
  """
  camb_data = np.loadtxt(fname)
  llp1     = np.arange(lmax + 1)*(1+np.arange(lmax + 1)) / (2.0*np.pi)
  cls       = np.zeros([6, lmax+1])
  cls[0,2:] = camb_data[:lmax-1,1] / llp1[2:] #TT
  cls[1,2:] = camb_data[:lmax-1,2] / llp1[2:] #EE
  cls[2,2:] = camb_data[:lmax-1,3] / llp1[2:] #BB
  cls[3,2:] = camb_data[:lmax-1,4] / llp1[2:] #TE

  # Construct Cov_S (correct array structure)
  Cov_S = np.zeros([lmax + 1, 3])

  if pure_filter == 'pure_B':
    for ell in range(2, lmax + 1):
      Cov_S[ell,0] = cls[0,ell] - cls[3,ell]**2/cls[1,ell]
      Cov_S[ell,1] = 0.
      Cov_S[ell,2] = cls[2,ell]

  if pure_filter == 'pure_E':
    for ell in range(2, lmax + 1):
      Cov_S[ell,0] = 0
      Cov_S[ell,1] = cls[1,ell] - cls[3,ell]**2/cls[0,ell]
      Cov_S[ell,2] = 0

  return cls, Cov_S

def DKR_residual_error_read_pure_EB_camb_cl(fname, lmax, pure_filter):
  """
  Read a .dat file from CAMB and return the cls
  cls -> different format to GL version
  For pure E/B prescription, Cov_S -> fully diagonal
  """
  camb_data = np.loadtxt(fname)
  llp1     = np.arange(lmax + 1)*(1+np.arange(lmax + 1)) / (2.0*np.pi)
  cls       = np.zeros([6, lmax+1])
  cls[0,2:] = camb_data[:lmax-1,1] / llp1[2:] #TT
  cls[1,2:] = camb_data[:lmax-1,2] / llp1[2:] #EE
  cls[2,2:] = camb_data[:lmax-1,3] / llp1[2:] #BB
  cls[3,2:] = camb_data[:lmax-1,4] / llp1[2:] #TE

  # Construct Cov_S (correct array structure)
  Cov_S = np.zeros([lmax + 1, 3])

  if pure_filter == 'pure_B':
    for ell in range(2, lmax + 1):
      Cov_S[ell,0] = cls[0,ell] - cls[3,ell]**2/cls[1,ell]
      Cov_S[ell,1] = 10**15
      Cov_S[ell,2] = cls[2,ell]

  if pure_filter == 'pure_E':
    for ell in range(2, lmax + 1):
      Cov_S[ell,0] = 10**15
      Cov_S[ell,1] = cls[1,ell] - cls[3,ell]**2/cls[0,ell]
      Cov_S[ell,2] = 10**15

  return cls, Cov_S

@jit(nopython=True, cache=True)
def numba_almxfl(a, f, lmax):
  """
  Use numba to do a similar operation to almxfl
  """
  j = 0
  for m in range (1 + lmax):
    a[j:(j + 1 + lmax - m)] *= f[m:(lmax + 1)]
    j += 1 + lmax - m
  return a

@jit(nopython=True, cache=True)
def numba_almxfl_vec(a, f, lmax):
  """
  Similar to numba_almxfl, except that here multiplication by (3)-element vectors
  Shapes: a -> [3, mmax(2lmax + 1 - mmax)//2 + lmax + 1], f -> [lmax+1, 3]
  """
  j = 0
  z = np.zeros(a.shape, dtype=np.complex128)
  for m in range (1 + lmax):
    for k in range(3):
      z[k,j:(j + 1 + lmax - m)] = a[k,j:(j + 1 + lmax - m)]*f[m:(lmax + 1),k]
    j += 1 + lmax - m

  return z

@jit(nopython=True, cache=True) 
def numba_almxfl_block(a, f, lmax):
  """
  Similar to numba_almxfl, except that here multiplication by blocks of (3,3)
  Shapes: f -> [lmax+1,3,3], a -> [3, mmax(2lmax + 1 - mmax)//2 + lmax + 1]
  """
  j = 0
  z = np.zeros(a.shape, dtype=np.complex128)
  for m in range (1 + lmax):
    z[0,j:(j + 1 + lmax - m)] = a[0,j:(j + 1 + lmax - m)]*f[m:(lmax + 1),0,0] + a[1,j:(j + 1 + lmax - m)]*f[m:(lmax + 1),0,1] + a[2,j:(j + 1 + lmax - m)]*f[m:(lmax + 1),0,2]
    z[1,j:(j + 1 + lmax - m)] = a[0,j:(j + 1 + lmax - m)]*f[m:(lmax + 1),1,0] + a[1,j:(j + 1 + lmax - m)]*f[m:(lmax + 1),1,1] + a[2,j:(j + 1 + lmax - m)]*f[m:(lmax + 1),1,2]
    z[2,j:(j + 1 + lmax - m)] = a[0,j:(j + 1 + lmax - m)]*f[m:(lmax + 1),2,0] + a[1,j:(j + 1 + lmax - m)]*f[m:(lmax + 1),2,1] + a[2,j:(j + 1 + lmax - m)]*f[m:(lmax + 1),2,2]
    j += 1 + lmax - m

  return z

@vectorize([float64(float64, float64)], nopython=True, target='parallel')
def numba_array_addition(x, y):
  """
  Use numba to do array addition (x + y) where both x and y have same shape
  """
  return x + y

@vectorize([complex128(complex128, complex128)], nopython=True, target='parallel')
def numba_array_addition_harmonic(x, y):
  """
  Use numba to do array addition (x + y) where both x and y have same shape
  """
  return x + y

@vectorize([float64(float64, float64)], nopython=True, target='parallel')
def numba_array_subtraction(x, y):
  """
  Use numba to do array subtraction (x - y) where both x and y have same shape
  """
  return x - y

@jit(nopython=True)
def numba_array_sub(x, y):
  """
  Use numba to do array subtraction (x - y) where x of shape [3] and y of shape [3,npix]
  """
  z = np.zeros(y.shape)
  for k in range(3):
    z[k,:] = x[k] - y[k,:]

  return z

@vectorize([float64(float64, float64)], nopython=True, target='parallel')
def numba_array_multiplication(x, y):
  """
  Use numba to do array multiplication (x*y) where both x and y have same shape
  """
  return x*y

@vectorize([complex128(complex128, complex128)], nopython=True, target='parallel')
def numba_array_multiplication_harmonic(x, y):
  """
  Use numba to do array multiplication (x*y) where both x and y have same shape
  """
  return x*y

@jit(nopython=True, cache=True)
def numba_array_manipulation_TypeA(x, y):
  """
  Use numba to do array multiplication (x*y) where
  x of shape [npix or (lmax+1),3,3] and y of shape [3,npix or (lmax+1)]
  """
  z = np.zeros(y.shape)
  for k in range(3):
    z[k,:] = x[:,k,0]*y[0,:] + x[:,k,1]*y[1,:] + x[:,k,2]*y[2,:]
  return z

@jit(nopython=True)
def numba_array_manipulation_TypeB(x, y):
  """
  Use numba to do compute the following: 1/(x+y) where
  x of shape [3] and y of shape[3,npix]
  """ 
  z = np.zeros(y.shape)
  for k in range(3):
    z[k,:] = 1./(x[k] + y[k,:])
  return z

@jit(nopython=True, cache=True)
def numba_array_manipulation_TypeC(a, b, c, d, e):
  """
  Use numba to compute the following: A*[B#(C*D) + E], where # is TypeA manipulation
  Shapes are as follows:
  a, d, e -> [3,npix], b -> [npix,3,3], c -> [3]
  Shape of output -> [3,npix], same as a and e
  """
  z = np.zeros(e.shape)
  for k in range(3):
    z[k,:] = a[k,:]*( (b[:,k,0]*(c[0]*d[0,:]) + b[:,k,1]*(c[1]*d[1,:]) + b[:,k,2]*(c[2]*d[2,:])) + e[k,:] )

  return z

@jit(nopython=True, cache=True)
def numba_array_manipulation_TypeD(a, b, c, d, e): 
  """
  Use numba to compute the following: A#[B*C + D#E], where # is TypeA manipulation
  Shapes are as follows:
  c, e -> [3,npix], a, d -> [npix,3,3], b -> [3] or constant
  Shape of output -> [3,npix], same as c and e
  """
  z = np.zeros(e.shape)
  for k in range(3):
    z[k,:] = (d[:,k,0]*e[0,:] + d[:,k,1]*e[1,:] + d[:,k,2]*e[2,:]) + (b*c[k,:]) # TODO: Can we include the remaining operation A# too?
  y = numba_array_manipulation_TypeA(a, z)  
  return y

@jit(nopython=True, cache=True)
def numba_array_manipulation_TypeA_mod(x, y):
  """
  Similar to TypeA, but y has a different shape
  Use numba to do array multiplication (x*y) where
  x of shape [npix or (lmax+1),3,3] and y of shape [npix or (lmax+1), 3]
  """
  z = np.zeros(y.shape)
  for k in range(3):
    z[:,k] = x[:,k,0]*y[:,0] + x[:,k,1]*y[:,1] + x[:,k,2]*y[:,2]
  return z

# TODO: Can we improve the Jacobi corrector below?
### @vectorize([float64(float64, float64, float64, float64)], nopython=True, target='parallel')
@jit(cache=True) # nopython=True
def jacobi_corrector(approx_alm, lmax_in, NSIDE, operator_P, s_coeff, inv_S_bar_plus_v, alpha, inv_norm_coeff, beam_jacobi=None):
  """
  We use Jacobi relaxation to refine the solution for s, to account for non-orthogonality of SHTs
  Shapes are as follows:
  alm -> typical alm, output -> typical alm
  """
  print("Initializing Jacobi corrector")
  norm_jacobi = 1. 
  jacobi_new = approx_alm.copy()
  while norm_jacobi >= 10**-9:
    jacobi_old = jacobi_new.copy()
    P_jacobi_old = numba_almxfl_block(jacobi_old, operator_P, lmax_in)
    alpha_inv_S_bar_plus_v_P_jacobi_old = numba_almxfl_vec(P_jacobi_old, inv_S_bar_plus_v, lmax_in)*alpha*inv_norm_coeff
    if beam_jacobi is not None:
      P_jacobi_old = numba_almxfl_vec(jacobi_old, beam_jacobi, lmax_in) ### This is actually beam_jacobi_old
    Y_P_jacobi_old = hp.alm2map(tuple(P_jacobi_old), nside=NSIDE, lmax=lmax_in, pol=True, verbose=False)
    Y_dag_Y_P_jacobi_old = np.array(hp.map2alm(Y_P_jacobi_old, lmax=lmax_in, pol=True, iter=0))
    if beam_jacobi is not None:
      Y_dag_Y_P_jacobi_old = numba_almxfl_vec(Y_dag_Y_P_jacobi_old, beam_jacobi, lmax_in)
    Y_dag_Y_P_jacobi_old = numba_almxfl_block(Y_dag_Y_P_jacobi_old, operator_P, lmax_in)
    A_jacobi_old = (Y_dag_Y_P_jacobi_old + alpha_inv_S_bar_plus_v_P_jacobi_old)
    inter_jacobi_term = numba_almxfl_vec(A_jacobi_old, s_coeff, lmax_in) 
    jacobi_term = numba_almxfl_block(inter_jacobi_term, operator_P, lmax_in)
    jacobi_new = jacobi_old + (approx_alm - jacobi_term)
    norm_jacobi = np.linalg.norm(jacobi_new - jacobi_old)/np.linalg.norm(jacobi_old)

  return jacobi_new

def jacobi_corrector_anisotropic(approx_alm, lmax_in, NSIDE, operator_P, s_coeff, inv_S_bar_plus_v, alpha, norm_coeff, inv_norm_coeff, beam_jacobi=None):
  """
  We use Jacobi relaxation to refine the solution for s, to account for non-orthogonality of SHTs
  Shapes are as follows:
  alm -> typical alm, output -> typical alm
  """
  print("Initializing Jacobi corrector")
  norm_jacobi = 1.
  jacobi_new = approx_alm.copy()
  while norm_jacobi >= 10**-9:
    jacobi_old = jacobi_new.copy()
    P_jacobi_old = numba_almxfl_block(jacobi_old, operator_P, lmax_in)
    alpha_inv_S_bar_plus_v_P_jacobi_old = numba_almxfl_vec(P_jacobi_old, inv_S_bar_plus_v, lmax_in)*alpha
    if beam_jacobi is not None:
      jacobi_old = numba_almxfl_vec(jacobi_old, beam_jacobi, lmax_in) ### This is actually beam_jacobi_old
    Y_jacobi_old = hp.alm2map(tuple(jacobi_old), nside=NSIDE, lmax=lmax_in, pol=True, verbose=False)
    
    Y_jacobi_old = np.array(hp.map2alm(Y_jacobi_old, lmax=lmax_in, pol=True, iter=0))
    Y_jacobi_old = hp.alm2map(tuple(Y_jacobi_old), nside=NSIDE, lmax=lmax_in, pol=True, verbose=False)*inv_norm_coeff
    Y_jacobi_old = generic_jacobi_corrector_DKR(Y_jacobi_old, lmax_in, NSIDE, norm_coeff, inv_norm_coeff)

    Y_dag_Y_jacobi_old = np.array(hp.map2alm(Y_jacobi_old, lmax=lmax_in, pol=True, iter=0))*norm_coeff
    if beam_jacobi is not None:
      Y_dag_Y_jacobi_old = numba_almxfl_vec(Y_dag_Y_jacobi_old, beam_jacobi, lmax_in)
    P_Y_dag_Y_jacobi_old = numba_almxfl_block(Y_dag_Y_jacobi_old, operator_P, lmax_in)
    A_jacobi_old = numba_array_addition_harmonic(P_Y_dag_Y_jacobi_old, alpha_inv_S_bar_plus_v_P_jacobi_old)
    inter_jacobi_term = numba_almxfl_vec(A_jacobi_old, s_coeff, lmax_in)
    jacobi_term = numba_almxfl_block(inter_jacobi_term, operator_P, lmax_in)
    jacobi_new = jacobi_old + (approx_alm - jacobi_term)
    norm_jacobi = np.linalg.norm(jacobi_new - jacobi_old)/np.linalg.norm(jacobi_old)
    print("jacobi correction %9f" % np.abs(approx_alm - jacobi_term).max())

  return jacobi_new

def jacobi_corrector_anisotropic_pure(approx_alm, lmax_in, NSIDE, s_coeff, inv_S_bar_plus_v, alpha, norm_coeff, inv_norm_coeff, beam_jacobi=None):
  """
  We use Jacobi relaxation to refine the solution for s, to account for non-orthogonality of SHTs
  Shapes are as follows:
  alm -> typical alm, output -> typical alm
  """
  print("Initializing Jacobi corrector")
  norm_jacobi = 1.
  jacobi_new = approx_alm.copy()
  while norm_jacobi >= 10**-9:
    jacobi_old = jacobi_new.copy()
    alpha_inv_S_bar_plus_v_jacobi_old = numba_almxfl_vec(jacobi_old, inv_S_bar_plus_v, lmax_in)*alpha
    if beam_jacobi is not None:
      jacobi_old = numba_almxfl_vec(jacobi_old, beam_jacobi, lmax_in) ### This is actually beam_jacobi_old
    Y_jacobi_old = hp.alm2map(tuple(jacobi_old), nside=NSIDE, lmax=lmax_in, pol=True, verbose=False)
    
    Y_jacobi_old = np.array(hp.map2alm(Y_jacobi_old, lmax=lmax_in, pol=True, iter=0))
    Y_jacobi_old = hp.alm2map(tuple(Y_jacobi_old), nside=NSIDE, lmax=lmax_in, pol=True, verbose=False)*inv_norm_coeff
    Y_jacobi_old = generic_jacobi_corrector_DKR(Y_jacobi_old, lmax_in, NSIDE, norm_coeff, inv_norm_coeff)

    Y_dag_Y_jacobi_old = np.array(hp.map2alm(Y_jacobi_old, lmax=lmax_in, pol=True, iter=0))*norm_coeff
    if beam_jacobi is not None:
      Y_dag_Y_jacobi_old = numba_almxfl_vec(Y_dag_Y_jacobi_old, beam_jacobi, lmax_in)
    A_jacobi_old = numba_array_addition_harmonic(Y_dag_Y_jacobi_old, alpha_inv_S_bar_plus_v_jacobi_old)
    jacobi_term = numba_almxfl_vec(A_jacobi_old, s_coeff, lmax_in)
    jacobi_new = jacobi_old + (approx_alm - jacobi_term)
    norm_jacobi = np.linalg.norm(jacobi_new - jacobi_old)/np.linalg.norm(jacobi_old)
    print("jacobi correction %.2E" % np.abs(approx_alm - jacobi_term).max())

  return jacobi_new

def generic_jacobi_corrector(input_vector, C_harmonic, inv_C_harmonic, lmax_in, NSIDE):
  print("Initializing YCY^+ DKR pixel Jacobi corrector")
  norm_jacobi = 1.
  jacobi_new = input_vector.copy()
  while norm_jacobi >= 10**-9:
    jacobi_old = jacobi_new.copy()
    Y_dag_jacobi_old = np.array(hp.map2alm(jacobi_old, lmax=lmax_in, pol=True, iter=0))
    C_Y_dag_jacobi_old = numba_almxfl_vec(Y_dag_jacobi_old, C_harmonic, lmax_in)
    Y_C_Y_dag_jacobi_old = hp.alm2map(tuple(C_Y_dag_jacobi_old), nside=NSIDE, lmax=lmax_in, pol=True, verbose=False)
    Y_dag_Y_C_Y_dag_jacobi_old = np.array(hp.map2alm(Y_C_Y_dag_jacobi_old, lmax=lmax_in, pol=True, iter=0))
    inv_C_Y_dag_Y_C_Y_dag_jacobi_old = numba_almxfl_vec(Y_dag_Y_C_Y_dag_jacobi_old, inv_C_harmonic, lmax_in)
    jacobi_term = hp.alm2map(tuple(inv_C_Y_dag_Y_C_Y_dag_jacobi_old), nside=NSIDE, lmax=lmax_in, pol=True, verbose=False)
    jacobi_new = jacobi_old + (input_vector - jacobi_term)
    norm_jacobi = np.linalg.norm(jacobi_new - jacobi_old)/np.linalg.norm(jacobi_old)

  return jacobi_new

def myop(input_vector, C_harmonic, inv_C_harmonic, lmax_in, NSIDE, norm_coeff, inv_norm_coeff):

    return hp.alm2map(tuple(numba_almxfl_vec(hp.map2alm(input_vector, lmax=lmax_in, iter=0, pol=True), C_harmonic, lmax_in)), nside=NSIDE, verbose=False, pol=True)*norm_coeff

def pseudo_inv(input_vector, C_harmonic, inv_C_harmonic, lmax_in, NSIDE, norm_coeff, inv_norm_coeff):

    return hp.alm2map(tuple(numba_almxfl_vec(hp.map2alm(input_vector, lmax=lmax_in, iter=0, pol=True), inv_C_harmonic, lmax_in)), nside=NSIDE, verbose=False, pol=True)*inv_norm_coeff

def onestep(sol, b, C_harmonic, inv_C_harmonic, lmax_in, NSIDE, norm_coeff, inv_norm_coeff):

    return sol + pseudo_inv(b - myop(sol, C_harmonic, inv_C_harmonic, lmax_in, NSIDE, norm_coeff, inv_norm_coeff), C_harmonic, inv_C_harmonic, lmax_in, NSIDE, norm_coeff, inv_norm_coeff)

def generic_jacobi_corrector_GL(input_vector, C_harmonic, inv_C_harmonic, lmax_in, NSIDE, norm_coeff, inv_norm_coeff):
  """
  Jacobi relaxation to correct for YCY^+
  """
  print("Initializing YCY^+ GL pixel Jacobi corrector")
  norm_jacobi = 1.
  jacobi_old = input_vector.copy()
  jacobi_new = onestep(jacobi_old, input_vector, C_harmonic, inv_C_harmonic, lmax_in, NSIDE, norm_coeff, inv_norm_coeff)
  while norm_jacobi >= 10**-9:
    jacobi_old = jacobi_new.copy()
    jacobi_new = onestep(jacobi_old, input_vector, C_harmonic, inv_C_harmonic, lmax_in, NSIDE, norm_coeff, inv_norm_coeff)
    norm_jacobi = np.linalg.norm(jacobi_new - jacobi_old)/np.linalg.norm(jacobi_old)

  return jacobi_new

def generic_jacobi_corrector_DKR(input_vector, lmax_in, NSIDE, norm_coeff, inv_norm_coeff):
  """
  Jacobi relaxation to correct for YY^+
  """
  print("Initializing YY^+ pixel Jacobi corrector")
  norm_jacobi = 1.
  jacobi_new = input_vector.copy()
  while norm_jacobi >= 10**-9:
    jacobi_old = jacobi_new.copy()
    Y_dag_jacobi_old = np.array(hp.map2alm(jacobi_old, lmax=lmax_in, pol=True, iter=0))
    Y_Y_dag_jacobi_old = hp.alm2map(tuple(Y_dag_jacobi_old), nside=NSIDE, lmax=lmax_in, pol=True, verbose=False)
    Y_dag_Y_Y_dag_jacobi_old = np.array(hp.map2alm(Y_Y_dag_jacobi_old, lmax=lmax_in, pol=True, iter=0))
    jacobi_term = hp.alm2map(tuple(Y_dag_Y_Y_dag_jacobi_old), nside=NSIDE, lmax=lmax_in, pol=True, verbose=False)
    jacobi_new = jacobi_old + (input_vector - jacobi_term)
    norm_jacobi = np.linalg.norm(jacobi_new - jacobi_old)/np.linalg.norm(jacobi_old)

  return jacobi_new

@vectorize([float64(float64, float64, float64)], nopython=True, target='parallel')
def compute_s_coeff(S_bar, v, z):
  """
  This function is used as zeroth order for Jacobi corrector
  Use numba to compute the constant coefficient required for 's'
  Shapes are as follows:
  S_bar -> [lmax+1,3], v & z -> [3]
  output -> [lmax+1,3]
  """
  return (S_bar + v)/(S_bar + z)

#@vectorize([float64(float64, float64, float64, float64)], nopython=True, target='parallel')
@jit(cache=True)
def compute_s_coeff_beam(inv_coeff_T, S_bar_plus_v, beam_square, operator_P):
  """
  Beamed version of "compute_s_coeff"
  Shapes are as follows:
  inv_coeff_T -> [3], S_bar_plus_v & beam_square -> [lmax+1,3], operator_P -> [lmax+1,3,3]
  output -> [lmax+1,3]
  """
  P_dag_S_bar_plus_v = numba_array_manipulation_TypeA_mod(operator_P, S_bar_plus_v)
  beam_square_P_dag_S_bar_plus_v = numba_array_multiplication(P_dag_S_bar_plus_v, beam_square)
  P_beam_square_P_dag_S_bar_plus_v = numba_array_manipulation_TypeA_mod(operator_P, beam_square_P_dag_S_bar_plus_v)
  return S_bar_plus_v/(P_beam_square_P_dag_S_bar_plus_v + inv_coeff_T)

#@vectorize([float64(float64, float64, float64, float64)], nopython=True, target='parallel')
@jit(cache=True)
def compute_s_coeff_beam_pure(inv_coeff, S_bar_plus_v, beam_square):
  """
  Beamed version of "compute_s_coeff" for pure mode
  Shapes are as follows:
  inv_coeff -> [3], S_bar_plus_v & beam_square -> [lmax+1,3]
  output -> [lmax+1,3]
  """
  beam_square_S_bar_plus_v = numba_array_multiplication(S_bar_plus_v, beam_square)
  
  return S_bar_plus_v/(beam_square_S_bar_plus_v + inv_coeff)

@vectorize([float64(float64, float64)], nopython=True, target='parallel')
def compute_inv_S_bar_plus_v(S_bar, v):
  """ 
  Use numba to compute the a constant coefficient required for jacobi_corrector
  Shapes are as follows:
  S_bar -> [lmax+1,3], v -> [3]
  output -> [lmax+1,3]
  """
  return 1./(S_bar + v) 

def compute_inv_2x2(input_matrix, lmax):
  """
  Compute inverse of a 3x3 matrix (with a 2x2 block) in vectorized manner
  """
  inv_input_matrix = np.zeros([lmax+1, 3, 3])

  for g in range(lmax+1):
    a = input_matrix[g,0,0]
    b = input_matrix[g,0,1]
    c = input_matrix[g,1,1]
    d = input_matrix[g,2,2]
    denom = (a*c) - b**2
    if d == 0:
      d_mod = 0
    else:
      d_mod = 1./d
    if denom == 0:
      denom_mod = 0
    else:
      denom_mod = 1./denom
    inv_input_matrix[g,0,0] = c*denom_mod
    inv_input_matrix[g,0,1] = -b*denom_mod
    inv_input_matrix[g,1,0] = -b*denom_mod
    inv_input_matrix[g,1,1] = a*denom_mod
    inv_input_matrix[g,2,2] = d_mod

  return inv_input_matrix

