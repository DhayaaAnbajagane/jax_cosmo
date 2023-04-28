# This module defines kernel functions for various tracers

from functools import partial
from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as np
from jax import jit
from jax import lax
from jax import vmap

import jax_cosmo.background as bkgrd
import jax_cosmo.constants as const
import jax_cosmo.power as power
import jax_cosmo.transfer as tklib
import jax_cosmo.bias as tracer_bias
from jax_cosmo.scipy.integrate import simps
from jax_cosmo.utils import a2z
from jax_cosmo.utils import z2a
import jax_cosmo.redshift as rds

from jax.experimental.ode import odeint
from jax.tree_util import register_pytree_node_class

from jax_cosmo.probes import NumberCounts, WeakLensing, weak_lensing_kernel

from jax_cosmo.angular_cl import _get_cl_ordering, _get_cov_blocks_ordering


__all__ = ["WeakLensing", "NumberCounts"]


@register_pytree_node_class
class Cosmology_fNL:
    def __init__(self, Omega_c, Omega_b, h, n_s, sigma8, Omega_k, w0, wa,
                 fNL = 0, A_lin = 0, A_log = 0, w_lin = 0, w_log = 0, phi_lin = 0, phi_log = 0, kstar = 0.005):
        """
        Cosmology object, stores primary and derived cosmological parameters.
        Parameters:
        -----------
        Omega_c, float
          Cold dark matter density fraction.
        Omega_b, float
          Baryonic matter density fraction.
        h, float
          Hubble constant divided by 100 km/s/Mpc; unitless.
        n_s, float
          Primordial scalar perturbation spectral index.
        sigma8, float
          Variance of matter density perturbations at an 8 Mpc/h scale
        Omega_k, float
          Curvature density fraction.
        w0, float
          First order term of dark energy equation
        wa, float
          Second order term of dark energy equation of state
        gamma: float
          Index of the growth rate (optional)
        fNL: float
          Amplitude of local-type fNL
        A_lin: float
          Amplitude of linear oscillations in Power spectrum
        A_log: float
          Amplitude of log oscillations in Power spectrum
        k_star: float
          Normalization used for log oscillations. Degenerate with Amplitude and phase
        phi_lin: float
          Phase of the linear oscillations
        phi_log: float
          Phase of the log oscillations

        Notes:
        ------
        If `gamma` is specified, the emprical characterisation of growth in
        terms of  dlnD/dlna = \omega^\gamma will be used to define growth throughout.
        Otherwise the linear growth factor and growth rate will be solved by ODE.
        """
        # Store primary parameters
        self._Omega_c = Omega_c
        self._Omega_b = Omega_b
        self._h = h
        self._n_s = n_s
        self._sigma8 = sigma8
        self._Omega_k = Omega_k
        self._w0 = w0
        self._wa = wa

        self._flags = {}
        self._flags["gamma_growth"] = False

        # Secondary optional parameters
        self._gamma   = None #We will never use gamma so this is fine
        self._fNL     = fNL
        self._A_lin   = A_lin
        self._A_log   = A_log
        self._w_lin   = w_lin
        self._w_log   = w_log
        self._phi_lin = phi_lin
        self._phi_log = phi_log
        self._kstar   = kstar

        # Create a workspace where functions can store some precomputed
        # results
        self._workspace = {}

    def __str__(self):
        return (
            "Cosmological parameters: \n"
            + "    h:        "
            + str(self.h)
            + " \n"
            + "    Omega_b:  "
            + str(self.Omega_b)
            + " \n"
            + "    Omega_c:  "
            + str(self.Omega_c)
            + " \n"
            + "    Omega_k:  "
            + str(self.Omega_k)
            + " \n"
            + "    w0:       "
            + str(self.w0)
            + " \n"
            + "    wa:       "
            + str(self.wa)
            + " \n"
            + "    n:        "
            + str(self.n_s)
            + " \n"
            + "    sigma8:   "
            + str(self.sigma8)
        )

    def __repr__(self):
        return self.__str__()

    # Operations for flattening/unflattening representation
    def tree_flatten(self):
        params = (
            self._Omega_c,
            self._Omega_b,
            self._h,
            self._n_s,
            self._sigma8,
            self._Omega_k,
            self._w0,
            self._wa,
        )

        params += (self._fNL,
                   self._A_lin,
                   self._A_log,
                   self._w_lin,
                   self._w_log,
                   self._phi_lin,
                   self._phi_log,
                   self._kstar)

        return (
            params,
            self._flags,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # Retrieve base parameters

        Omega_c, Omega_b, h, n_s, sigma8, Omega_k, w0, wa = children[:8]
        children = list(children[8:])

        # We extract the remaining parameters in reverse order from how they
        # were inserted

        kstar = children.pop()
        phi_log = children.pop()
        phi_lin = children.pop()
        w_log = children.pop()
        w_lin = children.pop()
        A_log = children.pop()
        A_lin = children.pop()
        fNL = children.pop()

        return cls(
            Omega_c=Omega_c,
            Omega_b=Omega_b,
            h=h,
            n_s=n_s,
            sigma8=sigma8,
            Omega_k=Omega_k,
            w0=w0,
            wa=wa,
            gamma=gamma,
            fNL=fNL,
            A_lin = A_lin,
            A_log = A_log,
            w_lin = w_lin,
            w_log = w_log,
            phi_lin = phi_lin,
            phi_log = phi_log,
            kstar = kstar
        )

    # Cosmological parameters, base and derived
    @property
    def Omega(self):
        return 1.0 - self._Omega_k

    @property
    def Omega_b(self):
        return self._Omega_b

    @property
    def Omega_c(self):
        return self._Omega_c

    @property
    def Omega_m(self):
        return self._Omega_b + self._Omega_c

    @property
    def Omega_de(self):
        return self.Omega - self.Omega_m

    @property
    def Omega_k(self):
        return self._Omega_k

    @property
    def k(self):
        return -np.sign(self._Omega_k).astype(np.int8)

    @property
    def sqrtk(self):
        return np.sqrt(np.abs(self._Omega_k))

    @property
    def h(self):
        return self._h

    @property
    def w0(self):
        return self._w0

    @property
    def wa(self):
        return self._wa

    @property
    def n_s(self):
        return self._n_s

    @property
    def sigma8(self):
        return self._sigma8

    @property
    def gamma(self):
        return self._gamma

    @property
    def fNL(self):
        return self._fNL

    @property
    def fNL(self):
        return self._fNL

    @property
    def A_lin(self):
        return self._A_lin

    @property
    def A_log(self):
        return self._A_log

    @property
    def w_lin(self):
        return self._w_lin

    @property
    def w_log(self):
        return self._w_log

    @property
    def phi_lin(self):
        return self._phi_lin

    @property
    def phi_log(self):
        return self._phi_log

    @property
    def kstar(self):
        return self._kstar




@jit
def density_kernel_nobias(cosmo, pzs, bias, z, ell):
    """
    Computes the number counts density kernel
    """
    x = [isinstance(pz, rds.delta_nz) for pz in pzs]
    if any(x):
        raise NotImplementedError(
            "Density kernel not properly implemented for delta redshift distributions"
        )
    # stack the dndz of all redshift bins
    dndz = np.stack([pz(z) for pz in pzs], axis=0)

    #DHAYAA: I REMOVED THE BIAS FROM HERE
    radial_kernel = dndz * bkgrd.H(cosmo, z2a(z))

    # Normalization,
    constant_factor = 1.0
    # Ell dependent factor
    ell_factor = 1.0
    return constant_factor * ell_factor * radial_kernel


@register_pytree_node_class
class NumberCounts_fNL(NumberCounts):
    """Class representing a galaxy clustering probe, with a bunch of bins

    DHAYAA: editing this so the bias isn't applied to the kernel here

    Parameters:
    -----------
    redshift_bins: nzredshift distributions
    Configuration:
    --------------
    has_rsd....
    """

    def kernel(self, cosmo, z, ell):
        """Compute the radial kernel for all nz bins in this probe.
        Returns:
        --------
        radial_kernel: shape (nbins, nz)
        """
        z = np.atleast_1d(z)
        # Extract parameters
        pzs, bias = self.params
        # Retrieve density kernel
        kernel = density_kernel_nobias(cosmo, pzs, bias, z, ell)
        return kernel

@register_pytree_node_class
class WeakLensing_fNL(WeakLensing):
    """
    Class representing a weak lensing probe, with a bunch of bins

    Parameters:
    -----------
    redshift_bins: list of nzredshift distributions
    ia_bias: (optional) if provided, IA will be added with the NLA model,
    either a single bias object or a list of same size as nzs
    multiplicative_bias: (optional) adds an (1+m) multiplicative bias, either single
    value or list of same length as redshift bins

    Configuration:
    --------------
    sigma_e: intrinsic galaxy ellipticity
    """

    def __init__(
        self,
        redshift_bins,
        ia_bias=None,
        multiplicative_bias=0.0,
        sigma_e=0.26,
        **kwargs
    ):
        bias = tracer_bias.constant_linear_bias(1) #This is done to monkeypatch angular_Cl_fNL code and include WL tracer

        # Depending on the Configuration we will trace or not the ia_bias in the
        # container
        if ia_bias is None:
            ia_enabled = False
            args = (redshift_bins, bias, multiplicative_bias)
        else:
            ia_enabled = True
            args = (redshift_bins, bias, multiplicative_bias, ia_bias)
        if "ia_enabled" not in kwargs.keys():
            kwargs["ia_enabled"] = ia_enabled

        kwargs["sigma_e"] = sigma_e

        self.params = args
        self.config = kwargs

    def kernel(self, cosmo, z, ell):
        """
        Compute the radial kernel for all nz bins in this probe.

        Returns:
        --------
        radial_kernel: shape (nbins, nz)
        """
        z = np.atleast_1d(z)
        # Extract parameters
        pzs, b, m = self.params[:3]
        kernel = weak_lensing_kernel(cosmo, pzs, z, ell)
        # If IA is enabled, we add the IA kernel
        if self.config["ia_enabled"]:
            bias = self.params[3]
            kernel += nla_kernel(cosmo, pzs, bias, z, ell)
        # Applies measurement systematics
        if isinstance(m, list):
            m = np.expand_dims(np.stack([mi for mi in m], axis=0), 1)
        kernel *= 1.0 + m
        return kernel

def angular_cl_fNL(
    cosmo, ell, probes, transfer_fn=tklib.Eisenstein_Hu, nonlinear_fn=power.halofit):
    """
    Computes angular Cls for the provided probes

    All using the Limber approximation

    Returns
    -------

    cls: [ell, ncls]
    """
    # Retrieve the maximum redshift probed
    zmax = max([p.zmax for p in probes])

    # We define a function that computes a single l, and vectorize it
    @partial(vmap, out_axes=1)
    def cl(ell):
        def integrand(a):
            # Step 1: retrieve the associated comoving distance
            chi = bkgrd.radial_comoving_distance(cosmo, a)
            z   = 1/a - 1

            # Step 2: get the power spectrum for this combination of chi and a
            k = (ell + 0.5) / np.clip(chi, 1.0)

            # pk should have shape [na]
            pk = power.nonlinear_matter_power(cosmo, k, a, transfer_fn, nonlinear_fn)

            # DHAYAA: I compute the bias separately, so that we include fNL
            bias = [p.params[1] for p in probes]

            delta_c = 1.686

            t = transfer_fn(cosmo, k)
            g = bkgrd.growth_factor(cosmo, a)

            R_H = 3e8 / (cosmo.h*100 * 1e3 / 3.086e+22)
            R_H = R_H / 3.086e+22

            cosmo_fac = cosmo.Omega_m * delta_c #Factor with LCDM cosmo params
            lin_bias  = [b(cosmo, z) for b in bias] #Standard linear bias
            fnl_bias  = [3*(b(cosmo, z) - 1) / k**2 / R_H**2 * cosmo.fNL / t / g * cosmo_fac for b in bias] #fNL bias

            # raise ValueError(fnl_bias, 1/k**2, cosmo.fNL, t, g, cosmo_fac, bias[0](cosmo, z))

            bias = np.vstack([b_l + b_f for b_l, b_f in zip(lin_bias, fnl_bias)]) #Final, total bias

            # Compute the kernels for all probes
            # DHAYAA: I added the bias back here. This is my last edit in this function
            kernels = np.vstack([p.kernel(cosmo, a2z(a), ell)*b for p, b in zip(probes, bias)])

            # Define an ordering for the blocks of the signal vector
            cl_index = np.array(_get_cl_ordering(probes))

            # Compute all combinations of tracers
            def combine_kernels(inds):
                return kernels[inds[0]] * kernels[inds[1]]

            # Now kernels has shape [ncls, na]
            kernels = lax.map(combine_kernels, cl_index)

            result = pk * kernels * bkgrd.dchioverda(cosmo, a) / np.clip(chi**2, 1.0)

            # We transpose the result just to make sure that na is first
            return result.T

        return simps(integrand, z2a(zmax), 1.0, 512) / const.c**2

    return cl(ell)
