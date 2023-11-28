# Based on the `skbio.stats.composition` module, with below copyright notice.
# The original COPYING.txt file can be found under licenses/scikit-bio.txt.
# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

r"""
Composition Statistics (:mod:`composition_stats`)
=================================================

.. currentmodule:: composition_stats

This module provides functions for compositional data analysis.

Many 'omics datasets are inherently compositional - meaning that they
are best interpreted as proportions or percentages rather than
absolute counts.

Formally, :math:`x` is a composition if :math:`\sum_{i=0}^D x_{i} = c`
and :math:`x_{i} > 0`, :math:`1 \leq i \leq D` and :math:`c` is a real
valued constant and there are :math:`D` components for each
composition. In this module :math:`c=1`. Compositional data can be
analyzed using Aitchison geometry. [1]_

However, in this framework, standard real Euclidean operations such as
addition and multiplication no longer apply. Only operations such as
perturbation and power can be used to manipulate this data.

This module allows two styles of manipulation of compositional data.
Compositional data can be analyzed using perturbation and power
operations, which can be useful for simulation studies. The
alternative strategy is to transform compositional data into the real
space.  Right now, the centre log ratio transform (clr) and
the isometric log ratio transform (ilr) [2]_ can be used to accomplish
this. This transform can be useful for performing standard statistical
tools such as parametric hypothesis testing, regressions and more.

The major caveat of using this framework is dealing with zeros.  In
the Aitchison geometry, only compositions with nonzero components can
be considered. The multiplicative replacement technique [3]_ can be
used to substitute these zeros with small pseudocounts without
introducing major distortions to the data.

Functions
---------

.. autosummary::
   :toctree:

   closure
   multiplicative_replacement
   perturb
   perturb_inv
   power
   inner
   clr
   clr_inv
   ilr
   ilr_inv
   alr
   alr_inv
   center
   centralize
   sbp_basis

References
----------
.. [1] V. Pawlowsky-Glahn, J. J. Egozcue, R. Tolosana-Delgado (2015),
   Modeling and Analysis of Compositional Data, Wiley, Chichester, UK

.. [2] J. J. Egozcue.,  "Isometric Logratio Transformations for
   Compositional Data Analysis" Mathematical Geology, 35.3 (2003)

.. [3] J. A. Martin-Fernandez,  "Dealing With Zeros and Missing Values in
   Compositional Data Sets Using Nonparametric Imputation",
   Mathematical Geology, 35.3 (2003)


Examples
--------

>>> import numpy as np

Consider a very simple environment with only 3 species. The species
in the environment are equally distributed and their proportions are
equivalent:

>>> otus = np.array([1./3, 1./3., 1./3])

Suppose that an antibiotic kills off half of the population for the
first two species, but doesn't harm the third species. Then the
perturbation vector, after closure, would be as follows:

>>> antibiotic = closure(np.array([1./2, 1./2, 1]))

And the resulting perturbation would be

>>> perturb(otus, antibiotic)
array([ 0.25,  0.25,  0.5 ])

"""

import numpy as np
import scipy.stats


def closure(mat, *, out=None):
    """
    Performs closure to ensure that all elements add up to 1.

    Parameters
    ----------
    mat : array_like
       a matrix of proportions where
       rows = compositions
       columns = components
    out : array_like or None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None, a
        freshly-allocated array is returned.

    Returns
    -------
    array_like, np.float64
       A matrix of proportions where all of the values
       are nonzero and each composition (row) adds up to 1

    Raises
    ------
    ValueError
       Raises an error if any values are negative.
    ValueError
       Raises an error if the matrix has more than 2 dimension.
    ValueError
       Raises an error if there is a row that has all zeros.

    Examples
    --------
    >>> import numpy as np
    >>> from composition_stats import closure
    >>> X = np.array([[2, 2, 6], [4, 4, 2]])
    >>> closure(X)
    array([[ 0.2,  0.2,  0.6],
           [ 0.4,  0.4,  0.2]])

    """
    mat = np.atleast_2d(mat)
    if out is not None:
        out = np.atleast_2d(out)
    if np.any(mat < 0):
        raise ValueError("Cannot have negative proportions")
    if mat.ndim > 2:
        raise ValueError("Input matrix can only have two dimensions or less")
    norm = mat.sum(axis=1, keepdims=True)
    if np.any(norm == 0):
        raise ValueError("Input matrix cannot have rows with all zeros")
    return np.divide(mat, norm, out=out).squeeze()


def multiplicative_replacement(mat, delta=None):
    r"""Replace all zeros with small non-zero values

    It uses the multiplicative replacement strategy [1]_ ,
    replacing zeros with a small positive :math:`\delta`
    and ensuring that the compositions still add up to 1.


    Parameters
    ----------
    mat: array_like
       a matrix of proportions where
       rows = compositions and
       columns = components
       each composition (row) must add up to unity (see :ref:`closure()`)
    delta: float, optional
       a small number to be used to replace zeros
       If delta is not specified, then the default delta is
       :math:`\delta = \frac{1}{N^2}` where :math:`N`
       is the number of components

    Returns
    -------
    numpy.ndarray, np.float64
       A matrix of proportions where all of the values
       are nonzero and each composition (row) adds up to 1

    Raises
    ------
    ValueError
       Raises an error if negative proportions are created due to a large
       `delta`.

    Notes
    -----
    This method will result in negative proportions if a large delta is chosen.

    References
    ----------
    .. [1] J. A. Martin-Fernandez. "Dealing With Zeros and Missing Values in
           Compositional Data Sets Using Nonparametric Imputation"


    Examples
    --------
    >>> import numpy as np
    >>> from composition_stats import multiplicative_replacement
    >>> X = np.array([[.2,.4,.4, 0],[0,.5,.5,0]])
    >>> multiplicative_replacement(X)
    array([[ 0.1875,  0.375 ,  0.375 ,  0.0625],
           [ 0.0625,  0.4375,  0.4375,  0.0625]])

    """
    z_mat = (mat == 0)

    num_feats = mat.shape[-1]
    tot = z_mat.sum(axis=-1, keepdims=True)

    if delta is None:
        delta = (1. / num_feats)**2

    zcnts = 1 - tot * delta
    if np.any(zcnts) < 0:
        raise ValueError('The multiplicative replacement created negative '
                         'proportions. Consider using a smaller `delta`.')
    mat = np.where(z_mat, delta, zcnts * mat)
    return mat.squeeze()


def perturb(x, y):
    r"""
    Performs the perturbation operation.

    This operation is defined as

    .. math::
        x \oplus y = C[x_1 y_1, \ldots, x_D y_D]

    :math:`C[x]` is the closure operation defined as

    .. math::
        C[x] = \left[\frac{x_1}{\sum_{i=1}^{D} x_i},\ldots,
                     \frac{x_D}{\sum_{i=1}^{D} x_i} \right]

    for some :math:`D` dimensional real vector :math:`x` and
    :math:`D` is the number of components for every composition.

    Parameters
    ----------
    x : array_like, float
        a matrix of proportions where
        rows = compositions and
        columns = components
        each composition (row) must add up to unity (see :ref:`closure()`)
    y : array_like, float
        a matrix of proportions where
        rows = compositions and
        columns = components
        each composition (row) must add up to unity (see :ref:`closure()`)

    Returns
    -------
    numpy.ndarray, np.float64
       A matrix of proportions where all of the values
       are nonzero and each composition (row) adds up to 1

    Examples
    --------
    >>> import numpy as np
    >>> from composition_stats import perturb
    >>> x = np.array([.1,.3,.4, .2])
    >>> y = np.array([1./6,1./6,1./3,1./3])
    >>> perturb(x,y)
    array([ 0.0625,  0.1875,  0.5   ,  0.25  ])

    """
    z = np.multiply(x, y)
    return closure(z, out=z)


def perturb_inv(x, y):
    r"""
    Performs the inverse perturbation operation.

    This operation is defined as

    .. math::
        x \ominus y = C[x_1 y_1^{-1}, \ldots, x_D y_D^{-1}]

    :math:`C[x]` is the closure operation defined as

    .. math::
        C[x] = \left[\frac{x_1}{\sum_{i=1}^{D} x_i},\ldots,
                     \frac{x_D}{\sum_{i=1}^{D} x_i} \right]


    for some :math:`D` dimensional real vector :math:`x` and
    :math:`D` is the number of components for every composition.

    Parameters
    ----------
    x : array_like
        a matrix of proportions where
        rows = compositions and
        columns = components
        each composition (row) must add up to unity (see :ref:`closure()`)
    y : array_like
        a matrix of proportions where
        rows = compositions and
        columns = components
        each composition (row) must add up to unity (see :ref:`closure()`)

    Returns
    -------
    numpy.ndarray, np.float64
       A matrix of proportions where all of the values
       are nonzero and each composition (row) adds up to 1

    Examples
    --------
    >>> import numpy as np
    >>> from composition_stats import perturb_inv
    >>> x = np.array([.1,.3,.4, .2])
    >>> y = np.array([1./6,1./6,1./3,1./3])
    >>> perturb_inv(x,y)
    array([ 0.14285714,  0.42857143,  0.28571429,  0.14285714])
    """
    z = np.divide(x, y)
    return closure(z, out=z)


def power(x, a):
    r"""
    Performs the power operation.

    This operation is defined as follows

    .. math::
        `x \odot a = C[x_1^a, \ldots, x_D^a]

    :math:`C[x]` is the closure operation defined as

    .. math::
        C[x] = \left[\frac{x_1}{\sum_{i=1}^{D} x_i},\ldots,
                     \frac{x_D}{\sum_{i=1}^{D} x_i} \right]

    for some :math:`D` dimensional real vector :math:`x` and
    :math:`D` is the number of components for every composition.

    Parameters
    ----------
    x : array_like, float
        a matrix of proportions where
        rows = compositions and
        columns = components
        each composition (row) must add up to unity (see :ref:`closure()`)
    a : float
        a scalar float

    Returns
    -------
    numpy.ndarray, np.float64
       A matrix of proportions where all of the values
       are nonzero and each composition (row) adds up to 1

    Examples
    --------
    >>> import numpy as np
    >>> from composition_stats import power
    >>> x = np.array([.1,.3,.4, .2])
    >>> power(x, .1)
    array([ 0.23059566,  0.25737316,  0.26488486,  0.24714631])

    """
    y = np.power(x, a)
    return closure(y, out=y)


def inner(x, y):
    r"""
    Calculates the Aitchson inner product.

    This inner product is defined as follows

    .. math::
        \langle x, y \rangle_a =
        \frac{1}{2D} \sum\limits_{i=1}^{D} \sum\limits_{j=1}^{D}
        \ln\left(\frac{x_i}{x_j}\right) \ln\left(\frac{y_i}{y_j}\right)

    Parameters
    ----------
    x : array_like
        a matrix of proportions where
        rows = compositions and
        columns = components
        each composition (row) must add up to unity (see :ref:`closure()`)
    y : array_like
        a matrix of proportions where
        rows = compositions and
        columns = components
        each composition (row) must add up to unity (see :ref:`closure()`)

    Returns
    -------
    numpy.ndarray
         inner product result

    Examples
    --------
    >>> import numpy as np
    >>> from composition_stats import inner
    >>> x = np.array([.1, .3, .4, .2])
    >>> y = np.array([.2, .4, .2, .2])
    >>> inner(x, y)  # doctest: +ELLIPSIS
    0.2107852473...
    """
    a, b = clr(x), clr(y)
    return a.dot(b.T)


def clr(mat, ignore_zero=False):
    r"""
    Performs centre log ratio transformation.

    This function transforms compositions from Aitchison geometry to
    the real space. The :math:`clr` transform is both an isometry and an
    isomorphism defined on the following spaces

    :math:`clr: S^D \rightarrow U`

    where :math:`U=
    \{x :\sum\limits_{i=1}^D x = 0 \; \forall x \in \mathbb{R}^D\}`

    It is defined for a composition :math:`x` as follows:

    .. math::
        clr(x) = \ln\left[\frac{x_1}{g_m(x)}, \ldots, \frac{x_D}{g_m(x)}\right]

    where :math:`g_m(x) = (\prod\limits_{i=1}^{D} x_i)^{1/D}` is the geometric
    mean of :math:`x`.

    Parameters
    ----------
    mat : array_like, float
       a matrix of proportions where
       rows = compositions and
       columns = components
       each composition (row) must add up to unity (see :ref:`closure()`)
    ignore_zero : bool
        whether to ignore zeros in ``mat``. This reproduces the behavior of the
        ``compositions`` R package.

    Returns
    -------
    numpy.ndarray
         clr transformed matrix

    Examples
    --------
    >>> import numpy as np
    >>> from composition_stats import clr
    >>> x = np.array([.1, .3, .4, .2])
    >>> clr(x)
    array([-0.79451346,  0.30409883,  0.5917809 , -0.10136628])

    """
    if ignore_zero:
        msk = mat.astype(bool)
        lmat = np.log(mat, where=msk)
        gm = lmat.mean(axis=-1, keepdims=True, where=msk)
        return np.where(msk, (lmat - gm).squeeze(), lmat)
    else:
        lmat = np.log(mat)
        gm = lmat.mean(axis=-1, keepdims=True)
        return (lmat - gm).squeeze()


def clr_inv(mat):
    r"""
    Performs inverse centre log ratio transformation.

    This function transforms compositions from the real space to
    Aitchison geometry. The :math:`clr^{-1}` transform is both an isometry,
    and an isomorphism defined on the following spaces

    :math:`clr^{-1}: U \rightarrow S^D`

    where :math:`U=
    \{x :\sum\limits_{i=1}^D x = 0 \; \forall x \in \mathbb{R}^D\}`

    This transformation is defined as follows

    .. math::
        clr^{-1}(x) = C[\exp( x_1, \ldots, x_D)]

    Parameters
    ----------
    mat : array_like, float
       a matrix of real values where
       rows = transformed compositions and
       columns = components
       each composition (row) must add up to unity (see :ref:`closure()`)

    Returns
    -------
    numpy.ndarray
         inverse clr transformed matrix

    Examples
    --------
    >>> import numpy as np
    >>> from composition_stats import clr_inv
    >>> x = np.array([.1, .3, .4, .2])
    >>> clr_inv(x)
    array([ 0.21383822,  0.26118259,  0.28865141,  0.23632778])

    """
    emat = np.exp(mat)
    return closure(emat, out=emat)


def ilr(mat, basis=None, check=True):
    r"""
    Performs isometric log ratio transformation.

    This function transforms compositions from Aitchison simplex to
    the real space. The :math: ilr` transform is both an isometry,
    and an isomorphism defined on the following spaces

    :math:`ilr: S^D \rightarrow \mathbb{R}^{D-1}`

    The ilr transformation is defined as follows

    .. math::
        ilr(x) =
        [\langle x, e_1 \rangle_a, \ldots, \langle x, e_{D-1} \rangle_a]

    where :math:`[e_1,\ldots,e_{D-1}]` is an orthonormal basis in the simplex.

    If an orthornormal basis isn't specified, the J. J. Egozcue orthonormal
    basis derived from Gram-Schmidt orthogonalization will be used by
    default.

    Parameters
    ----------
    mat: numpy.ndarray
       a matrix of proportions where
       rows = compositions and
       columns = components
       each composition (row) must add up to unity (see :ref:`closure()`)
    basis: numpy.ndarray, float, optional
        orthonormal basis for Aitchison simplex
        defaults to J.J.Egozcue orthonormal basis.
    check: bool
        Specifies if the basis is orthonormal.

    Examples
    --------
    >>> import numpy as np
    >>> from composition_stats import ilr
    >>> x = np.array([.1, .3, .4, .2])
    >>> ilr(x)
    array([-0.7768362 , -0.68339802,  0.11704769])

    Notes
    -----
    If the `basis` parameter is specified, it is expected to be a basis in the
    Aitchison simplex.  If there are `D-1` elements specified in `mat`, then
    the dimensions of the basis needs be `D-1 x D`, where rows represent
    basis vectors, and the columns represent proportions.
    """
    if basis is None:
        basis = clr_inv(_gram_schmidt_basis(mat.shape[-1]))
    else:
        if len(basis.shape) != 2:
            raise ValueError("Basis needs to be a 2D matrix, "
                             "not a %dD matrix." %
                             (len(basis.shape)))
        if check:
            _check_orthogonality(basis)

    return inner(mat, basis)


def ilr_inv(mat, basis=None, check=True):
    r"""
    Performs inverse isometric log ratio transform.

    This function transforms compositions from the real space to
    Aitchison geometry. The :math:`ilr^{-1}` transform is both an isometry,
    and an isomorphism defined on the following spaces

    :math:`ilr^{-1}: \mathbb{R}^{D-1} \rightarrow S^D`

    The inverse ilr transformation is defined as follows

    .. math::
        ilr^{-1}(x) = \bigoplus\limits_{i=1}^{D-1} x \odot e_i

    where :math:`[e_1,\ldots, e_{D-1}]` is an orthonormal basis in the simplex.

    If an orthonormal basis isn't specified, the J. J. Egozcue orthonormal
    basis derived from Gram-Schmidt orthogonalization will be used by
    default.


    Parameters
    ----------
    mat: numpy.ndarray, float
       a matrix of transformed proportions where
       rows = compositions and
       columns = components
       each composition (row) must add up to unity (see :ref:`closure()`)

    basis: numpy.ndarray, float, optional
        orthonormal basis for Aitchison simplex
        defaults to J.J.Egozcue orthonormal basis

    check: bool
        Specifies if the basis is orthonormal.


    Examples
    --------
    >>> import numpy as np
    >>> from composition_stats import ilr
    >>> x = np.array([.1, .3, .6,])
    >>> ilr_inv(x)
    array([ 0.34180297,  0.29672718,  0.22054469,  0.14092516])

    Notes
    -----
    If the `basis` parameter is specified, it is expected to be a basis in the
    Aitchison simplex.  If there are `D-1` elements specified in `mat`, then
    the dimensions of the basis needs be `D-1 x D`, where rows represent
    basis vectors, and the columns represent proportions.
    """

    if basis is None:
        basis = _gram_schmidt_basis(mat.shape[-1] + 1)
    else:
        if len(basis.shape) != 2:
            raise ValueError("Basis needs to be a 2D matrix, "
                             "not a %dD matrix." %
                             (len(basis.shape)))
        if check:
            _check_orthogonality(basis)
        # this is necessary, since the clr function
        # performs np.squeeze()
        basis = np.atleast_2d(clr(basis))

    return clr_inv(np.dot(mat, basis))


def alr(mat, denominator_idx=0):
    r"""
    Performs additive log ratio transformation.

    This function transforms compositions from a D-part Aitchison simplex to
    a non-isometric real space of D-1 dimensions. The argument
    `denominator_col` defines the index of the column used as the common
    denominator. The :math: `alr` transformed data are amenable to multivariate
    analysis as long as statistics don't involve distances.

    :math:`alr: S^D \rightarrow \mathbb{R}^{D-1}`

    The alr transformation is defined as follows

    .. math::
        alr(x) = \left[ \ln \frac{x_1}{x_D}, \ldots,
        \ln \frac{x_{D-1}}{x_D} \right]

    where :math:`D` is the index of the part used as common denominator.

    Parameters
    ----------
    mat: numpy.ndarray
       a matrix of proportions where
       rows = compositions and
       columns = components
       each composition (row) must add up to unity (see :ref:`closure()`)
    denominator_idx: int
       the index of the column (2D-matrix) or position (vector) of
       `mat` which should be used as the reference composition. By default
       `denominator_idx=0` to specify the first column or position.

    Returns
    -------
    numpy.ndarray
         alr-transformed data projected in a non-isometric real space
         of D-1 dimensions for a D-parts composition

    Examples
    --------
    >>> import numpy as np
    >>> from composition_stats import alr
    >>> x = np.array([.1, .3, .4, .2])
    >>> alr(x)
    array([ 1.09861229,  1.38629436,  0.69314718])
    """
    if mat.ndim == 2:
        mat_t = mat.T
        numerator_idx = list(range(0, mat_t.shape[0]))
        del numerator_idx[denominator_idx]
        lr = np.log(mat_t[numerator_idx, :]/mat_t[denominator_idx, :]).T
    elif mat.ndim == 1:
        numerator_idx = list(range(0, mat.shape[0]))
        del numerator_idx[denominator_idx]
        lr = np.log(mat[numerator_idx]/mat[denominator_idx])
    else:
        raise ValueError("mat must be either 1D or 2D")
    return lr


def alr_inv(mat, denominator_idx=0):
    r"""
    Performs inverse additive log ratio transform.

    This function transforms compositions from the non-isometric real space of
    alrs to Aitchison geometry.

    :math:`alr^{-1}: \mathbb{R}^{D-1} \rightarrow S^D`

    The inverse alr transformation is defined as follows

    .. math::
         alr^{-1}(x) = C[exp([y_1, y_2, ..., y_{D-1}, 0])]

    where :math:`C[x]` is the closure operation defined as

    .. math::
        C[x] = \left[\frac{x_1}{\sum_{i=1}^{D} x_i},\ldots,
                     \frac{x_D}{\sum_{i=1}^{D} x_i} \right]

    for some :math:`D` dimensional real vector :math:`x` and
    :math:`D` is the number of components for every composition.

    Parameters
    ----------
    mat: numpy.ndarray
       a matrix of alr-transformed data
    denominator_idx: int
       the index of the column (2D-composition) or position (1D-composition) of
       the output where the common denominator should be placed. By default
       `denominator_idx=0` to specify the first column or position.

    Returns
    -------
    numpy.ndarray
         Inverse alr transformed matrix or vector where rows sum to 1.

    Examples
    --------
    >>> import numpy as np
    >>> from composition_stats import alr, alr_inv
    >>> x = np.array([.1, .3, .4, .2])
    >>> alr_inv(alr(x))
    array([ 0.1,  0.3,  0.4,  0.2])
    """
    mat = np.array(mat)
    if mat.ndim == 2:
        mat_idx = np.insert(mat, denominator_idx,
                            np.repeat(0, mat.shape[0]), axis=1)
        comp = np.zeros(mat_idx.shape)
        comp[:, denominator_idx] = 1 / (np.exp(mat).sum(axis=1) + 1)
        numerator_idx = list(range(0, comp.shape[1]))
        del numerator_idx[denominator_idx]
        for i in numerator_idx:
            comp[:, i] = comp[:, denominator_idx] * np.exp(mat_idx[:, i])
    elif mat.ndim == 1:
        mat_idx = np.insert(mat, denominator_idx, 0, axis=0)
        comp = np.zeros(mat_idx.shape)
        comp[denominator_idx] = 1 / (np.exp(mat).sum(axis=0) + 1)
        numerator_idx = list(range(0, comp.shape[0]))
        del numerator_idx[denominator_idx]
        for i in numerator_idx:
            comp[i] = comp[denominator_idx] * np.exp(mat_idx[i])
    else:
        raise ValueError("mat must be either 1D or 2D")
    return comp


def center(mat):
    """
    Compute the geometric average of data.

    Parameters
    ----------
    mat : array_like
       a matrix of proportions where
       rows = compositions
       columns = components
       each composition (row) must add up to unity (see :ref:`closure()`)

    Returns
    -------
    array_like, np.float64
       central composition

    Examples
    --------
    >>> import numpy as np
    >>> from composition_stats import center
    >>> X = np.array([[.1,.3,.4, .2],[.2,.2,.2,.4]])
    >>> center(X)
    array([ 0.14854315,  0.25728427,  0.29708629,  0.29708629])

    """
    cen = scipy.stats.gmean(mat, axis=0)
    return closure(cen, out=cen)


def centralize(mat):
    r"""Center data around its geometric average.

    Parameters
    ----------
    mat : array_like, float
       a matrix of proportions where
       rows = compositions and
       columns = components
       each composition (row) must add up to unity (see :ref:`closure()`)

    Returns
    -------
    numpy.ndarray
         centered composition matrix

    Examples
    --------
    >>> import numpy as np
    >>> from composition_stats import centralize
    >>> X = np.array([[.1,.3,.4, .2],[.2,.2,.2,.4]])
    >>> centralize(X)
    array([[ 0.17445763,  0.30216948,  0.34891526,  0.17445763],
           [ 0.32495488,  0.18761279,  0.16247744,  0.32495488]])

    """
    cen = scipy.stats.gmean(mat, axis=0)
    return perturb_inv(mat, cen)


def _gram_schmidt_basis(n):
    """
    Builds clr transformed basis derived from
    gram schmidt orthogonalization

    Parameters
    ----------
    n : int
        Dimension of the Aitchison simplex
    """
    basis = np.zeros((n, n-1))
    for j in range(n-1):
        i = j + 1
        e = np.array([(1/i)]*i + [-1] +
                     [0]*(n-i-1))*np.sqrt(i/(i+1))
        basis[:, j] = e
    return basis.T


def sbp_basis(sbp):
    r"""
    Builds an orthogonal basis from a sequential binary partition (SBP). As
    explained in [1]_, the SBP is a hierarchical collection of binary
    divisions of compositional parts. The child groups are divided again until
    all groups contain a single part. The SBP can be encoded in a
    :math:`(D - 1) \times D` matrix where, for each row, parts can be grouped
    by -1 and +1 tags, and 0 for excluded parts. The `sbp_basis` method was
    originally derived from function `gsi.buildilrBase()` found in the R
    package `compositions` [2]_. The ith balance is computed as follows

    .. math::
        b_i = \sqrt{ \frac{r_i s_i}{r_i+s_i} }
        \ln \left( \frac{g(x_{r_i})}{g(x_{s_i})} \right)

    where :math:`b_i` is the ith balance corresponding to the ith row in the
    SBP, :math:`r_i` and :math:`s_i` and the number of respectively `+1` and
    `-1` labels in the ith row of the SBP and where :math:`g(x) =
    (\prod\limits_{i=1}^{D} x_i)^{1/D}` is the geometric mean of :math:`x`.

    Parameters
    ----------
    sbp: np.array, int
        A contrast matrix, also known as a sequential binary partition, where
        every row represents a partition between two groups of features. A part
        labelled `+1` would correspond to that feature being in the numerator
        of the given row partition, a part labelled `-1` would correspond to
        features being in the denominator of that given row partition, and `0`
        would correspond to features excluded in the row partition.

    Returns
    -------
    numpy.array
        An orthonormal basis in the Aitchison simplex

    Examples
    --------
    >>> import numpy as np
    >>> sbp = np.array([[1, 1,-1,-1,-1],
    ...                 [1,-1, 0, 0, 0],
    ...                 [0, 0, 1,-1,-1],
    ...                 [0, 0, 0, 1,-1]])
    ...
    >>> sbp_basis(sbp)
    array([[ 0.31209907,  0.31209907,  0.12526729,  0.12526729,  0.12526729],
           [ 0.36733337,  0.08930489,  0.18112058,  0.18112058,  0.18112058],
           [ 0.17882092,  0.17882092,  0.40459293,  0.11888261,  0.11888261],
           [ 0.18112058,  0.18112058,  0.18112058,  0.36733337,  0.08930489]])

    References
    ----------
    .. [1] Parent, S.É., Parent, L.E., Egozcue, J.J., Rozane, D.E.,
       Hernandes, A., Lapointe, L., Hébert-Gentile, V., Naess, K.,
       Marchand, S., Lafond, J., Mattos, D., Barlow, P., Natale, W., 2013.
       The plant ionome revisited by the nutrient balance concept.
       Front. Plant Sci. 4, 39, http://dx.doi.org/10.3389/fpls.2013.00039.
    .. [2] van den Boogaart, K. Gerald, Tolosana-Delgado, Raimon and Bren,
       Matevz, 2014. `compositions`: Compositional Data Analysis. R package
       version 1.40-1. https://CRAN.R-project.org/package=compositions.
    """

    n_pos = (sbp == 1).sum(axis=1)
    n_neg = (sbp == -1).sum(axis=1)
    psi = np.zeros(sbp.shape)
    for i in range(0, sbp.shape[0]):
        psi[i, :] = sbp[i, :] * np.sqrt((n_neg[i] / n_pos[i])**sbp[i, :] /
                                        np.sum(np.abs(sbp[i, :])))
    return clr_inv(psi)


def _check_orthogonality(basis):
    """
    Checks to see if basis is truly orthonormal in the
    Aitchison simplex

    Parameters
    ----------
    basis: numpy.ndarray
        basis in the Aitchison simplex
    """
    basis = np.atleast_2d(basis)
    if not np.allclose(inner(basis, basis), np.identity(len(basis)),
                       rtol=1e-4, atol=1e-6):
        raise ValueError("Aitchison basis is not orthonormal")
