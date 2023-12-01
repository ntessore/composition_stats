# The following parts are originally part of scikit-bio, with copyright notice
# as reproduced below.  The original COPYING.txt file can be found under
# licenses/scikit-bio.txt.
# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from unittest import TestCase
import numpy as np
import numpy.testing as npt
from composition_stats import (closure, multiplicative_replacement, perturb,
                               perturb_inv, power, inner, clr, clr_inv, ilr,
                               ilr_inv, alr, alr_inv, sbp_basis,
                               _gram_schmidt_basis, center, centralize)


class CompositionTests(TestCase):

    def setUp(self):
        # Compositional data
        self.cdata1 = np.array([[2, 2, 6],
                                [4, 4, 2]])
        self.cdata2 = np.array([2, 2, 6])

        self.cdata3 = np.array([[1, 2, 3, 0, 5],
                                [1, 0, 0, 4, 5],
                                [1, 2, 3, 4, 5]])
        self.cdata4 = np.array([1, 2, 3, 0, 5])
        self.cdata5 = [[2, 2, 6], [4, 4, 2]]
        self.cdata6 = [[1, 2, 3, 0, 5],
                       [1, 0, 0, 4, 5],
                       [1, 2, 3, 4, 5]]
        self.cdata7 = [np.exp(1), 1, 1]
        self.cdata8 = [np.exp(1), 1, 1, 1]

        # Simplicial orthonormal basis obtained from Gram-Schmidt
        self.ortho1 = [[0.44858053, 0.10905743, 0.22118102, 0.22118102],
                       [0.3379924, 0.3379924, 0.0993132, 0.22470201],
                       [0.3016453, 0.3016453, 0.3016453, 0.09506409]]

        # Real data
        self.rdata1 = [[0.70710678, -0.70710678, 0., 0.],
                       [0.40824829, 0.40824829, -0.81649658, 0.],
                       [0.28867513, 0.28867513, 0.28867513, -0.8660254]]

        # Bad datasets
        # negative count
        self.bad1 = np.array([1, 2, -1])
        # zero count
        self.bad2 = np.array([[[1, 2, 3, 0, 5]]])

    def test_closure(self):

        npt.assert_allclose(closure(self.cdata1),
                            np.array([[.2, .2, .6],
                                      [.4, .4, .2]]))
        npt.assert_allclose(closure(self.cdata2),
                            np.array([.2, .2, .6]))
        npt.assert_allclose(closure(self.cdata5),
                            np.array([[.2, .2, .6],
                                      [.4, .4, .2]]))
        with self.assertRaises(ValueError):
            closure(self.bad1)

        with self.assertRaises(ValueError):
            closure(self.bad2)

        # make sure that inplace modification is not occurring
        closure(self.cdata2)
        npt.assert_allclose(self.cdata2, np.array([2, 2, 6]))

    def test_closure_warning(self):
        with self.assertRaises(ValueError):
            closure([0., 0., 0.])

        with self.assertRaises(ValueError):
            closure([[0., 0., 0.],
                     [0., 5., 5.]])

    def test_perturb(self):
        pmat = perturb(closure(self.cdata1),
                       closure(np.array([1, 1, 1])))
        npt.assert_allclose(pmat,
                            np.array([[.2, .2, .6],
                                      [.4, .4, .2]]))

        pmat = perturb(closure(self.cdata1),
                       closure(np.array([10, 10, 20])))
        npt.assert_allclose(pmat,
                            np.array([[.125, .125, .75],
                                      [1./3, 1./3, 1./3]]))

        pmat = perturb(closure(self.cdata1),
                       closure(np.array([10, 10, 20])))
        npt.assert_allclose(pmat,
                            np.array([[.125, .125, .75],
                                      [1./3, 1./3, 1./3]]))

        pmat = perturb(closure(self.cdata2),
                       closure([1, 2, 1]))
        npt.assert_allclose(pmat, np.array([1./6, 2./6, 3./6]))

        pmat = perturb(closure(self.cdata5),
                       closure(np.array([1, 1, 1])))
        npt.assert_allclose(pmat,
                            np.array([[.2, .2, .6],
                                      [.4, .4, .2]]))

        # make sure that inplace modification is not occurring
        perturb(closure(self.cdata2), closure([1, 2, 3]))
        npt.assert_allclose(self.cdata2, np.array([2, 2, 6]))

    def test_power(self):
        pmat = power(closure(self.cdata1), 2)
        npt.assert_allclose(pmat,
                            np.array([[.04/.44, .04/.44, .36/.44],
                                      [.16/.36, .16/.36, .04/.36]]))

        pmat = power(closure(self.cdata2), 2)
        npt.assert_allclose(pmat, np.array([.04, .04, .36])/.44)

        pmat = power(closure(self.cdata5), 2)
        npt.assert_allclose(pmat,
                            np.array([[.04/.44, .04/.44, .36/.44],
                                      [.16/.36, .16/.36, .04/.36]]))

        # make sure that inplace modification is not occurring
        power(closure(self.cdata2), 4)
        npt.assert_allclose(self.cdata2, np.array([2, 2, 6]))

    def test_perturb_inv(self):
        pmat = perturb_inv(closure(self.cdata1),
                           closure([.1, .1, .1]))
        imat = perturb(closure(self.cdata1),
                       closure([10, 10, 10]))
        npt.assert_allclose(pmat, imat)
        pmat = perturb_inv(closure(self.cdata1),
                           closure([1, 1, 1]))
        npt.assert_allclose(pmat,
                            closure([[.2, .2, .6],
                                     [.4, .4, .2]]))
        pmat = perturb_inv(closure(self.cdata5),
                           closure([.1, .1, .1]))
        imat = perturb(closure(self.cdata1), closure([10, 10, 10]))
        npt.assert_allclose(pmat, imat)

        # make sure that inplace modification is not occurring
        perturb_inv(closure(self.cdata2), closure([1, 2, 3]))
        npt.assert_allclose(self.cdata2, np.array([2, 2, 6]))

    def test_inner(self):
        a = inner(closure(self.cdata5), closure(self.cdata5))
        npt.assert_allclose(a, np.array([[0.80463264, -0.50766667],
                                         [-0.50766667, 0.32030201]]))

        b = inner(closure(self.cdata7), closure(self.cdata7))
        npt.assert_allclose(b, 0.66666666666666663)

        # Make sure that orthogonality holds
        npt.assert_allclose(inner(closure(self.ortho1), closure(self.ortho1)),
                            np.identity(3), rtol=1e-04, atol=1e-06)

        with self.assertRaises(ValueError):
            inner(closure(self.cdata1), closure(self.cdata8))

        # make sure that inplace modification is not occurring
        inner(closure(self.cdata1), closure(self.cdata1))
        npt.assert_allclose(self.cdata1,
                            np.array([[2, 2, 6],
                                      [4, 4, 2]]))

    def test_multiplicative_replacement(self):
        amat = multiplicative_replacement(closure(self.cdata3))
        npt.assert_allclose(amat,
                            np.array([[0.087273, 0.174545, 0.261818,
                                       0.04, 0.436364],
                                      [0.092, 0.04, 0.04, 0.368, 0.46],
                                      [0.066667, 0.133333, 0.2,
                                       0.266667, 0.333333]]),
                            rtol=1e-5, atol=1e-5)

        amat = multiplicative_replacement(closure(self.cdata4))
        npt.assert_allclose(amat,
                            np.array([0.087273, 0.174545, 0.261818,
                                      0.04, 0.436364]),
                            rtol=1e-5, atol=1e-5)

        amat = multiplicative_replacement(closure(self.cdata6))
        npt.assert_allclose(amat,
                            np.array([[0.087273, 0.174545, 0.261818,
                                       0.04, 0.436364],
                                      [0.092, 0.04, 0.04, 0.368, 0.46],
                                      [0.066667, 0.133333, 0.2,
                                       0.266667, 0.333333]]),
                            rtol=1e-5, atol=1e-5)

        # make sure that inplace modification is not occurring
        multiplicative_replacement(closure(self.cdata4))
        npt.assert_allclose(self.cdata4, np.array([1, 2, 3, 0, 5]))

    def multiplicative_replacement_warning(self):
        with self.assertRaises(ValueError):
            multiplicative_replacement(closure([0, 1, 2]), delta=1)

    def test_clr(self):
        cmat = clr(closure(self.cdata1))
        A = np.array([.2, .2, .6])
        B = np.array([.4, .4, .2])

        npt.assert_allclose(cmat,
                            [np.log(A / np.exp(np.log(A).mean())),
                             np.log(B / np.exp(np.log(B).mean()))])
        cmat = clr(closure(self.cdata2))
        A = np.array([.2, .2, .6])
        npt.assert_allclose(cmat,
                            np.log(A / np.exp(np.log(A).mean())))

        cmat = clr(closure(self.cdata5))
        A = np.array([.2, .2, .6])
        B = np.array([.4, .4, .2])

        npt.assert_allclose(cmat,
                            [np.log(A / np.exp(np.log(A).mean())),
                             np.log(B / np.exp(np.log(B).mean()))])

        # make sure that inplace modification is not occurring
        clr(closure(self.cdata2))
        npt.assert_allclose(self.cdata2, np.array([2, 2, 6]))

        # test ignore_zero
        msg = 'divide by zero encountered'
        with self.assertWarnsRegex(RuntimeWarning, msg):
            cmat = clr(closure(self.cdata3), ignore_zero=False)
            assert not np.all(np.isfinite(cmat))
        cmat = clr(closure(self.cdata3), ignore_zero=True)
        assert np.all(np.isfinite(cmat))
        npt.assert_allclose(cmat[0, 0], -0.85029934)

    def test_clr_inv(self):
        npt.assert_allclose(clr_inv(self.rdata1), self.ortho1)
        npt.assert_array_almost_equal(clr(clr_inv(self.rdata1)), self.rdata1)

        # make sure that inplace modification is not occurring
        clr_inv(self.rdata1)
        npt.assert_allclose(self.rdata1,
                            np.array([[0.70710678, -0.70710678, 0., 0.],
                                      [0.40824829, 0.40824829,
                                       -0.81649658, 0.],
                                      [0.28867513, 0.28867513,
                                       0.28867513, -0.8660254]]))

    def test_center(self):
        cmat = center(closure(self.cdata1))
        npt.assert_allclose(cmat,
                            np.array([0.31010205, 0.31010205, 0.37979590]))
        cmat = center(closure(self.cdata5))
        npt.assert_allclose(cmat,
                            np.array([0.31010205, 0.31010205, 0.37979590]))

        # make sure that inplace modification is not occurring
        center(closure(self.cdata1))
        npt.assert_allclose(self.cdata1,
                            np.array([[2, 2, 6],
                                      [4, 4, 2]]))

    def test_centralize(self):
        cmat = centralize(closure(self.cdata1))
        npt.assert_allclose(cmat,
                            np.array([[0.22474487, 0.22474487, 0.55051026],
                                      [0.41523958, 0.41523958, 0.16952085]]))
        cmat = centralize(closure(self.cdata5))
        npt.assert_allclose(cmat,
                            np.array([[0.22474487, 0.22474487, 0.55051026],
                                      [0.41523958, 0.41523958, 0.16952085]]))

        # make sure that inplace modification is not occurring
        centralize(closure(self.cdata1))
        npt.assert_allclose(self.cdata1,
                            np.array([[2, 2, 6],
                                      [4, 4, 2]]))

    def test_ilr(self):
        mat = closure(self.cdata7)
        npt.assert_array_almost_equal(ilr(mat),
                                      np.array([0.70710678, 0.40824829]))

        # Should give same result as inner
        npt.assert_allclose(ilr(closure(self.ortho1)), np.identity(3),
                            rtol=1e-04, atol=1e-06)

        with self.assertRaises(ValueError):
            ilr(closure(self.cdata1), basis=self.cdata1)

        # make sure that inplace modification is not occurring
        ilr(closure(self.cdata1))
        npt.assert_allclose(self.cdata1,
                            np.array([[2, 2, 6],
                                      [4, 4, 2]]))

    def test_ilr_basis(self):
        table = np.array([[1., 10.],
                          [1.14141414, 9.90909091],
                          [1.28282828, 9.81818182],
                          [1.42424242, 9.72727273],
                          [1.56565657, 9.63636364]])
        basis = np.array([[0.80442968, 0.19557032]])
        res = ilr(closure(table), basis=basis)
        exp = np.array([np.log(1/10)*np.sqrt(1/2),
                        np.log(1.14141414 / 9.90909091)*np.sqrt(1/2),
                        np.log(1.28282828 / 9.81818182)*np.sqrt(1/2),
                        np.log(1.42424242 / 9.72727273)*np.sqrt(1/2),
                        np.log(1.56565657 / 9.63636364)*np.sqrt(1/2)])

        npt.assert_allclose(res, exp)

    def test_ilr_basis_one_dimension_error(self):
        table = np.array([[1., 10.],
                          [1.14141414, 9.90909091],
                          [1.28282828, 9.81818182],
                          [1.42424242, 9.72727273],
                          [1.56565657, 9.63636364]])
        basis = np.array([0.80442968, 0.19557032])
        with self.assertRaises(ValueError):
            ilr(closure(table), basis=basis)

    def test_ilr_inv(self):
        mat = closure(self.cdata7)
        npt.assert_array_almost_equal(ilr_inv(ilr(mat)), mat)

        npt.assert_allclose(ilr_inv(np.identity(3)), self.ortho1,
                            rtol=1e-04, atol=1e-06)

        with self.assertRaises(ValueError):
            ilr_inv(self.cdata1, basis=self.cdata1)

        # make sure that inplace modification is not occurring
        ilr_inv(self.cdata1)
        npt.assert_allclose(self.cdata1,
                            np.array([[2, 2, 6],
                                      [4, 4, 2]]))

    def test_ilr_basis_isomorphism(self):
        # tests to make sure that the isomorphism holds
        # with the introduction of the basis.
        basis = np.array([[0.80442968, 0.19557032]])
        table = np.array([[np.log(1/10)*np.sqrt(1/2),
                           np.log(1.14141414 / 9.90909091)*np.sqrt(1/2),
                           np.log(1.28282828 / 9.81818182)*np.sqrt(1/2),
                           np.log(1.42424242 / 9.72727273)*np.sqrt(1/2),
                           np.log(1.56565657 / 9.63636364)*np.sqrt(1/2)]]).T
        res = ilr(ilr_inv(table, basis=basis), basis=basis)
        npt.assert_allclose(res, table.squeeze())

        table = np.array([[1., 10.],
                          [1.14141414, 9.90909091],
                          [1.28282828, 9.81818182],
                          [1.42424242, 9.72727273],
                          [1.56565657, 9.63636364]])

        res = ilr_inv(np.atleast_2d(ilr(closure(table), basis=basis)).T,
                      basis=basis)
        npt.assert_allclose(res, closure(table.squeeze()))

    def test_ilr_inv_basis(self):
        exp = closure(np.array([[1., 10.],
                                [1.14141414, 9.90909091],
                                [1.28282828, 9.81818182],
                                [1.42424242, 9.72727273],
                                [1.56565657, 9.63636364]]))
        basis = np.array([[0.80442968, 0.19557032]])
        table = np.array([[np.log(1/10)*np.sqrt(1/2),
                           np.log(1.14141414 / 9.90909091)*np.sqrt(1/2),
                           np.log(1.28282828 / 9.81818182)*np.sqrt(1/2),
                           np.log(1.42424242 / 9.72727273)*np.sqrt(1/2),
                           np.log(1.56565657 / 9.63636364)*np.sqrt(1/2)]]).T
        res = ilr_inv(table, basis=basis)
        npt.assert_allclose(res, exp)

    def test_ilr_inv_basis_one_dimension_error(self):
        basis = clr(np.array([[0.80442968, 0.19557032]]))
        table = np.array([[np.log(1/10)*np.sqrt(1/2),
                           np.log(1.14141414 / 9.90909091)*np.sqrt(1/2),
                           np.log(1.28282828 / 9.81818182)*np.sqrt(1/2),
                           np.log(1.42424242 / 9.72727273)*np.sqrt(1/2),
                           np.log(1.56565657 / 9.63636364)*np.sqrt(1/2)]]).T
        with self.assertRaises(ValueError):
            ilr_inv(table, basis=basis)

    def test_alr(self):
        # 2d-composition
        comp1 = closure(self.cdata1)
        alr2d_byhand = np.array([np.log(comp1[:, 0]/comp1[:, 1]),
                                 np.log(comp1[:, 2]/comp1[:, 1])]).T
        alr2d_method = alr(comp1, denominator_idx=1)
        npt.assert_allclose(alr2d_byhand, alr2d_method)

        # 1d-composition
        comp2 = closure(self.cdata2)
        alr1d_byhand = np.array([np.log(comp2[0]/comp2[1]),
                                 np.log(comp2[2]/comp2[1])]).T
        alr1d_method = alr(comp2, denominator_idx=1)
        npt.assert_allclose(alr1d_byhand, alr1d_method)

        # make sure that inplace modification is not occurring
        alr(closure(self.cdata2))
        npt.assert_allclose(self.cdata2, np.array([2, 2, 6]))

    def test_alr_inv(self):
        # 2d-composition
        comp1 = closure(self.cdata1)
        alr2d_byhand = np.array([np.log(comp1[:, 0]/comp1[:, 1]),
                                 np.log(comp1[:, 2]/comp1[:, 1])]).T
        alr2d_method = alr(comp1, denominator_idx=1)
        B = 1/(1 + np.exp(alr2d_byhand[:, 0]) + np.exp(alr2d_byhand[:, 1]))
        A = B * np.exp(alr2d_byhand[:, 0])
        C = B * np.exp(alr2d_byhand[:, 1])
        alrinv2d_byhand = np.column_stack((A, B, C))
        alrinv2d_method = alr_inv(alr2d_method, denominator_idx=1)
        npt.assert_allclose(alrinv2d_byhand, alrinv2d_method)

        # 1d-composition
        comp2 = closure(self.cdata2)
        alr1d_byhand = np.array([np.log(comp2[0]/comp2[1]),
                                 np.log(comp2[2]/comp2[1])]).T
        alr1d_method = alr(comp2, denominator_idx=1)
        B = 1/(1 + np.exp(alr1d_byhand[0]) + np.exp(alr1d_byhand[1]))
        A = B * np.exp(alr1d_byhand[0])
        C = B * np.exp(alr1d_byhand[1])
        alrinv1d_byhand = np.column_stack((A, B, C))[0, :]
        alrinv1d_method = alr_inv(alr1d_method, denominator_idx=1)
        npt.assert_allclose(alrinv1d_byhand, alrinv1d_method)

        # make sure that inplace modification is not occurring
        alr_inv(self.rdata1)
        npt.assert_allclose(self.rdata1,
                            np.array([[0.70710678, -0.70710678, 0., 0.],
                                      [0.40824829, 0.40824829,
                                       -0.81649658, 0.],
                                      [0.28867513, 0.28867513,
                                       0.28867513, -0.8660254]]))

    def test_sbp_basis_gram_schmidt(self):
        gsbasis = clr_inv(_gram_schmidt_basis(5))
        sbp = np.array([[1, -1, 0, 0, 0],
                        [1, 1, -1, 0, 0],
                        [1, 1, 1, -1, 0],
                        [1, 1, 1, 1, -1]])
        sbpbasis = sbp_basis(sbp)
        npt.assert_allclose(gsbasis, sbpbasis)

    def test_sbp_basis_elementwise(self):
        sbp = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1],
                        [1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 0],
                        [1, 1, 1, 1, 1, 1, -1, 0, 0, 0, 0, 0],
                        [1, 1, -1, -1, -1, 1, 0, 0, 0, 0, 0, 0],
                        [1, -1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, -1, -1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0]])
        sbpbasis = sbp_basis(sbp)
        # by hand, element-wise
        r = np.apply_along_axis(func1d=lambda x: np.sum(x > 0),
                                axis=1, arr=sbp)
        s = np.apply_along_axis(func1d=lambda x: np.sum(x < 0),
                                axis=1, arr=sbp)
        psi = np.zeros(sbp.shape)
        for i in range(0, sbp.shape[0]):
            for j in range(0, sbp.shape[1]):
                if sbp[i, j] == 1:
                    psi[i, j] = np.sqrt(s[i]/(r[i]*(r[i]+s[i])))
                elif sbp[i, j] == -1:
                    psi[i, j] = -np.sqrt(r[i]/(s[i]*(r[i]+s[i])))
        basis_byhand = clr_inv(psi)
        npt.assert_allclose(basis_byhand, sbpbasis)
