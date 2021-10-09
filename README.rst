======================
Composition Statistics
======================

|Tests|  |Docs|

Python module for compositional data analysis. Install with pip::

    pip install composition_stats

The following functions are provided:

+-----------------------------------+------------------------------------------+
| ``closure``                       | Performs closure to ensure that all      |
|                                   | elements add up to 1.                    |
+-----------------------------------+------------------------------------------+
| ``multiplicative_replacement``    | Replace all zeros with small non-zero    |
|                                   | alues                                    |
+-----------------------------------+------------------------------------------+
| ``perturb``                       | Performs the perturbation operation.     |
+-----------------------------------+------------------------------------------+
| ``perturb_inv``                   | Performs the inverse perturbation        |
|                                   | operation.                               |
+-----------------------------------+------------------------------------------+
| ``power``                         | Performs the power operation.            |
+-----------------------------------+------------------------------------------+
| ``inner``                         | Calculates the Aitchson inner product.   |
+-----------------------------------+------------------------------------------+
| ``clr``                           | Performs centre log ratio                |
|                                   | transformation.                          |
+-----------------------------------+------------------------------------------+
| ``clr_inv``                       | Performs inverse centre log ratio        |
|                                   | transformation.                          |
+-----------------------------------+------------------------------------------+
| ``ilr``                           | Performs isometric log ratio             |
|                                   | transformation.                          |
+-----------------------------------+------------------------------------------+
| ``ilr_inv``                       | Performs inverse isometric log ratio     |
|                                   | transform.                               |
+-----------------------------------+------------------------------------------+
| ``alr``                           | Performs additive log ratio              |
|                                   | transformation.                          |
+-----------------------------------+------------------------------------------+
| ``alr_inv``                       | Performs inverse additive log ratio      |
|                                   | transform.                               |
+-----------------------------------+------------------------------------------+
| ``center``                        | Computes the geometric average of data.  |
+-----------------------------------+------------------------------------------+
| ``centralize``                    | Center data around its geometric         |
|                                   | average.                                 |
+-----------------------------------+------------------------------------------+
| ``sbp_basis``                     | Builds an orthogonal basis from a        |
|                                   | sequential binary partition (SBP).       |
+-----------------------------------+------------------------------------------+

Please see the `documentation`_ for details and a complete function reference.

This is a fork of the essential compositional data functions of the
``skbio.stats.composition`` module from `scikit-bio`_.  However, for reasons of
performance, the functions in this module expect inputs where compositions
already sum to unity, and do not call `closure()` on inputs internally.

.. _documentation: https://composition-stats.readthedocs.io/
.. _scikit-bio: https://github.com/biocore/scikit-bio

.. |Tests| image:: https://github.com/ntessore/composition_stats/actions/workflows/test.yml/badge.svg
   :target: https://github.com/ntessore/composition_stats/actions/workflows/test.yml
   :alt: Tests

.. |Docs| image:: https://readthedocs.org/projects/composition-stats/badge/?version=latest
   :target: https://composition-stats.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
