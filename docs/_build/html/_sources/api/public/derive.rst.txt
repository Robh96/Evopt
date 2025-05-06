Symbolic Regression (:py:mod:`evopt.derive`)
============================================

The derive module provides symbolic regression capabilities for equation discovery from the results data.
Built on top of the Miles Cranmer's `PySR` engine for symbolic regression.

Classes
-------

.. currentmodule:: evopt.derive

.. autoclass:: Derive
   :undoc-members:
   :show-inheritance:
   
   Model Configuration
   ~~~~~~~~~~~~~~~~~~~
   
   .. automethod:: __init__
   
   Regression & Prediction
   ~~~~~~~~~~~~~~~~~~~~~~~
   
   .. automethod:: fit
   .. automethod:: predict
   
   Visualization
   ~~~~~~~~~~~~~
   
   .. automethod:: parity_plot