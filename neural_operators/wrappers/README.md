# Model Wrappers

This directory contains wrapper classes designed to interface neural operator models with specific physical problems or application constraints.

## Contents

The main file is `wrap_model.py`, which provides a general utility for wrapping models. Other files are specific wrappers for different problems.

- **AirfoilWrapper.py**: Wrapper for airfoil flow simulation problems.
- **BAMPNO_Continuation_Wrapper.py**: Wrapper involving BAMPNO (possibly Batch Active Memory PNO) and continuation methods.
- **CrossTrussWrapper.py**: Wrapper for cross-truss structure problems.
- **StiffnessMatrixWrapper.py**: Wrapper managing stiffness matrices in structural mechanics problems.
