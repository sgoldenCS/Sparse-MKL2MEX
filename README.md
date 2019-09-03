# Sparse-MKL2MEX
A MATLAB mex interface to Intel's MKL for Sparse Matrix Multiply

This can be compiled on linux using MEX with the following linked files/libraries:
mex -I<INSTALL_DIR>/intel/mkl/include sparseMultiply.cpp <INSTALL_DIR>/intel/mkl/lib/intel64/libmkl_rt.so
