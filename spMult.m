function[y] = spMult(A,x)
%Only needed to pass CSR formatted matrix to the C++ code
%It may improve performance to use the line below directly
y = sparseMultiply(A',x);
end
