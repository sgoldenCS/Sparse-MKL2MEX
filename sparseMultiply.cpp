#include "mex.h"
#include "matrix.h"
#include <stdio.h>
#include <omp.h>
#include "mkl_spblas.h" 

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	mwIndex *col_index, *row_counts;
	double* vals;
	//const int *col_index, *row_first, *row_last, *row_counts;
	const int ncols = mxGetM(prhs[0]); 	//columns of real A
	const int mrows = mxGetN(prhs[0]); 	//rows of real A
	int nnz;
	MKL_INT ldx = mxGetM(prhs[1]);		//rows of real x
	MKL_INT vecs = mxGetN(prhs[1]);		//cols of real x
	MKL_INT ldy = (MKL_INT) mrows;		//rows of output == rows of real A

//	mexPrintf("A = %dx%d\nx = %dx%d\n",mrows,ncols,ldx,vecs);

	int *row_first, *row_last;
	row_first = (int*) mxCalloc(mrows,sizeof(int));
	row_last = (int*) mxCalloc(mrows,sizeof(int));
	if(nrhs != 2)
		mexErrMsgTxt("Two inputs required.");
	if(nlhs != 1)
		mexErrMsgTxt("One output required.");

	if(!(mxIsSparse(prhs[0])))
	{
		mexErrMsgTxt("First Argument is not sparse");
	} else {
		vals = (double *)mxGetData(prhs[0]);
		row_counts = mxGetJc(prhs[0]);
		nnz = (int) row_counts[mrows];
		col_index = mxGetIr(prhs[0]);
	}

	for (int i = 0; i < mrows; i++) 
	{
//		mexPrintf("%d, ",row_counts[i]);
		row_last[i] = (int) row_counts[i+1];
		row_first[i] = (int) row_counts[i];
	}
//	mexPrintf("%d\nAbove are row counts of A\n",row_counts[mrows]);

//	int new_col_index[nnz];
	int* new_col_index = (int*) mxCalloc(nnz,sizeof(int));
	for (int i = 0; i < nnz; i++) 
	{
		new_col_index[i] = (int) col_index[i];
//		mexPrintf("%d, ",new_col_index[i]);
	}
//	mexPrintf("\nAbove are column indices of A\n");

//	for (int i = 0; i < nnz; i++) 
//	{
//		mexPrintf("%f, ",vals[i]);
//	}
//	mexPrintf("\nAbove are the values of A\n");
	
	if(ncols != (int) ldx) {
		mexErrMsgTxt("Matrix dimensions do not match");
	}

	const double* x = (const double*) mxGetData(prhs[1]);

	/* Set up out_array */
	plhs[0] = mxCreateDoubleMatrix(mrows,vecs,mxREAL);
	double* y = (double*) mxGetData(plhs[0]);

//	mexPrintf("Setting up Matrix\n");
	sparse_matrix_t A;
	mkl_sparse_d_create_csr(&A,SPARSE_INDEX_BASE_ZERO,mrows,ncols,
			row_first,row_last,new_col_index,vals);
	matrix_descr descr;
	descr.type = SPARSE_MATRIX_TYPE_GENERAL;
//	mexPrintf("Starting Computation\n");
	mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE,1.0,A,descr,
			SPARSE_LAYOUT_COLUMN_MAJOR,x,vecs,ldx,0.0,y,ldy);

	mxFree(row_first);
	mxFree(row_last);
	mxFree(new_col_index);


}
