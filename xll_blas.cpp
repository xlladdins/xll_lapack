// xll_blas.cpp - BLAS wrappers
#include "xll_lapack.h"

using namespace xll;

AddIn xai_blas_vector_(
	Function(XLL_HANDLEX, "xll_blas_vector_", "\\BLAS.VECTOR")
	.Arguments({
		Arg(XLL_FPX, "v", "is an array of vector elments."),
		})
		.Uncalced()
	.Category("BLAS")
	.FunctionHelp("Return a handle to a BLAS vector.")
);
HANDLEX WINAPI xll_blas_vector_(_FPX* pv)
{
#pragma XLLEXPORT
	HANDLEX result = INVALID_HANDLEX;

	try {
		handle<blas::vector<double>> h(new blas::vector_alloc<double>(size(*pv), pv->array, 1));
		ensure(h);
		result = h.get();
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());
	}
	catch (...) {
		XLL_ERROR(__FUNCTION__ ": unknown exception");
	}

	return result;
}

AddIn xai_blas_matrix_(
	Function(XLL_HANDLEX, "xll_blas_matrix_", "\\BLAS.MATRIX")
	.Arguments({
		Arg(XLL_FPX, "v", "is an array of matrix elments."),
		Arg(XLL_BOOL, "_trans", "is an optional boolean indicating the matrix is transposed. Default is false.")
		})
	.Uncalced()
	.Category("BLAS")
	.FunctionHelp("Return a handle to a BLAS matrix.")
);
HANDLEX WINAPI xll_blas_matrix_(_FPX* pv, BOOL t)
{
#pragma XLLEXPORT
	HANDLEX result = INVALID_HANDLEX;

	try {
		handle<blas::matrix<double>> h(new blas::matrix_alloc<double>(pv->rows, pv->columns, pv->array, t ? CblasTrans : CblasNoTrans));
		ensure(h);
		result = h.get();
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());
	}
	catch (...) {
		XLL_ERROR(__FUNCTION__ ": unknown exception");
	}

	return result;
}

AddIn xai_blas_trans(
	Function(XLL_HANDLEX, "xll_blas_trans", "BLAS.TRANS")
	.Arguments({
		Arg(XLL_FPX, "m", "is a matrix."),
		})
		.Uncalced()
	.Category("BLAS")
	.FunctionHelp("Return a temporary handle to the transpose of a BLAS matrix.")
);
HANDLEX WINAPI xll_blas_trans(_FPX* pa)
{
#pragma XLLEXPORT
	HANDLEX result = INVALID_HANDLEX;

	try {
		blas::matrix<double> a = fpmatrix(pa);
		a = transpose(a);
		result = to_handle<blas::matrix<double>>(&a);
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());
	}
	catch (...) {
		XLL_ERROR(__FUNCTION__ ": unknown exception");
	}

	return result;
}

AddIn xai_blas_gemm(
	Function(XLL_FPX, "xll_blas_gemm", "BLAS.GEMM")
	.Arguments({
		Arg(XLL_FPX, "a", "is a matrix."),
		Arg(XLL_FPX, "b", "is a matrix."),
		Arg(XLL_FPX, "c", "is an optional matrix."),
		Arg(XLL_DOUBLE, "_alpha", "is an optional scaling factor. Default is 1."),
		Arg(XLL_DOUBLE, "_beta", "is an optional factor. Default is 0.")
		})
	.Category("BLAS")
	.FunctionHelp("Return the matrix product alpha a b + beta c.")
	.Documentation(R"(
Compute the matrix product \(c_{i,j} = \sum_k a_{i,k} b_{k,j}\).
)")
);
_FPX* WINAPI xll_blas_gemm(_FPX* pa, _FPX* pb, _FPX* pc, double alpha, double beta)
{
#pragma XLLEXPORT
	static FPX c;

	try {
		if (alpha == 0) {
			alpha = 1;
		}
		auto a = fpmatrix(pa);
		auto b = fpmatrix(pb);
		blas::matrix<double> c_;
		if (size(*pc) == 1 and pc->array[0] == 0) {
			c.resize(a.rows(), b.columns());
			pc = c.get();
		}
		c_ = fpmatrix(pc);

		blas::gemm(a, b, c_, alpha, beta);
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());

		return nullptr;
	}

	return pc;
}

AddIn xai_blas_tpmv(
	Function(XLL_FPX, "xll_blas_tpmv", "BLAS.TPMV")
	.Arguments({
		Arg(XLL_FPX, "a", "is a packed triangular matrix."),
		Arg(XLL_FPX, "x", "is a vector."),
		Arg(XLL_BOOL, "_upper", "indicates a is upper. Default is false"),
		Arg(XLL_BOOL, "_trans", "indicates a is transposed. Default is false"),
		})
		.Category("BLAS")
	.FunctionHelp("Return the product of packed triangular matrix a and vector x.")
	.Documentation(R"(
A triangular matrix is a square matrix that is either lower, with entries above the diagonal equal to 0, or
upper, with entries below the main diagonal equal to 0. 
)")
.SeeAlso({ "PACK", "UNPACK" })
);
_FPX* WINAPI xll_blas_tpmv(_FPX* pa, _FPX* px, BOOL upper, BOOL trans)
{
#pragma XLLEXPORT
	static FPX c;

	try {
		auto n = size(*px);
		c.resize(px->rows, px->columns);
		auto c_ = fpvector(c.get());
		c_.copy(n, px->array);
		blas::tp<double> a(blas::matrix(n, pa->array), upper ? CblasUpper : CblasLower, CblasNonUnit);
		if (trans) {
			blas::tpmv(transpose(a), c_);
		}
		else {
			blas::tpmv(a, c_);
		}
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());

		return nullptr;
	}

	return c.get();
}

AddIn xai_pack(
	Function(XLL_FPX, "xll_pack", "PACK")
	.Arguments({
		Arg(XLL_FPX, "A", "is a square matrix."),
		Arg(XLL_BOOL, "_upper", "is an optional argument indicating upper trangle of A is used. Default is lower.")
		})
	.FunctionHelp("Pack lower or upper triangle of A.")
	.Category("BLAS")
	.Documentation(R"(
Pack lower \([a_{ij}\) as \([a_{00}, a_{10}, a_{11}, a_{20}, a_{21}, a_{22},\ldots]\)
and upper as \([a_{00}, a_{01}, a_{11}, a_{02}, a_{12}, a_{22},\ldots]\).
)")
.SeeAlso({ "UNPACK" })
);
_FPX* WINAPI xll_pack(_FPX* pa, BOOL upper)
{
#pragma XLLEXPORT
	static FPX l;

	int n = pa->rows;
	if (n != pa->columns) {
		XLL_ERROR(__FUNCTION__ ": matrix must be square");

		return nullptr;
	}

	l.resize(1, (n * (n + 1)) / 2);

	if (upper) {
		blas::packu(n, pa->array, l.array());
	}
	else {
		blas::packl(n, pa->array, l.array());
	}

	return l.get();
}

AddIn xai_unpack(
	Function(XLL_FPX, "xll_unpack", "UNPACK")
	.Arguments({
		Arg(XLL_FPX, "L", "is a packed matrix."),
		Arg(XLL_LONG, "_ul", "is an optional upper (>0) or lower (<0) flag. Default is 0.")
		})
	.FunctionHelp("Unpack L into symmetric (ul = 0), upper triangular (ul > 0), or lower triangular (ul < 0) A.")
	.Category("BLAS")
	.Documentation(R"(
Restore a packed matrix to its full form.
)")
.SeeAlso({ "PACK" })
);
_FPX* WINAPI xll_unpack(_FPX* pl, long ul)
{
#pragma XLLEXPORT
	static FPX a;
	int m = size(*pl);

	// m = n(n+1)/2
	// n^2 + n - 2m = 0
	// b^2 - 4ac = 1 + 8m
	auto d = sqrt(1 + 8 * m);
	int n = static_cast<int>((-1 + d) / 2);
	a.resize(n, n);
	fpmatrix(a.get()).fill(0);

	if (ul > 0) {
		blas::unpacku(n, pl->array, a.array());
	}
	else if (ul < 0) {
		blas::unpackl(n, pl->array, a.array());
	}
	else {
		blas::unpacks(n, pl->array, a.array());
	}

	return a.get();
}
