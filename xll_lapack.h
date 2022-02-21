// xll_lapack.h - LAPACK for Excel
#pragma once
#include "xll/xll/xll.h"

// non-owning vector
inline blas::vector<double> fpvector(_FPX* pa)
{
	if (size(*pa) == 1 and pa->array[0] != 0) {
		handle<blas::vector<double>> v(pa->array[0]);
		if (v) {
			return *v.ptr();
		}
	}

	return blas::vector<double>(size(*pa), pa->array, 1);
}
// non-owning matrix
inline blas::matrix<double> fpmatrix(_FPX* pa)
{
	static const std::type_info& fp_array = typeid(_FPX);
	static const std::type_info& fpx_array = typeid(FPX);
	static const std::type_info& blas_matrix = typeid(blas::matrix<double>);

	if (size(*pa) == 1 and pa->array[0] != 0) {
		if (const auto p = to_pointer<_FPX>(pa->array[0]); typeid(*p) == fp_array) {
			return blas::matrix<double>(p->rows, p->columns, p->array);
		}
		if (const auto p = to_pointer<FPX>(pa->array[0]); typeid(*p) == fpx_array) {
			return blas::matrix<double>(p->rows(), p->columns(), p->array());
		}
		if (const auto p = to_pointer<blas::matrix<double>>(pa->array[0]); typeid(*p) == blas_matrix) {
			return *p;
		}
	}

	return blas::matrix<double>(pa->rows, pa->columns, pa->array);
}
