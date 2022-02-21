// xll_lapack.h - LAPACK for Excel
#pragma once
#include "fms_blas/fms_blas.h"
#include "fms_blas/fms_lapack.h"
#include "xll/xll/xll.h"

namespace xll {

	inline const std::type_info& fp_array = typeid(_FPX);
	inline const std::type_info& fpx_array = typeid(FPX);
	inline const std::type_info& blas_vector = typeid(blas::vector<double>);
	inline const std::type_info& blas_matrix = typeid(blas::matrix<double>);

	// non-owning vector
	inline blas::vector<double> fpvector(_FPX* pa)
	{
		if (size(*pa) == 1 and pa->array[0] != 0) {
			if (const auto p = to_pointer<_FPX>(pa->array[0]); typeid(*p) == fp_array) {
				return blas::vector<double>(size(*p), p->array);
			}
			if (const auto p = to_pointer<FPX>(pa->array[0]); typeid(*p) == fpx_array) {
				return blas::vector<double>(p->size(), p->array());
			}
			if (const auto p = to_pointer<blas::vector<double>>(pa->array[0]); typeid(*p) == blas_vector) {
				return *p;
			}
		}

		return blas::vector<double>(size(*pa), pa->array);
	}

	// non-owning matrix
	inline blas::matrix<double> fpmatrix(_FPX* pa)
	{
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

}