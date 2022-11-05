#ifndef SPARSEGPU_ADDCSRDENSE_CUH
#define SPARSEGPU_ADDCSRDENSE_CUH

#include <sparsegpu/types/MatrixCSR.cuh>
#include <sparsegpu/types/MatrixDense.cuh>

namespace sparsegpu{
    template<typename ValueLeft, typename IndexLeft,
            typename ValueRight, typename IndexRight>
    auto addCSR(const MatrixCSR<ValueLeft, IndexLeft>& csr,
                const MatrixDense<ValueRight, IndexRight>& dense) ->
    MatrixDense<decltype(ValueLeft(0) + ValueRight(0)),
            decltype(IndexLeft(0) + IndexRight(0))> {
        using Value = decltype(ValueLeft(0) + ValueRight(0));
        using Index = decltype(IndexLeft(0) + IndexRight(0));

        MatrixDense<Value, Index> result{dense.rows, dense.columns};

        if (std::is_same_v<ValueRight,  Value>) {
            std::copy(dense.data, dense.data + dense.rows * dense.columns, result.data);
        } else {
            for (Index i(0); i < dense.rows * dense.columns; ++i){
                result.data[i] = Value(dense.data[i]);
            }
        }

        for(Index row(0); row < dense.rows; row++){
            for(Index element(csr.row_offsets[row]); element < csr.row_offsets[row + 1]; element++){
                result.data[row * dense.rows + csr.column_indices[element]] += csr.values[element];
            }
        }

        return result;
    }

    template<typename ValueLeft, typename IndexLeft,
            typename ValueRight, typename IndexRight>
    auto operator + (const MatrixCSR<ValueLeft, IndexLeft>& csr,
                     const MatrixDense<ValueRight, IndexRight>& dense){
        return MatrixDense(csr, dense);
    }

    template<typename ValueLeft, typename IndexLeft,
            typename ValueRight, typename IndexRight>
    auto operator + (const MatrixDense<ValueRight, IndexRight>& dense,
                     const MatrixCSR<ValueLeft, IndexLeft>& csr){
        return MatrixDense(csr, dense);
    }
}

#endif //SPARSEGPU_ADDCSRDENSE_CUH
