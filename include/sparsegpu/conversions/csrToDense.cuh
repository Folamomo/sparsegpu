
#ifndef SPARSEGPU_CSRTODENSE_CUH
#define SPARSEGPU_CSRTODENSE_CUH

#include <sparsegpu/types/MatrixCSR.cuh>
#include <sparsegpu/types/MatrixDense.cuh>

namespace sparsegpu{
    template<typename Value, typename Index>
    MatrixDense<Value, Index> toDense(const MatrixCSR<Value, Index> &csr){
        MatrixDense<Value, Index> result{csr.rows, csr.columns};
        for(Index row(0); row < csr.rows; ++row){
            for(Index column(0); column < csr.columns; column++){
                result.data[row* result.columns + column] = Value(0);
            }
            for(Index element = csr.row_offsets[row]; element < csr.row_offsets[row + 1]; ++element){
                result.data[row * result.columns + csr.column_indices[element]] = csr.values[element];
            }
        }
        return result;
    }
}

#endif //SPARSEGPU_CSRTODENSE_CUH
