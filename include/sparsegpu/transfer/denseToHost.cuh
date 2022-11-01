#ifndef SPARSEGPU_DENSETOHOST_CUH
#define SPARSEGPU_DENSETOHOST_CUH

#include <sparsegpu/types/MatrixDense.cuh>
#include <sparsegpu/types/MatrixDenseDev.cuh>

namespace sparsegpu{
    template <typename Value, typename Index>
    MatrixDense<Value, Index> toHost(const MatrixDenseDev<Value, Index>& dev){
        auto result = MatrixDense<Value, Index>(dev.rows, dev.columns, dev.element_count);
        cudaMemcpy(result.data, dev.data, dev.rows * dev.columns, cudaMemcpyKind::cudaMemcpyHostToDevice);
        return result;
    }
}

#endif //SPARSEGPU_DENSETOHOST_CUH
