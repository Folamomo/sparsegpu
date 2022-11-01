#ifndef SPARSEGPU_DENSETODEV_CUH
#define SPARSEGPU_DENSETODEV_CUH

#include <sparsegpu/types/MatrixDense.cuh>
#include <sparsegpu/types/MatrixDenseDev.cuh>

namespace sparsegpu{
    template <typename Value, typename Index>
    MatrixDenseDev<Value, Index> toDev(const MatrixDense<Value, Index> &host) {
        auto result = MatrixDenseDev<Value, Index>(host.rows, host.columns);
        cudaMemcpy(result.data, host.data, host.rows * host.columns, cudaMemcpyKind::cudaMemcpyHostToDevice);
        return result;
    }
}

#endif //SPARSEGPU_DENSETODEV_CUH
