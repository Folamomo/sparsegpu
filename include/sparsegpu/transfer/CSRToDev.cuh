#ifndef SPARSEGPU_CSRTODEV_CUH
#define SPARSEGPU_CSRTODEV_CUH

#include <sparsegpu/types/MatrixCSR.cuh>
#include <sparsegpu/types/MatrixCSRDev.cuh>

namespace sparsegpu{
    template <typename Value, typename Index>
    sparsegpu::MatrixCSRDev<Value, Index> toDev(const MatrixCSR<Value, Index>& host){
        auto result = MatrixCSRDev<Value, Index>(host.rows, host.columns, host.element_count);
        cudaMemcpy(result.values, host.values, result.element_count, cudaMemcpyKind::cudaMemcpyHostToDevice);
        cudaMemcpy(result.column_indices, host.column_indices, result.element_count, cudaMemcpyKind::cudaMemcpyHostToDevice);
        cudaMemcpy(result.row_offsets, host.row_offsets, result.rows + 1, cudaMemcpyKind::cudaMemcpyHostToDevice);
        return result;
    }
}

#endif //SPARSEGPU_CSRTODEV_CUH
