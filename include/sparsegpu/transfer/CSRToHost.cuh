#ifndef SPARSEGPU_CSRTOHOST_CUH
#define SPARSEGPU_CSRTOHOST_CUH

#include <sparsegpu/types/MatrixCSR.cuh>
#include <sparsegpu/types/MatrixCSRDev.cuh>

namespace sparsegpu{
    template <typename Value, typename Index>
    sparsegpu::MatrixCSR<Value, Index> toHost(const MatrixCSRDev<Value, Index>& dev){
        auto result = MatrixCSR<Value, Index>(dev.rows, dev.columns, dev.element_count);
        cudaMemcpy(result.values, dev.values, result.element_count, cudaMemcpyKind::cudaMemcpyDeviceToHost);
        cudaMemcpy(result.column_indices, dev.column_indices, result.element_count, cudaMemcpyKind::cudaMemcpyDeviceToHost);
        cudaMemcpy(result.row_offsets, dev.row_offsets, result.rows + 1, cudaMemcpyKind::cudaMemcpyDeviceToHost);
        return result;
    }
}

#endif //SPARSEGPU_CSRTOHOST_CUH
