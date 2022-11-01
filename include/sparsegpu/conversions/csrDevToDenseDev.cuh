#ifndef SPARSEGPU_CSRDEVTODENSEDEV_CUH
#define SPARSEGPU_CSRDEVTODENSEDEV_CUH

#include <sparsegpu/types/MatrixCSRDev.cuh>
#include <sparsegpu/types/MatrixDenseDev.cuh>

#define GRID_DIM 0x10000

namespace sparsegpu{
    namespace impl{
        template<typename Value, typename Index>
        __global__ void csrToDenseKernel(
                Value* data,
                const Index* row_offsets, const Index* column_indices, const Value* values,
                const Index rows, const Index columns
                ){
            for (Index row = blockIdx.x; row < rows; row += GRID_DIM){
                //clear row
                for(Index column(0); column < columns; column += 32){
                    data[row * columns + column] = Index(0);
                }
                __syncwarp();
                //set non-zero values
                for(Index element = row_offsets[row]; element < row_offsets[row + 1]; element += 32){
                    data[row * columns + column_indices[element]] = values[element];
                }
            }
        }
    }
    template<typename Value, typename Index>
    MatrixDenseDev<Value, Index> toDense(const MatrixCSRDev<Value, Index> &csr){
        MatrixDenseDev<Value, Index> result{csr.rows, csr.columns};
        impl::csrToDenseKernel<<<GRID_DIM, 32>>>(
                result.data,
                csr.row_offsets,
                csr.column_indices,
                csr.values,
                csr.rows,
                csr.columns
                );
        return result;
    }
}

#undef GRID_DIM
#endif //SPARSEGPU_CSRDEVTODENSEDEV_CUH
