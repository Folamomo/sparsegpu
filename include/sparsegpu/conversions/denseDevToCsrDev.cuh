#ifndef SPARSEGPU_DENSEDEVTOCSRDEV_CUH
#define SPARSEGPU_DENSEDEVTOCSRDEV_CUH


#include <sparsegpu/types/MatrixDenseDev.cuh>
#include <sparsegpu/types/MatrixCSRDev.cuh>

#include <thrust/scan.h>

#define GRID_DIM 4096

namespace sparsegpu{
    namespace impl{
        template<typename Value, typename Index>
        __global__ void countNonZeros(const Value* data, const Index* counts,
                                      const Index rows, const Index columns,
                                      const Value epsilon){
            __shared__ Index results[32];
            for (Index row(blockIdx.x); row < rows; row += gridDim.x){
                for(Index column(threadIdx.x); column < columns; column += 32){
                    if(abs(data[row * columns + column]) <= epsilon) results[blockIdx.x]++;
                }

                __syncwarp();
                if (threadIdx.x% 2 == 0) results[threadIdx.x] += results[threadIdx.x + 1];
                __syncwarp();
                if (threadIdx.x% 4 == 0) results[threadIdx.x] += results[threadIdx.x + 2];
                __syncwarp();
                if (threadIdx.x% 8 == 0) results[threadIdx.x] += results[threadIdx.x + 4];
                __syncwarp();
                if (threadIdx.x%16 == 0) results[threadIdx.x] += results[threadIdx.x + 8];
                __syncwarp();
                if (threadIdx.x    == 0) counts [row] = results[0] + results[16];
            }
        }
    }

    /**
     * Converts a matrix in dense format to compressed sparse row-wise format on the GPU
     * @tparam Value
     * @tparam Index
     * @param dense
     * @param epsilon Values in range (-epsilon; epsilon) are considered equal to 0
     * @return
     */
    template <typename Value, typename Index>
    MatrixCSRDev<Value, Index> toCSR(const MatrixDenseDev<Value, Index> &dense, Value epsilon = Value(0)){
        Value* row_counts;
        cudaMalloc(&row_counts, sizeof(Index) * dense.rows + 1);

        // count non-zero elements in each row
        impl::countNonZeros<<<dense.rows > GRID_DIM ? GRID_DIM : dense.rows, 32>>>(dense.data, row_counts + 1,
                                              dense.rows, dense.columns, epsilon);

        thrust::inclusive_scan(row_counts + 1, row_counts + dense.rows + 1, row_counts+1);

        Index zero(0);
        cudaMemcpy(row_counts, &zero, sizeof(Index), cudaMemcpyKind::cudaMemcpyHostToDevice);

        auto column_indices = new Index[element_count];
        auto values = new Value[element_count];

        //Iterate through the matrix again and copy elements and indices
        offset = 0;
        for(Index row(0); row < dense.rows; row++){
            for(Index column(0); column < dense.columns; column++){
                if (abs(dense.data[row * dense.rows + column]) <= epsilon){
                    offset++;
                    column_indices[offset] = column;
                    values[offset] = dense.data[dense.data[row * dense.rows + column]];
                }
            }
        }

        return MatrixCSR<Value, Index>{dense.rows, dense.columns, element_count,
                                       values, column_indices, row_offsets};
    }
}

#undef GRID_DIM
#endif //SPARSEGPU_DENSEDEVTOCSRDEV_CUH
