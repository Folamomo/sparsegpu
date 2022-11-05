#ifndef SPARSEGPU_DENSEDEVTOCSRDEV_CUH
#define SPARSEGPU_DENSEDEVTOCSRDEV_CUH


#include <sparsegpu/types/MatrixDenseDev.cuh>
#include <sparsegpu/types/MatrixCSRDev.cuh>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>

#define GRID_DIM 4096

namespace sparsegpu{
    namespace impl{
        template<typename Value, typename Index>
        __global__ void countNonZeros(const Value* data, const Index* counts,
                                      const Index rows, const Index columns,
                                      const Value epsilon){
            for (Index row(blockIdx.x); row < rows; row += gridDim.x){
                Index elements_in_row(0);
                for(Index column(0); column < columns; column += blockDim.x){
                    int notZero = column + threadIdx.x < columns && abs(data[row * columns + column]) > epsilon;
                    elements_in_row += __syncthreads_count(notZero);
                }
                if (threadIdx.x == 0){
                    counts[row+1] = elements_in_row;
                }
            }
        }



        template<typename Value, typename Index>
        __global__ void copyElementsAndColumnIndices(
                const Value* data,
                const Index* row_offsets,
                const Index* column_indices,
                const Value* values,
                const Index rows, const Index columns,
                const Value epsilon){
            for (Index row(blockIdx.x); row < rows; row += gridDim.x){
                Index result_row_offset = row_offsets[row];
                const Index next_row_offset = row_offsets[row+1];
                Index left_column(0);

                //until all non-zeros in a row are found
                while (result_row_offset < next_row_offset){
                    Index column = left_column + threadIdx.x;

                    int predicate = column < columns && (data[row * columns + column]) > epsilon;
                    unsigned ballot = __ballot_sync(0xffffffff, predicate);

                    if (predicate) {
                        //count how many non-zeros were found to the left of this thread
                        int prev_zeros = __popcll(ballot & (0xffffffff << (warpSize - threadIdx.x)));
                        Index offset = result_row_offset + prev_zeros;
                        column_indices[offset] = column;
                        values[offset] = data[row * columns + column];
                    }

                    //move result pointer by how many non-zeros were found
                    result_row_offset += __popcll(ballot);
                    left_column += warpSize;
                }
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
        cudaMalloc(&row_counts, sizeof(Index) * (dense.rows + 1));

        // count non-zero element_count in each row
        impl::countNonZeros<<<dense.rows > GRID_DIM ? GRID_DIM : dense.rows, 512>>>(dense.data, row_counts + 1,
                                              dense.rows, dense.columns, epsilon);

        //compute partial sum of row counts in-place
        thrust::device_ptr<Index> row_counts_thrust(row_counts);
        thrust::inclusive_scan(row_counts_thrust + 1, row_counts_thrust + dense.rows + 1, row_counts_thrust + 1);

        //set first element in row_count to 0
        Index zero(0);
        cudaMemcpy(row_counts, &zero, sizeof(Index), cudaMemcpyKind::cudaMemcpyHostToDevice);

        //last element of row_counts now contains the count of all elements
        Index element_count;
        cudaMemcpy(&element_count, &(row_counts[dense.rows]), sizeof(Index), cudaMemcpyKind::cudaMemcpyDeviceToHost);

        //allocate space for indices and values on the gpu
        Index* column_indices;
        Value* values;
        cudaMalloc(&column_indices, sizeof(Index) * element_count.rows );
        cudaMalloc(&values, sizeof(Value) * element_count.rows );

        //Iterate through the matrix again and copy element_count and indices
        impl::copyElementsAndColumnIndices<<<dense.rows > GRID_DIM ? GRID_DIM : dense.rows, warpSize>>>(
                dense.data,
                row_counts,
                column_indices,
                values,
                dense.rows,
                dense.columns,
                epsilon
                );

        return MatrixCSRDev<Value, Index>{dense.rows, dense.columns, element_count,
                                       values, column_indices, row_counts};
    }
}

#undef GRID_DIM
#endif //SPARSEGPU_DENSEDEVTOCSRDEV_CUH
