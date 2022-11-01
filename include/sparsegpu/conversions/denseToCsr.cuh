#ifndef SPARSEGPU_DENSETOCSR_CUH
#define SPARSEGPU_DENSETOCSR_CUH

#include <sparsegpu/types/MatrixDense.cuh>
#include <sparsegpu/types/MatrixCSR.cuh>

namespace sparsegpu{

    /**
     * Converts a matrix in dense format to compressed sparse row-wise format
     * @tparam Value
     * @tparam Index
     * @param dense
     * @param epsilon Values in range (-epsilon; epsilon) are considered equal to 0
     * @return
     */
    template <typename Value, typename Index>
    MatrixCSR<Value, Index> toCSR(const MatrixDense<Value, Index> &dense, Value epsilon = Value(0)){
        auto row_offsets = new Index[dense.rows + 1];

        //Iterate through the matrix once and count non-zero elements in each row
        Index offset = 0;
        row_offsets[0] = Index(0);
        for(Index row(0); row < dense.rows; row++){
            for(Index column(0); column < dense.columns; column++){
                if (abs(dense.data[row * dense.rows + column]) <= epsilon){
                    offset++;
                }
            }
            row_offsets[row + 1] = offset;
        }

        Index element_count = offset;

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

#endif //SPARSEGPU_DENSETOCSR_CUH
