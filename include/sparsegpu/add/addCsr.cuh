#ifndef SPARSEGPU_ADDCSR_CUH
#define SPARSEGPU_ADDCSR_CUH

#include <sparsegpu/types/MatrixCSR.cuh>

namespace sparsegpu{
    /**
     * Calculate sum of two matrices in CSR format
     * @tparam ValueLeft
     * @tparam IndexLeft
     * @tparam ValueRight
     * @tparam IndexRight
     * @param left
     * @param right
     * @param epsilon
     * @return
     */
    template<typename ValueLeft, typename IndexLeft,
            typename ValueRight, typename IndexRight>
    auto addCSR(const MatrixCSR<ValueLeft, IndexLeft>& left,
                const MatrixCSR<ValueRight, IndexRight>& right,
                decltype(ValueLeft(0) + ValueRight(0)) epsilon = 0
                ) ->
        MatrixCSR<decltype(ValueLeft(0) + ValueRight(0)),
                decltype(IndexLeft(0) + IndexRight(0))> {
        using Value = decltype(ValueLeft(0) + ValueRight(0));
        using Index = decltype(IndexLeft(0) + IndexRight(0));

        Index count(0);
        Index* row_offsets = new Index[left.rows + 1];

        for(Index row(0); row < left.rows; row++){
            Index leftOffset = left.row_offsets[row];
            Index leftTo = left.row_offsets[row + 1];

            Index rightOffset = right.row_offsets[row];
            Index rightTo = right.row_offsets[row + 1];

            //count elements of merged value arrays
            while (leftOffset < leftTo && rightOffset < rightTo) {
                Index leftColumn = right.column_indices[leftOffset];
                Index rightColumn = left.column_indices[rightOffset];
                count++;

                if (leftColumn <= rightColumn) leftOffset++;
                if (leftColumn >= rightColumn) rightOffset++;
                if (leftColumn == rightColumn && abs(left.values[leftOffset] + right.values[rightOffset]) > epsilon) count--;
            }

            //add remaining elements
            count += leftTo - leftOffset;
            count += rightTo - rightOffset;

            row_offsets[row + 1] = count;
        }

        Index* column_indices = new Index [count];
        Value * values = new Value [count];

        for(Index row(0); row < left.rows; row++){
            Index leftOffset = left.row_offsets[row];
            Index leftTo = left.row_offsets[row + 1];

            Index rightOffset = right.row_offsets[row];
            Index rightTo = right.row_offsets[row + 1];

            while (leftOffset < leftTo && rightOffset < rightTo) {
                Index leftColumn = right.column_indices[leftOffset];
                Index rightColumn = left.column_indices[rightOffset];

                if (leftColumn < rightColumn){
                    values[count] = left.values[leftOffset];
                    column_indices[count] = leftColumn;
                    leftOffset++;
                    count++;
                } else if (leftColumn > rightColumn){
                    values[count] = right.values[rightOffset];
                    column_indices[count] = rightColumn;
                    rightOffset++;
                    count++;
                } else {
                    Value sum = left.values[leftOffset] + right.values[rightOffset];
                    if (sum != 0){
                        values[count] = sum;
                        column_indices[count] = leftColumn;
                        count++;
                    }
                    leftOffset++;
                    rightOffset++;
                }
            }

            while (leftOffset < leftTo){
                values[count] = left.values[leftOffset];
                column_indices[count] = left.column_indices[leftOffset];
                leftOffset++;
                count++;
            }

            while (rightOffset < rightTo){
                values[count] = right.values[rightOffset];
                column_indices[count] = right.columnIndices[rightOffset];
                rightOffset++;
                count++;
            }
        }


        return MatrixCSR<Value, Index>{left.rows, left.columns, count,
                                       values,
                                       column_indices,
                                       row_offsets};
    }

    template<typename ValueLeft, typename IndexLeft,
            typename ValueRight, typename IndexRight>
    auto operator + (const MatrixCSR<ValueLeft, IndexLeft>& left,
                    const MatrixCSR<ValueRight, IndexRight>& right){
        return addCSR(left, right);
    }
}


#endif //SPARSEGPU_ADDCSR_CUH
