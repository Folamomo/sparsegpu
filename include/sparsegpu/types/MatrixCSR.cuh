#ifndef SPARSEGPU_MATRIXCSR_CUH
#define SPARSEGPU_MATRIXCSR_CUH

#include <cstdint>

namespace sparsegpu{

    /**
     * Stores matrices in Compressed Sparse Row-wise format on the Host
     * @tparam Value type of values stored in the matrix
     * @tparam Index type used to store indices and offsets
     */
    template<typename Value, typename Index = int32_t>
    class MatrixCSR{
    public:
        Index rows;
        Index columns;
        Index element_count;

        Value* values;
        Index* column_indices;
        Index* row_offsets;

        /**
         * Constructs an empty CSR matrix with space for element_count elements on the Host
         * Values, column indices and row_offsets are allocated and should be initialized immediately
         * @param rows height of the matrix
         * @param cols width of the matrix
         * @param element_count number of nonzero elements
         */
        MatrixCSR(Index rows, Index columns, Index element_count):
            rows(rows),
            columns(columns),
            element_count(element_count),
            values(new Value[element_count]),
            column_indices(new Index[element_count]),
            row_offsets(new Index[rows + 1]){}


        /**
         * Constructs CSR matrix using previously allocated storage on th GPU
         * @param rows height of the matrix
         * @param cols width of the matrix
         * @param element_count number of nonzero elements
         * @param values pointer to allocated values
         * @param column_indices pointer to allocated column indices
         * @param row_offsets pointer to allocated row offsets
         */

        MatrixCSR(Index rows, Index columns, Index element_count,
                     Value* values, Index* column_indices, Index* row_offsets) noexcept:
        rows(rows),
        columns(columns),
        element_count(element_count),
        values(values),
        column_indices(column_indices),
        row_offsets(row_offsets){
            
        }

        /**
         * Copy constructor
         * @param other
         */
        MatrixCSR(const MatrixCSR<Value, Index>& other):
                MatrixCSR(other.rows, other.columns, other.element_count){
            std::copy(other.values, other.values + element_count, values);
            std::copy(other.column_indices, other.column_indices + element_count, column_indices);
            std::copy(other.row_offsets, other.row_offsets + rows + 1, row_offsets);
        }

        /**
         * Move constructor
         * @param other
         */
        MatrixCSR(MatrixCSR<Value, Index>&& other) noexcept:
                MatrixCSR(other.rows, other.columns, other.element_count,
                             other.values, other.column_indices, other.row_offsets){
            other.values = nullptr;
            other.column_indices = nullptr;
            other.row_offsets = nullptr;
        }

        ~MatrixCSR(){
            delete[](values);
            delete[](column_indices);
            delete[](row_offsets);
        }

        /**
         * Creates an empty matrix
         * @param rows
         * @param columns
         * @return Empty matrix
         */
        static MatrixCSR<Value, Index> empty(Index rows, Index columns){
            MatrixCSR<Value, Index> result{rows, columns, 0};
            std::fill_n(result.row_offsets, rows + 1, Index(0));
            return result;
        }
    };
}

#endif //SPARSEGPU_MATRIXCSR_CUH
