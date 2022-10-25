#ifndef SPARSEGPU_MATRIXCSRDEV_CUH
#define SPARSEGPU_MATRIXCSRDEV_CUH

#include <cstdint>

namespace sparsegpu{

    /**
     * Stores matrices in Compressed Sparse Row-wise format on the GPU
     * @tparam Value type of values stored in the matrix
     * @tparam Index type used to store indices and offsets
     */
    template<typename Value, typename Index = uint32_t>
    class MatrixCSRDev{
    public:
        Value* values;
        Index* column_indices;
        Index* row_offsets;

        Index element_count;
        Index rows;
        Index columns;

        /**
         * Constructs an empty CSR matrix with space for element_count elements on the GPU
         * Values, column indices and row_offsets are allocated and should be initialized immediately
         * @param rows height of the matrix
         * @param cols width of the matrix
         * @param element_count number of nonzero elements
         */
        MatrixCSRDev(Index rows, Index cols, Index element_count):
            rows(rows),
            columns(columns),
            element_count(element_count)
        {
            cudaMalloc(&values, element_count * sizeof (Value));
            cudaMalloc(&column_indices, element_count * sizeof(Index));
            cudaMalloc(&row_offsets, (rows + 1)* sizeof(Index));
        }

        /**
         * Constructs CSR matrix using previously allocated storage on th GPU
         * @param rows height of the matrix
         * @param cols width of the matrix
         * @param element_count number of nonzero elements
         * @param values pointer to allocated values
         * @param column_indices pointer to allocated column indices
         * @param row_offsets pointer to allocated row offsets
         */

        MatrixCSRDev(Index rows, Index cols, Index element_count,
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
        MatrixCSRDev(const MatrixCSRDev<Value, Index>& other):
                MatrixCSRDev(other.rows, other.columns, other.element_count){
            cudaMemcpy(values, other.values, element_count * sizeof(Value), cudaMemcpyKind::cudaMemcpyDeviceToDevice);
            cudaMemcpy(column_indices, other.columnIndices, element_count * sizeof(Index), cudaMemcpyKind::cudaMemcpyDeviceToDevice);
            cudaMemcpy(row_offsets, other.rowOffsets, (other.rows + 1) * sizeof(Index), cudaMemcpyKind::cudaMemcpyDeviceToDevice);
        }

        /**
         * Move constructor
         * @param other
         */
        MatrixCSRDev(MatrixCSRDev<Value, Index>&& other) noexcept:
                MatrixCSRDev(other.rows, other.cols, other.size,
                             other.values, other.column_indices, other.row_offsets){
            other.values = nullptr;
            other.columnIndices = nullptr;
            other.rowOffsets = nullptr;
        }

        ~MatrixCSRDev(){
            cudaFree(values);
            cudaFree(column_indices);
            cudaFree(row_offsets);
        }

        /**
         * Creates an empty matrix
         * @param rows
         * @param columns
         * @return Empty matrix
         */
        static MatrixCSRDev<Value, Index> empty(Index rows, Index columns){
            return MatrixCSRDev<Value, Index>(rows, columns, 0);
        }
    };
}

#endif //SPARSEGPU_MATRIXCSRDEV_CUH
