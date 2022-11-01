#ifndef SPARSEGPU_MATRIXDENSE_CUH
#define SPARSEGPU_MATRIXDENSE_CUH

namespace sparsegpu {
    template<typename Value, typename Index>
    class MatrixDense {
    public:
        Index rows;
        Index columns;
        Value *data;
    public:
        /**
         * Constructs matrix of given dimensions. data should be initialized immediately.
         * @param rows
         * @param columns
         */
        MatrixDense(Index rows, Index columns) : rows(rows), columns(columns) {
            data = new Value[columns * rows];
        }

        /**
         * Constructs matrix using preallocated storage
         * @param rows
         * @param columns
         * @param data
         */
        MatrixDense(Index rows, Index columns, Value *data) : rows(rows), columns(columns), data(data) {}


        /**
         * Copy constructor
         * @param other
         */
        MatrixDense(const MatrixDense<Value, Index> &other) :
                MatrixDense(other.rows, other.columns) {
            std::copy(other.data, other.data + rows * columns, data);
        }

        /**
         * Move constructor
         * @param other
         */
        MatrixDense(MatrixDense<Value, Index> &&other) noexcept:
                MatrixDense(rows, columns, data) {
            other.data = nullptr;
        }

        Value &operator()(Index row, Index column) {
            return data[row * this->cols + column];
        }

        ~MatrixDense() {
            delete[] data;
        }

        static MatrixDense<Value, Index> empty(Index rows, Index columns){
            MatrixDense<Value, Index> result{rows, columns};
            std::fill_n(result.data, rows * columns, Value(0));
            return result;
        }
    };
}

#endif //SPARSEGPU_MATRIXDENSE_CUH
