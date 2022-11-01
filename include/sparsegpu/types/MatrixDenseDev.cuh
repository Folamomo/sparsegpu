
#ifndef SPARSEGPU_MATRIXDENSEDEV_CUH
#define SPARSEGPU_MATRIXDENSEDEV_CUH

#include <thrust/fill.h>

namespace sparsegpu {
    template<typename Value, typename Index>
    class MatrixDenseDev {
    public:
        Index rows;
        Index columns;
        Value *data;
    public:
        /**
         * Constructs matrix of given dimensions on the GPU. data should be initialized immediately.
         * @param rows
         * @param columns
         */
        MatrixDenseDev(Index rows, Index columns) : rows(rows), columns(columns) {
            cudaMalloc(&data, sizeof(Value) * columns * rows);
        }

        /**
         * Constructs matrix using preallocated storage on the GPU
         * @param rows
         * @param columns
         * @param data
         */
        MatrixDenseDev(Index rows, Index columns, Value *data) : rows(rows), columns(columns), data(data) {}


        /**
         * Copy constructor
         * @param other
         */
        MatrixDenseDev(const MatrixDenseDev<Value, Index> &other) :
                MatrixDenseDev(other.rows, other.columns) {
            cudaMemcpy(data, other.data, sizeof(Value) * columns * rows);
        }

        /**
         * Move constructor
         * @param other
         */
        MatrixDenseDev(MatrixDenseDev<Value, Index> &&other) noexcept:
                MatrixDenseDev(rows, columns, data) {
            other.data = nullptr;
        }

        __device__ Value &operator()(Index row, Index column) {
            return data[row * this->cols + column];
        }

        ~MatrixDenseDev() {
            cudaFree(data);
        }

        static MatrixDenseDev<Index, Value> empty(Index rows, Index columns) {
            auto result = MatrixDenseDev<Index, Value>{rows, columns};
            thrust::fill_n(result.data, rows * columns, Value(0));
        }
    };
}

#endif //SPARSEGPU_MATRIXDENSEDEV_CUH
