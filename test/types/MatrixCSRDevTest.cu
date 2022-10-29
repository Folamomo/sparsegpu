#include <sparsegpu/MatrixCSRDev.cuh>
#include <sparsegpu/MatrixCSR.cuh>
#include "sparsegpu/transfer/CSRToDev.cuh"
#include "sparsegpu/transfer/CSRToHost.cuh"

int main(){
    sparsegpu::MatrixCSR<double> host{1, 1, 0};
    auto dev = sparsegpu::toDev(host);
    sparsegpu::MatrixCSRDev<double> d{1, 1, 0};
    return 0;
}
