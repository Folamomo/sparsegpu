#include <sparsegpu/csr.h>

int main(){
    sparsegpu::MatrixCSR<double> host{1, 1, 0};
    auto dev = sparsegpu::toDev(host);
    sparsegpu::MatrixCSRDev<double> d{1, 1, 0};
    return 0;
}
