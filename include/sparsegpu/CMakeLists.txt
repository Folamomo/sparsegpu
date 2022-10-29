target_sources(sparsegpu
        PUBLIC
        MatrixCSRDev.cuh
        MatrixCSR.cuh
        )
add_subdirectory(transfer)