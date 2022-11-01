target_sources(sparsegpu
        PUBLIC
        csr.h
        dense.h
        )
add_subdirectory(transfer)
add_subdirectory(types)
add_subdirectory(conversions)