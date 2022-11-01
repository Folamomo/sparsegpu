target_sources(sparsegpu
        PUBLIC
        csrToDense.cuh
        csrDevToDenseDev.cuh
        denseToCsr.cuh
        denseDevToCsrDev.cuh
        )