add_executable(MatrixCSRDevTest MatrixCSRDevTest.cu)
target_link_libraries(MatrixCSRDevTest sparsegpu)

add_test(NAME "MatrixCSRDevTest" COMMAND MatrixCSRDevTest)