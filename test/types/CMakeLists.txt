add_executable(MatrixCSRDevTest MatrixCSRDevTest.cu)
target_link_libraries(MatrixCSRDevTest sparsegpu)

add_test(NAME "MatrixCSRDevTest" COMMAND MatrixCSRDevTest)


add_executable(MatrixDenseTest MatrixDenseTest.cu)
target_link_libraries(MatrixDenseTest sparsegpu)

add_test(NAME "MatrixDenseTest" COMMAND MatrixDenseTest)