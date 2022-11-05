add_executable(csrAdd csrAdd.cu)
target_link_libraries(csrAdd sparsegpu)

add_test(NAME "csrAdd" COMMAND csrAdd)


