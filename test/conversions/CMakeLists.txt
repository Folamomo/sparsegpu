add_executable(csrToDense csrToDense.cu)
target_link_libraries(csrToDense sparsegpu)

add_test(NAME "csrToDense" COMMAND csrToDense)


add_executable(csrToDenseDev csrToDenseDev.cu)
target_link_libraries(csrToDenseDev sparsegpu)

add_test(NAME "csrToDenseDev" COMMAND csrToDenseDev)