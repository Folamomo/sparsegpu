add_executable(smokeTest smoke.cu)
target_link_libraries(smokeTest sparsegpu)

add_test(NAME "smokeTest" COMMAND smokeTest)
