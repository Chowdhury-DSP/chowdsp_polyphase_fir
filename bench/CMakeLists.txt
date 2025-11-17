message(STATUS "chowdsp_polyphase_fir -- Configuring benchmarks")

CPMAddPackage(
    NAME benchmark
    URI "gh:google/benchmark@1.9.4"
    OPTIONS "BENCHMARK_ENABLE_TESTING OFF"
)

add_executable(bench_chowdsp_polyphase_fir bench.cpp)
target_link_libraries(bench_chowdsp_polyphase_fir PRIVATE chowdsp_polyphase_fir chowdsp_lib benchmark::benchmark)
target_compile_features(bench_chowdsp_polyphase_fir PRIVATE cxx_std_20)
