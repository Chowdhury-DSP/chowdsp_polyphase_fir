message(STATUS "chowdsp_polyphase_fir -- Configuring tests")

include(CPM.cmake)

CPMAddPackage("gh:catchorg/Catch2@3.2.1")
include(${Catch2_SOURCE_DIR}/extras/Catch.cmake)

CPMAddPackage("gh:Chowdhury-DSP/chowdsp_utils#abbf68c32b456c3d7209fe5ddb97ef8027c45eb8")
setup_chowdsp_lib(chowdsp_lib MODULES chowdsp_filters)
target_compile_features(chowdsp_lib PRIVATE cxx_std_20)

add_executable(test_chowdsp_polyphase_fir test.cpp)
target_link_libraries(test_chowdsp_polyphase_fir PRIVATE chowdsp_polyphase_fir chowdsp_lib Catch2::Catch2WithMain)
# target_include_directories(test_chowdsp_polyphase_fir PRIVATE ${pffft_SOURCE_DIR})
# target_compile_definitions(test_chowdsp_polyphase_fir PRIVATE _USE_MATH_DEFINES=1)
target_compile_features(test_chowdsp_polyphase_fir PRIVATE cxx_std_20)
