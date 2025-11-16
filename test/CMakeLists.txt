message(STATUS "chowdsp_polyphase_fir -- Configuring tests")

include(CPM.cmake)

CPMAddPackage("gh:catchorg/Catch2@3.2.1")
include(${Catch2_SOURCE_DIR}/extras/Catch.cmake)

CPMAddPackage("gh:Chowdhury-DSP/chowdsp_utils#b8959ece1de97e2341388fc1028c27acc81711ac")
setup_chowdsp_lib(chowdsp_lib MODULES chowdsp_filters)
target_compile_features(chowdsp_lib PRIVATE cxx_std_20)

add_executable(test_chowdsp_polyphase_fir test.cpp)
target_link_libraries(test_chowdsp_polyphase_fir PRIVATE chowdsp_polyphase_fir chowdsp_lib Catch2::Catch2WithMain)
target_compile_features(test_chowdsp_polyphase_fir PRIVATE cxx_std_20)
