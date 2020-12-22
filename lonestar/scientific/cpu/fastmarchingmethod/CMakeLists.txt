add_subdirectory(util)

app(fmm fmm.cpp speed_function.cpp boundary.cpp)
set_property(TARGET fmm PROPERTY CXX_STANDARD 17)
set_property(TARGET fmm APPEND PROPERTY COMPILE_FLAGS " -Werror=unused ")
target_link_libraries(fmm PUBLIC google-segystack PUBLIC numpy-cpp)

app(fmm_structured fmm_structured.cpp)
set_property(TARGET fmm_structured PROPERTY CXX_STANDARD 17)
target_link_libraries(fmm_structured PRIVATE numpy-cpp)
target_include_directories(fmm_structured PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/util")
