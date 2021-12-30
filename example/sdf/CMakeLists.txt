PROJECT(CUJ-EXAMPLE-SDF)

ADD_EXECUTABLE(
	example_sdf
	"main.cpp" "vec.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/../../test/test/cuda/cuda.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/../../test/test/cuda/cuda.h")
SET_PROPERTY(TARGET example_sdf PROPERTY CXX_STANDARD 20)
SET_PROPERTY(TARGET example_sdf PROPERTY CXX_STANDARD_REQUIRED ON)
TARGET_LINK_LIBRARIES(example_sdf PUBLIC CUDA::nvrtc CUDA::cudart)
TARGET_LINK_LIBRARIES(example_sdf PUBLIC cuj)
