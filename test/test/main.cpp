#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

#if CUJ_ENABLE_CUDA
#include <cuda.h>
#endif

int main(int argc, char *argv[])
{
#if CUJ_ENABLE_CUDA
    cuInit(0);
#endif

    return Catch::Session().run(argc, argv);
}
