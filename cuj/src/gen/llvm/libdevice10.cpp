#if CUJ_ENABLE_CUDA

#include <string>

#include <cuj/common.h>

CUJ_NAMESPACE_BEGIN(cuj::gen::libdev)

const unsigned char LIBDEVICE_10_BITCODE[] = {
#include "./libdevice10.inl"
};

const size_t LIBDEVICE_10_BITCODE_SIZE = sizeof(LIBDEVICE_10_BITCODE);

std::string get_libdevice_str()
{
    std::string libdev_str;
    libdev_str.resize(LIBDEVICE_10_BITCODE_SIZE);
    std::memcpy(
        libdev_str.data(), LIBDEVICE_10_BITCODE, LIBDEVICE_10_BITCODE_SIZE);
    return libdev_str;
}

CUJ_NAMESPACE_END(cuj::gen::libdev)

#endif // #if CUJ_ENABLE_CUDA
