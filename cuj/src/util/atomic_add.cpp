#include <cuj/util/atomic_add.h>

#ifdef _MSC_VER
#include <Windows.h>
#endif

CUJ_NAMESPACE_BEGIN(cuj)

#ifdef _MSC_VER

float atomic_add_float(float *dst, float value)
{
    auto iptr = reinterpret_cast<volatile LONG *>(dst);
    LONG expected = *iptr;

    while(true)
    {
        float old_value_f;
        std::memcpy(&old_value_f, &expected, sizeof(float));
        const float new_value_f = old_value_f + value;

        LONG new_value;
        std::memcpy(&new_value, &new_value_f, sizeof(float));
        
        const LONG actual = InterlockedCompareExchange(iptr, new_value, expected);
        if(actual == expected)
            return old_value_f;
        expected = actual;
    }
}

double atomic_add_double(double *dst, double value)
{
    auto iptr = reinterpret_cast<volatile LONGLONG*>(dst);
    LONGLONG expected = *iptr;

    while(true)
    {
        double old_value_f;
        std::memcpy(&old_value_f, &expected, sizeof(double));
        const double new_value_f = old_value_f + value;

        LONGLONG new_value;
        std::memcpy(&new_value, &new_value_f, sizeof(double));

        const LONGLONG actual = InterlockedCompareExchange64(iptr, new_value, expected);
        if(actual == expected)
            return old_value_f;
        expected = actual;
    }
}

#else

float atomic_add_float(float *dst, float value)
{
    auto iptr = reinterpret_cast<volatile int32_t *>(dst);
    int32_t expected = *iptr;

    while(true)
    {
        float old_value_f;
        std::memcpy(&old_value_f, &expected, sizeof(float));
        const float new_value_f = old_value_f + value;

        int32_t new_value;
        std::memcpy(&new_value, &new_value_f, sizeof(float));

        const int32_t actual = __sync_val_compare_and_swap(iptr, expected, new_value);
        if(actual == expected)
            return old_value_f;
        expected = actual;
    }
}

double atomic_add_double(double *dst, double value)
{
    auto iptr = reinterpret_cast<volatile int64_t *>(dst);
    int64_t expected = *iptr;

    while(true)
    {
        double old_value_f;
        std::memcpy(&old_value_f, &expected, sizeof(double));
        const double new_value_f = old_value_f + value;

        int64_t new_value;
        std::memcpy(&new_value, &new_value_f, sizeof(double));

        const int64_t actual = __sync_val_compare_and_swap(iptr, expected, new_value);
        if(actual == expected)
            return old_value_f;
        expected = actual;
    }
}

#endif // #ifdef _MSC_VER #else #endif

CUJ_NAMESPACE_END(cuj)
