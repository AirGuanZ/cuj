#include <cuj/builtin/string.h>

CUJ_NAMESPACE_BEGIN(cuj::builtin)

i32 strlen(Pointer<char> str)
{
    i32 ret = 0;
    $while(*str)
    {
        str = str + 1;
        ret = ret + 1;
    };
    return ret;
}

i32 strcmp(Pointer<char> a, Pointer<char> b)
{
    i32 ret = 0;
    $while(true)
    {
        char_t ch_a = *a;
        char_t ch_b = *b;
        $if(ch_a < ch_b)
        {
            ret = -1;
            $break;
        }
        $elif(ch_a > ch_b)
        {
            ret = 1;
            $break;
        }
        $elif(!ch_a)
        {
            ret = 0;
            $break;
        };
        a = a + 1;
        b = b + 1;
    };
    return ret;
}

void strcpy(Pointer<char> dst, Pointer<char> src)
{
    $while(true)
    {
        char_t ch = *src;
        $if(ch)
        {
            *dst = ch;
            dst = dst + 1;
            src = src + 1;
        }
        $else
        {
            $break;
        };
    };
    *dst = '\0';
}

void memcpy(Pointer<void> dst, Pointer<void> src, usize bytes)
{
    Pointer<uint8_t> ch_dst = ptr_cast<uint8_t>(dst);
    Pointer<uint8_t> ch_src = ptr_cast<uint8_t>(src);

    usize i = 0;
    $while(i < bytes)
    {
        ch_dst[i] = ch_src[i];
        i = i + 1;
    };

}

void memset(Pointer<void> dst, i32 ch, usize bytes)
{
    Pointer<uint8_t> ch_dst = ptr_cast<uint8_t>(dst);

    usize i = 0;
    $while(i < bytes)
    {
        ch_dst[i] = cast<uint8_t>(ch);
        i = i + 1;
    };
}

CUJ_NAMESPACE_END(cuj::builtin)
