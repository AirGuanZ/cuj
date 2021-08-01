#pragma once

#include <cuj/common.h>

CUJ_NAMESPACE_BEGIN(cuj)

class UntypedOwner
{
public:

    UntypedOwner() = default;

    template<typename T, typename = std::enable_if_t<
                            std::is_class_v<T> &&
                           !std::is_same_v<rm_cvref_t<T>, UntypedOwner>>>
    explicit UntypedOwner(T &&value)
    {
        deleter_ = std::make_unique<ValueDeleter<T>>(std::move(value));
    }

    template<typename T, typename D = std::default_delete<T>>
    explicit UntypedOwner(T *pointer)
    {
        deleter_ = std::make_unique<PointerDeleter<T, D>>(pointer);
    }

    UntypedOwner(const UntypedOwner &) = delete;

    UntypedOwner(UntypedOwner &&other) noexcept
        : deleter_(std::move(other.deleter_))
    {
        
    }

    ~UntypedOwner() = default;

    UntypedOwner &operator=(const UntypedOwner &) = delete;

    UntypedOwner &operator=(UntypedOwner &&other) noexcept
    {
        deleter_ = std::move(other.deleter_);
        return *this;
    }

    bool has_value() const
    {
        return deleter_ != nullptr;
    }

    operator bool() const
    {
        return has_value();
    }

    template<typename T = void>
    const T *get() const
    {
        return static_cast<const T*>(deleter_.get());
    }

    template<typename T = void>
    T *get()
    {
        return static_cast<T *>(deleter_->get());
    }

private:

    struct Deleter
    {
        virtual ~Deleter() = default;

        virtual const void *get() const = 0;

        virtual void *get() = 0;
    };

    template<typename T>
    struct ValueDeleter : Deleter
    {
        explicit ValueDeleter(T &&val) noexcept : value(std::move(val)) { }

        const void *get() const override { return &value; }
              void *get()       override { return &value; }

        T value;
    };

    template<typename T, typename D>
    struct PointerDeleter : Deleter
    {
        explicit PointerDeleter(T *pointer) : ptr(pointer) { }

        const void *get() const override { return ptr.get(); }
              void *get()       override { return ptr.get(); }

        std::unique_ptr<T, D> ptr;
    };

    std::unique_ptr<Deleter> deleter_;
};

CUJ_NAMESPACE_END(cuj)
