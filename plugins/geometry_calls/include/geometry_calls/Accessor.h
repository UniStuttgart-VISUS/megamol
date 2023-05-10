#pragma once

#include <limits>
#include <type_traits>

#if !defined(_MSC_VER)
#if (__cplusplus < 201703L)
namespace std {
template<class T, class U>
constexpr bool is_same_v = is_same<T, U>::value;
}
#endif
#if (__cplusplus < 201402L)
namespace std {
template<bool B, class T = void>
using enable_if_t = typename enable_if<B, T>::type;
}
#endif
#endif

namespace megamol::geocalls {


template<class T>
T const* access(char const* ptr, size_t idx, size_t stride) {
    return reinterpret_cast<T const*>(ptr + idx * stride);
}


/**
 * Interface for accessor classes.
 */
class Accessor {
public:
    Accessor() = default;
    Accessor(Accessor const& rhs) = default;
    Accessor(Accessor&& rhs) noexcept = default;
    Accessor& operator=(Accessor const& rhs) = default;
    Accessor& operator=(Accessor&& rhs) noexcept = default;
    virtual float Get_f(size_t idx) const = 0;
    virtual double Get_d(size_t idx) const = 0;
    virtual uint64_t Get_u64(size_t idx) const = 0;
    virtual unsigned int Get_u32(size_t idx) const = 0;
    virtual unsigned short Get_u16(size_t idx) const = 0;
    virtual unsigned char Get_u8(size_t idx) const = 0;
    virtual ~Accessor() = default;
};


/**
 * Implementation of an accessor into a strided array.
 */
template<class T>
class Accessor_Impl : public Accessor {
public:
    Accessor_Impl(char const* ptr, size_t stride) : ptr_{ptr}, stride_{stride} {}

    Accessor_Impl(Accessor_Impl const& rhs) : ptr_{rhs.ptr_}, stride_{rhs.stride_} {}

    Accessor_Impl(Accessor_Impl&& rhs) noexcept : ptr_{nullptr}, stride_{0} {
        std::swap(ptr_, rhs.ptr_);
        std::swap(stride_, rhs.stride_);
    }

    Accessor_Impl& operator=(Accessor_Impl const& rhs) {
        ptr_ = rhs.ptr_;
        stride_ = rhs.stride_;
        return *this;
    }

    Accessor_Impl& operator=(Accessor_Impl&& rhs) noexcept {
        *this = rhs;
        rhs.ptr_ = nullptr;
        rhs.stride_ = 0;
        return *this;
    }

    template<class R>
    std::enable_if_t<std::is_same_v<T, R>, R> Get(size_t const idx) const {
        return *access<T>(ptr_, idx, stride_);
    }

    template<class R>
    std::enable_if_t<!std::is_same_v<T, R>, R> Get(size_t const idx) const {
        return static_cast<R>(Get<T>(idx));
    }

    float Get_f(size_t idx) const override {
        return Get<float>(idx);
    }

    double Get_d(size_t idx) const override {
        return Get<double>(idx);
    }

    uint64_t Get_u64(size_t idx) const override {
        return Get<uint64_t>(idx);
    }

    unsigned int Get_u32(size_t idx) const override {
        return Get<unsigned int>(idx);
    }

    unsigned short Get_u16(size_t idx) const override {
        return Get<unsigned short>(idx);
    }

    unsigned char Get_u8(size_t idx) const override {
        return Get<unsigned char>(idx);
    }

    ~Accessor_Impl() override = default;

private:
    char const* ptr_;
    size_t stride_;
};


/**
 * Accessor class reporting const values, for instance globals.
 */
template<class T, bool Norm>
class Accessor_Val : public Accessor {
public:
    Accessor_Val(T const val) : val_(val) {}

    Accessor_Val(Accessor_Val const& rhs) = default;

    Accessor_Val(Accessor_Val&& rhs) = default;

    Accessor_Val& operator=(Accessor_Val const& rhs) = default;

    Accessor_Val& operator=(Accessor_Val&& rhs) = default;

    template<class R>
    std::enable_if_t<!Norm, R> Get() const {
        return static_cast<R>(this->val_);
    }

    template<class R>
    std::enable_if_t<Norm, R> Get() const {
        return static_cast<R>(this->val_) / static_cast<R>(std::numeric_limits<T>::max());
    }

    float Get_f(size_t idx) const override {
        return Get<float>();
    }

    double Get_d(size_t idx) const override {
        return Get<double>();
    }

    uint64_t Get_u64(size_t idx) const override {
        return Get<uint64_t>();
    }

    unsigned int Get_u32(size_t idx) const override {
        return Get<unsigned int>();
    }

    unsigned short Get_u16(size_t idx) const override {
        return Get<unsigned short>();
    }

    unsigned char Get_u8(size_t idx) const override {
        return Get<unsigned char>();
    }

    ~Accessor_Val() override = default;

private:
    T val_;
};


/**
 * Dummy accessor for an empty array;
 */
class Accessor_0 : public Accessor {
public:
    Accessor_0() = default;

    Accessor_0(Accessor_0 const& rhs) = default;

    Accessor_0(Accessor_0&& rhs) = default;

    Accessor_0& operator=(Accessor_0 const& rhs) = default;

    Accessor_0& operator=(Accessor_0&& rhs) = default;

    float Get_f(size_t idx) const override {
        return static_cast<float>(0);
    }

    double Get_d(size_t idx) const override {
        return static_cast<double>(0);
    }

    uint64_t Get_u64(size_t idx) const override {
        return static_cast<uint64_t>(0);
    }

    unsigned int Get_u32(size_t idx) const override {
        return static_cast<unsigned int>(0);
    }

    unsigned short Get_u16(size_t idx) const override {
        return static_cast<unsigned short>(0);
    }

    unsigned char Get_u8(size_t idx) const override {
        return static_cast<unsigned char>(0);
    }

    ~Accessor_0() override = default;

private:
};


/**
 * Dummy accessor for reporting the idx back;
 */
class Accessor_Idx : public Accessor {
public:
    Accessor_Idx() = default;

    Accessor_Idx(Accessor_Idx const& rhs) = default;

    Accessor_Idx(Accessor_Idx&& rhs) = default;

    Accessor_Idx& operator=(Accessor_Idx const& rhs) = default;

    Accessor_Idx& operator=(Accessor_Idx&& rhs) = default;

    float Get_f(size_t idx) const override {
        return static_cast<float>(idx);
    }

    double Get_d(size_t idx) const override {
        return static_cast<double>(idx);
    }

    uint64_t Get_u64(size_t idx) const override {
        return static_cast<uint64_t>(idx);
    }

    unsigned int Get_u32(size_t idx) const override {
        return static_cast<unsigned int>(idx);
    }

    unsigned short Get_u16(size_t idx) const override {
        return static_cast<unsigned short>(idx);
    }

    unsigned char Get_u8(size_t idx) const override {
        return static_cast<unsigned char>(idx);
    }

    ~Accessor_Idx() override = default;

private:
};

} // end namespace megamol::geocalls
