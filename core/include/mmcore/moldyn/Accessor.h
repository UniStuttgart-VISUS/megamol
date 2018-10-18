#pragma once

namespace megamol {
namespace core {
namespace moldyn {


template <class T> T const* access(char const* ptr, size_t idx, size_t stride) {
    return reinterpret_cast<T const*>(ptr + idx * stride);
}


class Accessor {
public:
    virtual float Get_f(size_t idx) const = 0;
    virtual double Get_d(size_t idx) const = 0;
    virtual uint64_t Get_u64(size_t idx) const = 0;
    virtual unsigned int Get_u32(size_t idx) const = 0;
    virtual unsigned short Get_u16(size_t idx) const = 0;
    virtual unsigned char Get_u8(size_t idx) const = 0;
    virtual ~Accessor() = default;
};


template <class T> class Accessor_Impl : public Accessor {
public:
    Accessor_Impl(char const* ptr, size_t stride) : ptr_{ptr}, stride_{stride} {}

    template <class R> std::enable_if_t<std::is_same_v<T, R>, R> Get(size_t const idx) const {
        return *access<T>(ptr_, idx, stride_);
    }

    template <class R> std::enable_if_t<!std::is_same_v<T, R>, R> Get(size_t const idx) const {
        return static_cast<R>(Get<T>(idx));
    }

    float Get_f(size_t idx) const override { return Get<float>(idx); }

    double Get_d(size_t idx) const override { return Get<double>(idx); }

    uint64_t Get_u64(size_t idx) const override { return Get<uint64_t>(idx); }

    unsigned int Get_u32(size_t idx) const override { return Get<unsigned int>(idx); }

    unsigned short Get_u16(size_t idx) const override { return Get<unsigned short>(idx); }

    unsigned char Get_u8(size_t idx) const override { return Get<unsigned char>(idx); }

    virtual ~Accessor_Impl() = default;

private:
    char const* ptr_;
    size_t stride_;
};


class Accessor_0 : public Accessor {
public:
    Accessor_0() = default;

    float Get_f(size_t idx) const override { return static_cast<float>(0); }

    double Get_d(size_t idx) const override { return static_cast<double>(0); }

    uint64_t Get_u64(size_t idx) const override { return static_cast<uint64_t>(0); }

    unsigned int Get_u32(size_t idx) const override { return static_cast<unsigned int>(0); }

    unsigned short Get_u16(size_t idx) const override { return static_cast<unsigned short>(0); }

    unsigned char Get_u8(size_t idx) const override { return static_cast<unsigned char>(0); }

    virtual ~Accessor_0() = default;

private:
};

} // end namespace moldyn
} // end namespace core
} // end namespace megamol