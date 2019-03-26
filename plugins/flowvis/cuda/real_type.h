#pragma once

#include <vector_types.h>
#include <helper_math.h>

#include <type_traits>

// Typedefs
#if USE_DOUBLE
typedef double Real;
typedef double2 Real2;
typedef double3 Real3;
typedef double4 Real4;
#else
typedef float Real;
typedef float2 Real2;
typedef float3 Real3;
typedef float4 Real4;
#endif

// Template for making differently sized vectors
template <typename float_type, int dimension> struct real_t {};

template <> struct real_t<float, 1>
{
    using value_type = float;
    using type = float;

    template <typename input_float_type> static __host__ __device__ type make(input_float_type a)
    {
        return static_cast<value_type>(a);
    }
};

template <> struct real_t<float, 2>
{
    using value_type = float;
    using type = float2;

    template <typename input_float_type> static __host__ __device__ type make(input_float_type a, input_float_type b)
    {
        return make_float2(static_cast<value_type>(a), static_cast<value_type>(b));
    }
};

template <> struct real_t<float, 3>
{
    using value_type = float;
    using type = float3;

    template <typename input_float_type> static __host__ __device__ type make(input_float_type a, input_float_type b, input_float_type c)
    {
        return make_float3(static_cast<value_type>(a), static_cast<value_type>(b), static_cast<value_type>(c));
    }
};

template <> struct real_t<float, 4>
{
    using value_type = float;
    using type = float4;

    template <typename input_float_type> static __host__ __device__ type make(input_float_type a, input_float_type b, input_float_type c, input_float_type d)
    {
        return make_float4(static_cast<value_type>(a), static_cast<value_type>(b), static_cast<value_type>(c), static_cast<value_type>(d));
    }
};

template <> struct real_t<double, 1>
{
    using value_type = double;
    using type = double;

    template <typename input_float_type> static __host__ __device__ type make(input_float_type a)
    {
        return static_cast<value_type>(a);
    }
};

template <> struct real_t<double, 2>
{
    using value_type = double;
    using type = double2;

    template <typename input_float_type> static __host__ __device__ type make(input_float_type a, input_float_type b)
    {
        return make_double2(static_cast<value_type>(a), static_cast<value_type>(b));
    }
};

template <> struct real_t<double, 3>
{
    using value_type = double;
    using type = double3;

    template <typename input_float_type> static __host__ __device__ type make(input_float_type a, input_float_type b, input_float_type c)
    {
        return make_double3(static_cast<value_type>(a), static_cast<value_type>(b), static_cast<value_type>(c));
    }
};

template <> struct real_t<double, 4>
{
    using value_type = double;
    using type = double4;

    template <typename input_float_type> static __host__ __device__ type make(input_float_type a, input_float_type b, input_float_type c, input_float_type d)
    {
        return make_double4(static_cast<value_type>(a), static_cast<value_type>(b), static_cast<value_type>(c), static_cast<value_type>(d));
    }
};

template <typename float_type, int dimension, typename input_float_type = float_type>
inline __host__ __device__
typename std::enable_if<dimension == 1, typename real_t<float_type, dimension>::type>::type make_real(input_float_type a)
{
    return real_t<float_type, 1>::make(a);
}

template <typename float_type, int dimension, typename input_float_type = float_type>
inline __host__ __device__
typename std::enable_if<dimension == 2, typename real_t<float_type, dimension>::type>::type make_real(input_float_type a)
{
    return real_t<float_type, 2>::make(a, a);
}

template <typename float_type, int dimension, typename input_float_type = float_type>
inline __host__ __device__
typename std::enable_if<dimension == 2, typename real_t<float_type, dimension>::type>::type make_real(input_float_type a, input_float_type b)
{
    return real_t<float_type, 2>::make(a, b);
}

template <typename float_type, int dimension, typename input_float_type = float_type>
inline __host__ __device__
typename std::enable_if<dimension == 2, typename real_t<float_type, dimension>::type>::type make_real(typename real_t<input_float_type, 2>::type v)
{
    return real_t<float_type, 2>::make(v.x, v.y);
}

template <typename float_type, int dimension, typename input_float_type = float_type>
inline __host__ __device__
typename std::enable_if<dimension == 2, typename real_t<float_type, dimension>::type>::type make_real(typename real_t<input_float_type, 3>::type v)
{
    return real_t<float_type, 2>::make(v.x, v.y);
}

template <typename float_type, int dimension, typename input_float_type = float_type>
inline __host__ __device__
typename std::enable_if<dimension == 2, typename real_t<float_type, dimension>::type>::type make_real(typename real_t<input_float_type, 4>::type v)
{
    return real_t<float_type, 2>::make(v.x, v.y);
}

template <typename float_type, int dimension, typename input_float_type = float_type>
inline __host__ __device__
typename std::enable_if<dimension == 3, typename real_t<float_type, dimension>::type>::type make_real(input_float_type a)
{
    return real_t<float_type, 3>::make(a, a, a);
}

template <typename float_type, int dimension, typename input_float_type = float_type>
inline __host__ __device__
typename std::enable_if<dimension == 3, typename real_t<float_type, dimension>::type>::type make_real(input_float_type a, input_float_type b, input_float_type c)
{
    return real_t<float_type, 3>::make(a, b, c);
}

template <typename float_type, int dimension, typename input_float_type = float_type>
inline __host__ __device__
typename std::enable_if<dimension == 3, typename real_t<float_type, dimension>::type>::type make_real(typename real_t<input_float_type, 3>::type v)
{
    return real_t<float_type, 3>::make(v.x, v.y, v.z);
}

template <typename float_type, int dimension, typename input_float_type = float_type>
inline __host__ __device__
typename std::enable_if<dimension == 3, typename real_t<float_type, dimension>::type>::type make_real(typename real_t<input_float_type, 4>::type v)
{
    return real_t<float_type, 3>::make(v.x, v.y, v.z);
}

template <typename float_type, int dimension, typename input_float_type = float_type>
inline __host__ __device__
typename std::enable_if<dimension == 4, typename real_t<float_type, dimension>::type>::type make_real(input_float_type a)
{
    return real_t<float_type, 4>::make(a, a, a, a);
}

template <typename float_type, int dimension, typename input_float_type = float_type>
inline __host__ __device__
typename std::enable_if<dimension == 4, typename real_t<float_type, dimension>::type>::type make_real(input_float_type a, input_float_type b, input_float_type c, input_float_type d)
{
    return real_t<float_type, 4>::make(a, b, c, d);
}

template <typename float_type, int dimension, typename input_float_type = float_type>
inline __host__ __device__
typename std::enable_if<dimension == 4, typename real_t<float_type, dimension>::type>::type make_real(typename real_t<input_float_type, 4>::type v)
{
    return real_t<float_type, 4>::make(v.x, v.y, v.z, v.w);
}

#if USE_DOUBLE
// Addition
inline __host__ __device__ Real4 operator+(Real4 a, Real4 b)
{
    return make_real<Real, 4>(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
inline __host__ __device__ Real3 operator+(Real3 a, Real3 b)
{
    return make_real<Real, 3>(a.x + b.x, a.y + b.y, a.z + b.z);
}

// Subtraction
inline __host__ __device__ Real4 operator-(Real4 a, Real4 b)
{
    return make_real<Real, 4>(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
inline __host__ __device__ Real3 operator-(Real3 a, Real3 b)
{
    return make_real<Real, 3>(a.x - b.x, a.y - b.y, a.z - b.z);
}

// Multiplication
inline __host__ __device__ Real4 operator*(Real4 a, Real b)
{
    return make_real<Real, 4>(a.x * b, a.y * b, a.z * b, a.w * b);
}
inline __host__ __device__ Real3 operator*(Real3 a, Real b)
{
    return make_real<Real, 3>(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ Real4 operator*(Real b, Real4 a)
{
    return make_real<Real, 4>(b * a.x, b * a.y, b * a.z, b * a.w);
}
inline __host__ __device__ Real3 operator*(Real b, Real3 a)
{
    return make_real<Real, 3>(b * a.x, b * a.y, b * a.z);
}
inline __host__ __device__ Real4 operator*(Real4 a, Real4 b)
{
    return make_real<Real, 4>(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
inline __host__ __device__ Real3 operator*(Real3 a, Real3 b)
{
    return make_real<Real, 3>(a.x * b.x, a.y * b.y, a.z * b.z);
}

// Division
inline __host__ __device__ Real4 operator/(Real4 a, Real b)
{
    return make_real<Real, 4>(a.x / b, a.y / b, a.z / b, a.w / b);
}
inline __host__ __device__ Real3 operator/(Real3 a, Real b)
{
    return make_real<Real, 3>(a.x / b, a.y / b, a.z / b);
}

// Inversion
inline __host__ __device__ Real4 operator-(Real4 a)
{
    return make_real<Real, 4>(-a.x, -a.y, -a.z, -a.w);
}
inline __host__ __device__ Real3 operator-(Real3 a)
{
    return make_real<Real, 3>(-a.x, -a.y, -a.z);
}

// Vector operations
inline __host__ __device__ Real dot(Real4 a, Real4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}
inline __host__ __device__ Real dot(Real3 a, Real3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__ Real dot(Real2 a, Real2 b)
{
    return a.x * b.x + a.y * b.y;
}

inline __host__ __device__ Real length(Real4 v)
{
    return sqrt(dot(v, v));
}
inline __host__ __device__ Real length(Real3 v)
{
    return sqrt(dot(v, v));
}
inline __host__ __device__ Real length(Real2 v)
{
    return sqrt(dot(v, v));
}

inline __host__ __device__ Real3 cross(Real3 a, Real3 b)
{
    return make_real<Real, 3>(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

// Minima and maxima
inline __host__ __device__ Real fminf(Real a, Real b)
{
    return a < b ? a : b;
}
inline __host__ __device__ Real fmaxf(Real a, Real b)
{
    return a > b ? a : b;
}

inline __host__ __device__ Real2 fminf(Real2 a, Real2 b)
{
    return make_real<Real, 2>(fminf(a.x, b.x), fminf(a.y, b.y));
}
inline __host__ __device__ Real2 fmaxf(Real2 a, Real2 b)
{
    return make_real<Real, 2>(fmaxf(a.x, b.x), fmaxf(a.y, b.y));
}

inline __host__ __device__ Real3 fminf(Real3 a, Real3 b)
{
    return make_real<Real, 3>(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}
inline __host__ __device__ Real3 fmaxf(Real3 a, Real3 b)
{
    return make_real<Real, 3>(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

inline __host__ __device__ Real4 fminf(Real4 a, Real4 b)
{
    return make_real<Real, 4>(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z), fminf(a.w, b.w));
}
inline __host__ __device__ Real4 fmaxf(Real4 a, Real4 b)
{
    return make_real<Real, 4>(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z), fmaxf(a.w, b.w));
}

// Absolutes
inline __host__ __device__ Real2 fabs(Real2 v)
{
    return make_real<Real, 2>(fabs(v.x), fabs(v.y));
}
inline __host__ __device__ Real3 fabs(Real3 v)
{
    return make_real<Real, 3>(fabs(v.x), fabs(v.y), fabs(v.z));
}
inline __host__ __device__ Real4 fabs(Real4 v)
{
    return make_real<Real, 4>(fabs(v.x), fabs(v.y), fabs(v.z), fabs(v.w));
}
#endif

// Safe normalization
inline __host__ __device__ Real4 normalizeSafe(Real4 v)
{
    Real len = length(v);

    if (len > 0)
    {
        return v / len;
    }

    return make_real<Real, 4>(0.0);
}
inline __host__ __device__ Real3 normalizeSafe(Real3 v)
{
    Real len = length(v);

    if (len > 0)
    {
        return v / len;
    }

    return make_real<Real, 3>(0.0);
}
