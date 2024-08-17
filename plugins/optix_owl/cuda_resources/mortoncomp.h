#pragma once

#include <cstdint>

#include <owl/common/math/box.h>

#include "particle.h"

#include "CUDAUtils.h"

namespace megamol {
namespace optix_owl {
namespace device {
using namespace owl::common;

// https://stackoverflow.com/questions/49748864/morton-reverse-encoding-for-a-3d-grid

/* Morton encoding in binary (components 21-bit: 0..2097151)
                0zyxzyxzyxzyxzyxzyxzyxzyxzyxzyxzyxzyxzyxzyxzyxzyxzyxzyxzyxzyxzyx */
#define BITMASK_0000000001000001000001000001000001000001000001000001000001000001 UINT64_C(18300341342965825)
#define BITMASK_0000001000001000001000001000001000001000001000001000001000001000 UINT64_C(146402730743726600)
#define BITMASK_0001000000000000000000000000000000000000000000000000000000000000 UINT64_C(1152921504606846976)
/*              0000000ccc0000cc0000cc0000cc0000cc0000cc0000cc0000cc0000cc0000cc */
#define BITMASK_0000000000000011000000000011000000000011000000000011000000000011 UINT64_C(844631138906115)
#define BITMASK_0000000111000000000011000000000011000000000011000000000011000000 UINT64_C(126113986927919296)
/*              00000000000ccccc00000000cccc00000000cccc00000000cccc00000000cccc */
#define BITMASK_0000000000000000000000000000000000001111000000000000000000001111 UINT64_C(251658255)
#define BITMASK_0000000000000000000000001111000000000000000000001111000000000000 UINT64_C(1030792212480)
#define BITMASK_0000000000011111000000000000000000000000000000000000000000000000 UINT64_C(8725724278030336)
/*              000000000000000000000000000ccccccccccccc0000000000000000cccccccc */
#define BITMASK_0000000000000000000000000000000000000000000000000000000011111111 UINT64_C(255)
#define BITMASK_0000000000000000000000000001111111111111000000000000000000000000 UINT64_C(137422176256)
/*                                                         ccccccccccccccccccccc */
#define BITMASK_21BITS UINT64_C(2097151)


CU_CALLABLE static inline void morton_decode(uint64_t m, uint32_t& xto, uint32_t& yto, uint32_t& zto) {
    constexpr uint64_t const mask0 = 0b0000000001000001000001000001000001000001000001000001000001000001,
                             mask1 = 0b0000001000001000001000001000001000001000001000001000001000001000,
                             mask2 = 0b0001000000000000000000000000000000000000000000000000000000000000,
                             mask3 = 0b0000000000000011000000000011000000000011000000000011000000000011,
                             mask4 = 0b0000000111000000000011000000000011000000000011000000000011000000,
                             mask5 = 0b0000000000000000000000000000000000001111000000000000000000001111,
                             mask6 = 0b0000000000000000000000001111000000000000000000001111000000000000,
                             mask7 = 0b0000000000011111000000000000000000000000000000000000000000000000,
                             mask8 = 0b0000000000000000000000000000000000000000000000000000000011111111,
                             mask9 = 0b0000000000000000000000000001111111111111000000000000000000000000;
    uint64_t x = m, y = m >> 1, z = m >> 2;

    /* 000c00c00c00c00c00c00c00c00c00c00c00c00c00c00c00c00c00c00c00c00c */
    x = (x & mask0) | ((x & mask1) >> 2) | ((x & mask2) >> 4);
    y = (y & mask0) | ((y & mask1) >> 2) | ((y & mask2) >> 4);
    z = (z & mask0) | ((z & mask1) >> 2) | ((z & mask2) >> 4);
    /* 0000000ccc0000cc0000cc0000cc0000cc0000cc0000cc0000cc0000cc0000cc */
    x = (x & mask3) | ((x & mask4) >> 4);
    y = (y & mask3) | ((y & mask4) >> 4);
    z = (z & mask3) | ((z & mask4) >> 4);
    /* 00000000000ccccc00000000cccc00000000cccc00000000cccc00000000cccc */
    x = (x & mask5) | ((x & mask6) >> 8) | ((x & mask7) >> 16);
    y = (y & mask5) | ((y & mask6) >> 8) | ((y & mask7) >> 16);
    z = (z & mask5) | ((z & mask6) >> 8) | ((z & mask7) >> 16);
    /* 000000000000000000000000000ccccccccccccc0000000000000000cccccccc */
    x = (x & mask8) | ((x & mask9) >> 16);
    y = (y & mask8) | ((y & mask9) >> 16);
    z = (z & mask8) | ((z & mask9) >> 16);
    /* 0000000000000000000000000000000000000000000ccccccccccccccccccccc */

    xto = x;

    yto = y;

    zto = z;
}

static inline uint64_t morton_encode(uint32_t xsrc, uint32_t ysrc, uint32_t zsrc) {
    constexpr uint64_t const mask0 = 0b0000000001000001000001000001000001000001000001000001000001000001,
                             mask1 = 0b0000001000001000001000001000001000001000001000001000001000001000,
                             mask2 = 0b0001000000000000000000000000000000000000000000000000000000000000,
                             mask3 = 0b0000000000000011000000000011000000000011000000000011000000000011,
                             mask4 = 0b0000000111000000000011000000000011000000000011000000000011000000,
                             mask5 = 0b0000000000000000000000000000000000001111000000000000000000001111,
                             mask6 = 0b0000000000000000000000001111000000000000000000001111000000000000,
                             mask7 = 0b0000000000011111000000000000000000000000000000000000000000000000,
                             mask8 = 0b0000000000000000000000000000000000000000000000000000000011111111,
                             mask9 = 0b0000000000000000000000000001111111111111000000000000000000000000;
    uint64_t x = xsrc, y = ysrc, z = zsrc;
    /* 0000000000000000000000000000000000000000000ccccccccccccccccccccc */
    x = (x & mask8) | ((x << 16) & mask9);
    y = (y & mask8) | ((y << 16) & mask9);
    z = (z & mask8) | ((z << 16) & mask9);
    /* 000000000000000000000000000ccccccccccccc0000000000000000cccccccc */
    x = (x & mask5) | ((x << 8) & mask6) | ((x << 16) & mask7);
    y = (y & mask5) | ((y << 8) & mask6) | ((y << 16) & mask7);
    z = (z & mask5) | ((z << 8) & mask6) | ((z << 16) & mask7);
    /* 00000000000ccccc00000000cccc00000000cccc00000000cccc00000000cccc */
    x = (x & mask3) | ((x << 4) & mask4);
    y = (y & mask3) | ((y << 4) & mask4);
    z = (z & mask3) | ((z << 4) & mask4);
    /* 0000000ccc0000cc0000cc0000cc0000cc0000cc0000cc0000cc0000cc0000cc */
    x = (x & mask0) | ((x << 2) & mask1) | ((x << 4) & mask2);
    y = (y & mask0) | ((y << 2) & mask1) | ((y << 4) & mask2);
    z = (z & mask0) | ((z << 2) & mask1) | ((z << 4) & mask2);
    /* 000c00c00c00c00c00c00c00c00c00c00c00c00c00c00c00c00c00c00c00c00c */
    return x | (y << 1) | (z << 2);
}

using morton_prefix_t = unsigned int;

struct MortonConfig {
    static const uint64_t mask = 0b111111111111111111000000000000000000000000000000000000000;
    static const int prefix_offset = 39;
    static const uint64_t factor = (1 << 19) - 1;
    static const int code_offset = 9;
};

struct MortonCompPKDlet {
    box3f bounds;
    unsigned int begin, end;
    morton_prefix_t prefix;
};

struct MortonCompParticle {
    unsigned int dim : 2;
    unsigned int code : 30;

    CU_CALLABLE vec3f from(morton_prefix_t const prefix, vec3f const& span, vec3f const& lower) const {
        auto const combined_code = (static_cast<uint64_t>(code) << MortonConfig::code_offset) +
                                   (static_cast<uint64_t>(prefix) << MortonConfig::prefix_offset);

        static float const ffactor = MortonConfig::factor;

        uint32_t x, y, z;
        morton_decode(combined_code, x, y, z);
#ifdef __CUDACC__
        vec3f basePos(
            fmaf(x / ffactor, span.x, lower.x), fmaf(y / ffactor, span.y, lower.y), fmaf(z / ffactor, span.z, lower.z));
#else
        vec3f basePos(x / ffactor, y / ffactor, z / ffactor);
        basePos *= span;
        basePos += lower;
#endif

        return basePos;
    }
};

struct MortonCompGeomData {
    MortonCompPKDlet* treeletBuffer;
    MortonCompParticle* particleBuffer;
    float particleRadius;
    box3f bounds;
};
} // namespace device
} // namespace optix_owl
} // namespace megamol
