#pragma once

#include <owl/common/math/box.h>
#include <owl/common/math/vec.h>

#include "particle.h"

#include "CUDAUtils.h"

namespace megamol {
namespace optix_owl {
namespace device {
using namespace owl::common;

constexpr int const dec_val = 6;

typedef union {
    unsigned int ui;
    struct {
        unsigned char a;
        unsigned char b;
        unsigned char c;
        unsigned char d;
    } parts;
} byte_cast;

static int const spkd_array_size = 3;
struct GridCompPKDlet {
    box3f bounds;
    unsigned int begin, end;
    vec3f lower;
    unsigned char sx[spkd_array_size], sy[spkd_array_size], sz[spkd_array_size];
};

struct GridCompParticle {
    unsigned char dim : 2;
    unsigned char sx_idx : 2;
    unsigned char sy_idx : 2;
    unsigned char sz_idx : 2;
    unsigned char x;
    unsigned char y;
    unsigned char z;
};

struct GridCompGeomData {
    GridCompPKDlet* treeletBuffer;
    GridCompParticle* particleBuffer;
    float particleRadius;
};

inline CU_CALLABLE vec3f decode_spart(GridCompParticle const& part, GridCompPKDlet const& treelet) {
    constexpr const float factor = 1.0f / static_cast<float>(1 << dec_val);
    vec3f pos;
    //device::QPKDParticle qp;
    byte_cast bc;
    bc.ui = 0;
    bc.parts.a = part.x;
    bc.parts.b = treelet.sx[part.sx_idx];
#if 0
    pos.x = fmaf(static_cast<float>(bc.ui), factor, treelet.lower.x);
#else
    pos.x = static_cast<float>(bc.ui) * factor + treelet.lower.x;
#endif
    //qp.x = bc.ui;
    bc.parts.a = part.y;
    bc.parts.b = treelet.sy[part.sy_idx];
#if 0
    pos.y = fmaf(static_cast<float>(bc.ui), factor, treelet.lower.y);
#else
    pos.y = static_cast<float>(bc.ui) * factor + treelet.lower.y;
#endif
    //qp.y = bc.ui;
    bc.parts.a = part.z;
    bc.parts.b = treelet.sz[part.sz_idx];
#if 0
    pos.z = fmaf(static_cast<float>(bc.ui), factor, treelet.lower.z);
#else
    pos.z = static_cast<float>(bc.ui) * factor + treelet.lower.z;
#endif
    //qp.z = bc.ui;
    //return megamol::optix_hpg::decode_coord(qp /*, glm::vec3(), glm::vec3()*/);
    return pos;
}
} // namespace device
} // namespace optix_owl
} // namespace megamol
