// code originally from https://github.com/UniStuttgart-VISUS/rtxpkd_ldav2020
// modified for MegaMol

// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

// ======================================================================== //
// Modified 2019-2025 VISUS - University of Stuttgart                       //
// ======================================================================== //

#pragma once

#include <owl/common/math/box.h>
#include <owl/common/math/vec.h>

#include "CUDAUtils.h"

namespace megamol {
namespace optix_owl {
namespace device {
using namespace owl::common;
struct Particle {
    vec3f pos;
    void set_dim(int const dim) {
        auto x_val = reinterpret_cast<uint32_t*>(&pos.x);
        *x_val = (*x_val & ~1u) | (dim & 1u);
        auto y_val = reinterpret_cast<uint32_t*>(&pos.y);
        *y_val = (*y_val & ~1u) | ((dim >> 1) & 1u);
    }
    CU_CALLABLE int get_dim() const {
        int dim = 0;
        auto x_val = reinterpret_cast<uint32_t const*>(&pos.x);
        dim = (dim & ~1u) | (*x_val & 1u);
        auto y_val = reinterpret_cast<uint32_t const*>(&pos.y);
        dim = (dim & ~2u) | ((*y_val & 1u) << 1);
        return dim;
    }
};

struct PKDlet {
    //! bounding box of all particles (including the radius)
    box3f bounds;
    //! begin/end range in the common particles array
    unsigned int begin, end;
};
} // namespace device
} // namespace optix_owl
} // namespace megamol
