/*
 * CompositingCalls.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef MESH_CALLS_H_INCLUDED
#define MESH_CALLS_H_INCLUDED

#include <memory>

#include "mmcore/CallGeneric.h"

#include "compositing.h"

#include "glowl/Texture2D.hpp"

namespace megamol {
namespace compositing {

class COMPOSITING_API CallTexture2D
    : public core::CallGeneric<std::shared_ptr<glowl::Texture2D>, core::BasicMetaData> {
public:
    inline CallTexture2D() : CallGeneric<std::shared_ptr<glowl::Texture2D>, core::BasicMetaData>() {}
    ~CallTexture2D() = default;

    static const char* ClassName(void) { return "CallTexture2D"; }
    static const char* Description(void) { return "Transports a shared pointer to an OpenGL texture object"; }
};

/** Description class typedef */
typedef megamol::core::factories::CallAutoDescription<CallTexture2D> CallTexture2DDescription;

} // namespace mesh
} // namespace megamol


#endif // !MESH_CALLS_H_INCLUDED
