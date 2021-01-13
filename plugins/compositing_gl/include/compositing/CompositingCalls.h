/*
 * CompositingCalls.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef COMPOSITING_CALLS_H_INCLUDED
#define COMPOSITING_CALLS_H_INCLUDED

#include <memory>

#include "mmcore/CallGeneric.h"
#include "mmcore/view/Camera_2.h"

#define GLOWL_OPENGL_INCLUDE_GLAD
#include "glowl/FramebufferObject.hpp"
#include "glowl/Texture2D.hpp"

namespace megamol {
namespace compositing {

class CallTexture2D
    : public core::GenericVersionedCall<std::shared_ptr<glowl::Texture2D>, core::EmptyMetaData> {
public:
    inline CallTexture2D() : GenericVersionedCall<std::shared_ptr<glowl::Texture2D>, core::EmptyMetaData>() {}
    ~CallTexture2D() = default;

    static const char* ClassName(void) { return "CallTexture2D"; }
    static const char* Description(void) { return "Transports a shared pointer to an OpenGL texture object"; }
};

class CallCamera : public core::GenericVersionedCall<core::view::Camera_2, core::EmptyMetaData> {
public:
    inline CallCamera() : GenericVersionedCall<core::view::Camera_2, core::EmptyMetaData>() {}
    ~CallCamera() = default;

    static const char* ClassName(void) { return "CallCamera"; }
    static const char* Description(void) { return "Transports a camera (copy)"; }
};

class CallFramebufferGL
        : public core::GenericVersionedCall<std::shared_ptr<glowl::FramebufferObject>, core::EmptyMetaData> {
public:
    inline CallFramebufferGL()
            : GenericVersionedCall<std::shared_ptr<glowl::FramebufferObject>, core::EmptyMetaData>() {}
    ~CallFramebufferGL() = default;

    static const char* ClassName(void) { return "CallFramebufferGL"; }
    static const char* Description(void) { return "Transports a framebuffer object"; }
};

/** Description class typedef */
typedef megamol::core::factories::CallAutoDescription<CallTexture2D> CallTexture2DDescription;
typedef megamol::core::factories::CallAutoDescription<CallCamera> CallCameraDescription;
typedef megamol::core::factories::CallAutoDescription<CallFramebufferGL> CallFramebufferGLDescription;

} // namespace compositing
} // namespace megamol


#endif // !COMPOSITING_CALLS_H_INCLUDED
