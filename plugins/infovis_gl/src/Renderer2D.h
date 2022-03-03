#pragma once

#include "mmcore_gl/view/Renderer2DModuleGL.h"

#include "vislib_gl/graphics/gl/GLSLComputeShader.h"
#include "vislib_gl/graphics/gl/GLSLGeometryShader.h"
#include "vislib_gl/graphics/gl/GLSLShader.h"
#include "vislib_gl/graphics/gl/GLSLTesselationShader.h"

namespace megamol::infovis_gl {

#define DEBUG_NAME(name) name, (#name "\0")

class Renderer2D : public core_gl::view::Renderer2DModuleGL {
public:
    Renderer2D() : core_gl::view::Renderer2DModuleGL() {}

    virtual ~Renderer2D(){};

protected:
    void computeDispatchSizes(
        uint64_t numItems, GLint const localSizes[3], GLint const maxCounts[3], GLuint dispatchCounts[3]) const;

    static std::tuple<double, double> mouseCoordsToWorld(
        double mouse_x, double mouse_y, core::view::Camera const& cam, int width, int height);

    void makeDebugLabel(GLenum identifier, GLuint name, const char* label) const;
    void debugNotify(GLuint name, const char* message) const;
    void debugPush(GLuint name, const char* groupLabel) const;
    void debugPop() const;
};

} // namespace megamol::infovis_gl
