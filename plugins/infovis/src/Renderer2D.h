#ifndef MEGAMOL_INFOVIS_RENDERER2D_H_INCLUDED
#define MEGAMOL_INFOVIS_RENDERER2D_H_INCLUDED

#include "mmcore/view/Renderer2DModule.h"

#include "vislib/graphics/gl/GLSLComputeShader.h"
#include "vislib/graphics/gl/GLSLGeometryShader.h"
#include "vislib/graphics/gl/GLSLShader.h"
#include "vislib/graphics/gl/GLSLTesselationShader.h"

namespace megamol {
namespace infovis {

#define DEBUG_NAME(name) name, (#name "\0")

class Renderer2D : public core::view::Renderer2DModule {
public:
    Renderer2D() : core::view::Renderer2DModule() {}

    virtual ~Renderer2D(){};

protected:
    void computeDispatchSizes(
        uint64_t numItems, GLint const localSizes[3], GLint const maxCounts[3], GLuint dispatchCounts[3]) const;

    bool makeProgram(std::string prefix, vislib::graphics::gl::GLSLShader& program) const;
    bool makeProgram(std::string prefix, vislib::graphics::gl::GLSLGeometryShader& program) const;
    bool makeProgram(std::string prefix, vislib::graphics::gl::GLSLTesselationShader& program) const;
    bool makeProgram(std::string prefix, vislib::graphics::gl::GLSLComputeShader& program) const;

    void makeDebugLabel(GLenum identifier, GLuint name, const char* label) const;
    void debugNotify(GLuint name, const char* message) const;
    void debugPush(GLuint name, const char* groupLabel) const;
    void debugPop() const;
};

} // end namespace infovis
} // end namespace megamol

#endif // MEGAMOL_INFOVIS_RENDERER2D_H_INCLUDED