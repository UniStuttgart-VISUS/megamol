#ifndef MEGAMOL_INFOVIS_RENDERER2D_H_INCLUDED
#define MEGAMOL_INFOVIS_RENDERER2D_H_INCLUDED

#include "mmcore/view/Renderer2DModuleGL.h"

#include "vislib_gl/graphics/gl/GLSLComputeShader.h"
#include "vislib_gl/graphics/gl/GLSLGeometryShader.h"
#include "vislib_gl/graphics/gl/GLSLShader.h"
#include "vislib_gl/graphics/gl/GLSLTesselationShader.h"

namespace megamol {
namespace infovis_gl {

#define DEBUG_NAME(name) name, (#name "\0")

    class Renderer2D : public core::view::Renderer2DModuleGL {
    public:
        Renderer2D() : core::view::Renderer2DModuleGL() {}

        virtual ~Renderer2D(){};

    protected:
        void computeDispatchSizes(
            uint64_t numItems, GLint const localSizes[3], GLint const maxCounts[3], GLuint dispatchCounts[3]) const;

        void makeDebugLabel(GLenum identifier, GLuint name, const char* label) const;
        void debugNotify(GLuint name, const char* message) const;
        void debugPush(GLuint name, const char* groupLabel) const;
        void debugPop() const;
    };

} // end namespace infovis
} // end namespace megamol

#endif // MEGAMOL_INFOVIS_RENDERER2D_H_INCLUDED
