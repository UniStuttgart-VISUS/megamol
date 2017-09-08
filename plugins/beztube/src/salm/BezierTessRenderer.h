/*
 * BezierTessRenderer.h
 *
 * Copyright (C) 2016 by TU Dresden, MegaMol Team, Mirko Salm, Sebastian Grottel
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "vislib/graphics/gl/IncludeAllGL.h"
#include "AbstractBezierRenderer.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/graphics/gl/GLSLTesselationShader.h"
#include <memory>
#include "salm/IndexBuffer.h"
#include "salm/ShaderBufferContent.h"
#include "mmcore/utility/ShaderSourceFactory.h"
#include "salm/ShaderBuffer.h"
#include "salm/SplineTubesTess.h"

namespace megamol {
namespace beztube {
namespace salm {

    class BezierTessRenderer : public AbstractBezierRenderer {
    public:
        static const char *ClassName(void) { return "BezierTessRenderer"; }
        static const char *Description(void) { return "Renderer for bézier curve"; }
        static bool IsAvailable(void) {
#ifdef _WIN32
#if defined(DEBUG) || defined(_DEBUG)
            HDC dc = ::wglGetCurrentDC();
            HGLRC rc = ::wglGetCurrentContext();
            ASSERT(dc != NULL);
            ASSERT(rc != NULL);
#endif // DEBUG || _DEBUG
#endif // _WIN32
            return vislib::graphics::gl::GLSLTesselationShader::AreExtensionsAvailable();
        }
        BezierTessRenderer();
        virtual ~BezierTessRenderer();
    protected:

        virtual bool shader_required(void) const { return true; }

        virtual bool create(void);
        virtual void release(void);
        virtual bool render(core::view::CallRender3D& call);

    private:

        class gpuResources {
        public:
            gpuResources(core::utility::ShaderSourceFactory& shaderFactor);
            ~gpuResources();

            inline bool Error() const { return error; }

            const unsigned int cylinderXSize = 8;
            const unsigned int cylinderYSize = 12;

            IndexBuffer indexBuffer;
            vislib::graphics::gl::GLSLTesselationShader shader;
            
            ShaderBufferContent perFrameData;
            ShaderBuffer perFrameUBO;
            ShaderBufferContent staticData;
            ShaderBuffer staticUBO;

            SplineTubesTess tubes;

        private:
            bool error;
        };

        std::shared_ptr<gpuResources> gpuRes;

    };

}
}
}
