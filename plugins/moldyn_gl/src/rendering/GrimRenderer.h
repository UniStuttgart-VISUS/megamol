/*
 * GrimRenderer.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_MOLDYN_GRIMRENDERER_H_INCLUDED
#define MEGAMOL_MOLDYN_GRIMRENDERER_H_INCLUDED

#include "moldyn/ParticleGridDataCall.h"

#include "mmcore/Call.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/CallClipPlane.h"
#include "mmcore_gl/view/CallGetTransferFunctionGL.h"
#include "mmcore_gl/view/Renderer3DModuleGL.h"

#include "vislib/Array.h"
#include "vislib/Pair.h"
#include "vislib/SmartPtr.h"
#include "vislib/Trace.h"
#include "vislib/assert.h"
#include "vislib/forceinline.h"
#include "vislib/math/Cuboid.h"
#include "vislib/math/Plane.h"
#include "vislib/math/Point.h"
#include "vislib/math/Vector.h"
#include "vislib/math/mathfunctions.h"
#include "vislib/math/mathtypes.h"
#include "vislib/sys/sysfunctions.h"

#include "glowl/glowl.h"
#include "mmcore_gl/utility/ShaderFactory.h"
#include "vislib_gl/graphics/gl/FramebufferObject.h"

#include <climits>

#include <glm/glm.hpp>


namespace megamol {
namespace moldyn_gl {
namespace rendering {


/**
 * Renderer for gridded imposters
 */
class GrimRenderer : public core_gl::view::Renderer3DModuleGL {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "GrimRenderer";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Renderer of gridded imposters.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    /** Ctor. */
    GrimRenderer(void);

    /** Dtor. */
    virtual ~GrimRenderer(void);

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);

    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool GetExtents(core_gl::view::CallRender3DGL& call);

    /**
     * Implementation of 'Release'.
     */
    virtual void release(void);

    /**
     * The render callback.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool Render(core_gl::view::CallRender3DGL& call);

private:
    /**
     * Utility class storing additional rendering information about a
     * given cell
     */
    class CellInfo {
    public:
        class CacheItem {
        public:
            GLuint data[2];

            CacheItem() {
                this->data[0] = 0;
                this->data[1] = 0;
            }

            ~CacheItem() {
                if (this->data[0] != 0) {
                    ::glDeleteBuffersARB(2, this->data);
                }
                this->data[0] = 0;
                this->data[1] = 0;
            }

            inline bool operator==(const CacheItem& rhs) {
                return (this->data[0] == rhs.data[0]) && (this->data[1] == rhs.data[1]);
            }
        };

        /** Flag if the cell is visible now (inside the frustum; not occluded) */
        bool isvisible;

        /** Flag if the cell was visible in the last render */
        bool wasvisible;

        /** Flag if the cell is far enought to be rendered with dots */
        bool dots;

        /** The max radius found in this cell */
        float maxrad;

        /** The occlusion query object */
        unsigned int oQuery;

        /** gpu-ram caching variables */
        std::vector<CacheItem> cache;

        /**
         * Ctor
         */
        CellInfo(void);

        /**
         * Dtor
         */
        ~CellInfo(void);

        /**
         * Test for equality
         *
         * @param rhs The right hand side operand
         *
         * @return 'true' if 'this' and 'rhs' are equal.
         */
        inline bool operator==(const CellInfo& rhs) const {
            return (this->isvisible == rhs.isvisible) && (this->wasvisible == rhs.wasvisible) &&
                   (this->dots == rhs.dots) && (vislib::math::IsEqual(this->maxrad, rhs.maxrad));
        }
    };

    /**
     * Sorts the grid cells by their distance to the viewer
     *
     * @param lhs The left hand side operand
     * @param rhs The right hand side operand
     *
     * @return The distance sort info
     */
    static bool depthSort(const vislib::Pair<unsigned int, float>& lhs, const vislib::Pair<unsigned int, float>& rhs);

    void set_cam_uniforms(std::shared_ptr<glowl::GLSLProgram> shader, glm::mat4 view_matrix_inv,
        glm::mat4 view_matrix_inv_transp, glm::mat4 mvp_matrix, glm::mat4 mvp_matrix_transp, glm::mat4 mvp_matrix_inv,
        glm::vec4 camPos, glm::vec4 curlightDir);

    /** The sphere shader */
    std::shared_ptr<glowl::GLSLProgram> sphereShader;

    /** The vanilla sphere shader */
    std::shared_ptr<glowl::GLSLProgram> vanillaSphereShader;

    /** The shader to init the depth fbo */
    std::shared_ptr<glowl::GLSLProgram> initDepthShader;

    /** The shader to init the depth mip-map generation */
    std::shared_ptr<glowl::GLSLProgram> initDepthMapShader;

    /** The shader for the depth mip-mapping ping-ponging */
    std::shared_ptr<glowl::GLSLProgram> depthMipShader;

    /** The shader to render far-away, solid-coloured points */
    std::shared_ptr<glowl::GLSLProgram> pointShader;

    /** The shader to init the depth buffer with points */
    std::shared_ptr<glowl::GLSLProgram> initDepthPointShader;

    /** Von Guido aus */
    std::shared_ptr<glowl::GLSLProgram> vertCntShader;

    /** Von Guido aus */
    std::shared_ptr<glowl::GLSLProgram> vertCntShade2r;

    /** The frame buffer object for the depth estimate */
    vislib_gl::graphics::gl::FramebufferObject fbo;

    /** buffers for depth-max mip map */
    vislib_gl::graphics::gl::FramebufferObject depthmap[2];

    /** The call for data */
    core::CallerSlot getDataSlot;

    /** The call for Transfer function */
    core::CallerSlot getTFSlot;

    /** The call for light sources */
    core::CallerSlot getLightsSlot;

    /** Flag to activate per cell culling */
    core::param::ParamSlot useCellCullSlot;

    /** Flag to activate per vertex culling */
    core::param::ParamSlot useVertCullSlot;

    /** Flag to activate output of percentage of culled cells */
    core::param::ParamSlot speakCellPercSlot;

    /** Flag to activate output of number of vertices */
    core::param::ParamSlot speakVertCountSlot;

    /** De-/Activates deferred shading with normal generation */
    core::param::ParamSlot deferredShadingSlot;

    /** A simple black-to-white transfer function texture as fallback */
    unsigned int greyTF;

    /** Cell distances */
    std::vector<vislib::Pair<unsigned int, float>> cellDists;

    /** Cell rendering informations */
    std::vector<CellInfo> cellInfos;

    /** Bytes of the GPU-Memory available for caching */
    SIZE_T cacheSize;

    /** Bytes of the GPU-Memory used by the caching */
    SIZE_T cacheSizeUsed;

    /** Frame buffer object used for deferred shading */
    vislib_gl::graphics::gl::FramebufferObject dsFBO;

    /** The sphere shader */
    std::shared_ptr<glowl::GLSLProgram> deferredSphereShader;

    /** The vanilla sphere shader */
    std::shared_ptr<glowl::GLSLProgram> deferredVanillaSphereShader;

    /** The shader to render far-away, solid-coloured points */
    std::shared_ptr<glowl::GLSLProgram> deferredPointShader;

    /** The deferred shader */
    std::shared_ptr<glowl::GLSLProgram> deferredShader;

    /** The hash of the incoming data */
    SIZE_T inhash;
};

} /* end namespace rendering */
} // namespace moldyn_gl
} /* end namespace megamol */

#endif /* MEGAMOL_MOLDYN_GRIMRENDERER_H_INCLUDED */
