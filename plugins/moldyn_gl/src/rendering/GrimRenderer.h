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
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/ParamSlot.h"
#include "mmstd/renderer/CallClipPlane.h"
#include "mmstd_gl/renderer/CallGetTransferFunctionGL.h"
#include "mmstd_gl/renderer/Renderer3DModuleGL.h"

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


namespace megamol::moldyn_gl::rendering {


/**
 * Renderer for gridded imposters
 */
class GrimRenderer : public mmstd_gl::Renderer3DModuleGL {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "GrimRenderer";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Renderer of gridded imposters.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    /** Ctor. */
    GrimRenderer();

    /** Dtor. */
    ~GrimRenderer() override;

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() override;

    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool GetExtents(mmstd_gl::CallRender3DGL& call) override;

    /**
     * Implementation of 'Release'.
     */
    void release() override;

    /**
     * The render callback.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool Render(mmstd_gl::CallRender3DGL& call) override;

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
        CellInfo();

        /**
         * Dtor
         */
        ~CellInfo();

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
        glm::vec4 cam_pos, glm::vec4 curlight_dir);

    /** The sphere shader */
    std::shared_ptr<glowl::GLSLProgram> sphere_shader_;

    /** The vanilla sphere shader */
    std::shared_ptr<glowl::GLSLProgram> vanilla_sphere_shader_;

    /** The shader to init the depth fbo */
    std::shared_ptr<glowl::GLSLProgram> init_depth_shader_;

    /** The shader to init the depth mip-map generation */
    std::shared_ptr<glowl::GLSLProgram> init_depth_map_shader_;

    /** The shader for the depth mip-mapping ping-ponging */
    std::shared_ptr<glowl::GLSLProgram> depth_mip_shader_;

    /** The shader to render far-away, solid-coloured points */
    std::shared_ptr<glowl::GLSLProgram> point_shader_;

    /** The shader to init the depth buffer with points */
    std::shared_ptr<glowl::GLSLProgram> init_depth_point_shader_;

    /** Von Guido aus */
    std::shared_ptr<glowl::GLSLProgram> vert_cnt_shader_;

    /** Von Guido aus */
    std::shared_ptr<glowl::GLSLProgram> vert_cnt_shader_2_;

    /** The frame buffer object for the depth estimate */
    vislib_gl::graphics::gl::FramebufferObject fbo_;

    /** buffers for depth-max mip map */
    vislib_gl::graphics::gl::FramebufferObject depthmap_[2];

    /** The call for data */
    core::CallerSlot get_data_slot_;

    /** The call for Transfer function */
    core::CallerSlot get_tf_slot_;

    /** The call for light sources */
    core::CallerSlot get_lights_slot_;

    /** Flag to activate per cell culling */
    core::param::ParamSlot use_cell_cull_slot_;

    /** Flag to activate per vertex culling */
    core::param::ParamSlot use_vert_cull_slot_;

    /** Flag to activate output of percentage of culled cells */
    core::param::ParamSlot speak_cell_perc_slot_;

    /** Flag to activate output of number of vertices */
    core::param::ParamSlot speak_vert_count_slot_;

    /** De-/Activates deferred shading with normal generation */
    core::param::ParamSlot deferred_shading_slot_;

    /** A simple black-to-white transfer function texture as fallback */
    unsigned int grey_tf_;

    /** Cell distances */
    std::vector<vislib::Pair<unsigned int, float>> cell_dists_;

    /** Cell rendering informations */
    std::vector<CellInfo> cell_infos_;

    /** Bytes of the GPU-Memory available for caching */
    SIZE_T cache_size_;

    /** Bytes of the GPU-Memory used by the caching */
    SIZE_T cache_size_used_;

    /** Frame buffer object used for deferred shading */
    vislib_gl::graphics::gl::FramebufferObject ds_fbo_;

    /** The sphere shader */
    std::shared_ptr<glowl::GLSLProgram> deferred_sphere_shader_;

    /** The vanilla sphere shader */
    std::shared_ptr<glowl::GLSLProgram> deferred_vanilla_sphere_shader_;

    /** The shader to render far-away, solid-coloured points */
    std::shared_ptr<glowl::GLSLProgram> deferred_point_shader_;

    /** The deferred shader */
    std::shared_ptr<glowl::GLSLProgram> deferred_shader_;

    /** The hash of the incoming data */
    SIZE_T inhash_;
};

} // namespace megamol::moldyn_gl::rendering

#endif /* MEGAMOL_MOLDYN_GRIMRENDERER_H_INCLUDED */
