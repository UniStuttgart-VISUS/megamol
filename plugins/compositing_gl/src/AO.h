#pragma once

#include "geometry_calls/VolumetricDataCall.h"
#include "glowl/glowl.h"
#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore_gl/utility/ShaderFactory.h"
#include "mmstd_gl/renderer/Renderer3DModuleGL.h"


#define _USE_MATH_DEFINES
#include <math.h>

#include <vector>

namespace megamol::compositing_gl {

class AO : public megamol::mmstd_gl::Renderer3DModuleGL {

public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "AO";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "AO";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static inline bool IsAvailable(void) {
        return true;
    }

    /**
     * Initialises a new instance.
     */
    AO(void);

    /**
     * Finalises an instance.
     */
    ~AO(void);

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create(void);


    bool GetExtents(mmstd_gl::CallRender3DGL& call) override;

    /**
     * Implementation of 'Release'.
     */
    void release(void);

    bool Render(mmstd_gl::CallRender3DGL& call) override;

private:
    
    core::CalleeSlot output_tex_slot_; // Ambient occlusion texture (left slot)

    core::CallerSlot voxels_tex_slot_; // VolumetricDataCall (right slot)

    core::CallerSlot get_lights_slot_;

    /** Slot for querying normals render target texture, i.e. a rhs connection */
    core::CallerSlot normals_tex_slot_;

    /** Slot for querying depth render target texture, i.e. a rhs connection */
    core::CallerSlot depth_tex_slot_;

    /** Slot for querying color render target texture, i.e. a rhs connection */
    core::CallerSlot color_tex_slot_;

    /** Slot for querying camera, i.e. a rhs connection */
    core::CallerSlot camera_slot_;


    GLuint texture_handle_;
    GLuint voxel_handle_;
    GLuint vertex_array_;
    GLuint vbo_;

    std::shared_ptr<glowl::Texture2D> color_tex_;
    std::shared_ptr<glowl::Texture2D> depth_tex_;
    std::shared_ptr<glowl::Texture2D> normal_tex_;

    glm::mat4 cur_mvp_inv_;
    glm::vec3 cur_cam_pos_;
    glm::vec4 cur_light_dir_;

    vislib::math::Cuboid<float> cur_clip_box_;

    int cur_vp_width_;
    int cur_vp_height_;

    std::unique_ptr<msf::ShaderFactoryOptionsOpenGL> shader_options_flags_;

    std::shared_ptr<glowl::GLSLProgram> lighting_prgm_;

    std::unique_ptr<glowl::BufferObject> ao_dir_ubo_;

    core::param::ParamSlot vol_size_slot_;
    core::param::ParamSlot ao_cone_apex_slot_;
    core::param::ParamSlot enable_lighting_slot_;
    core::param::ParamSlot ao_offset_slot_;
    core::param::ParamSlot ao_strength_slot_;
    core::param::ParamSlot ao_cone_length_slot_;

    /**
    * render deferreded pass with volume, depth, normal and color texture
    */
    void renderAmbientOcclusion();

    /**
    * generate cone directions to be used in AO shader
    */
    void generate3ConeDirections(std::vector<glm::vec4>& directions, float apex);

    /**
    * get volume data from volumetric data call
    */
    bool updateVolumeData(const unsigned int frameID);

    /**
    * recreate resources like shaders and buffers
    */
    bool recreateResources(void);
};

} // namespace megamol::compositing_gl
