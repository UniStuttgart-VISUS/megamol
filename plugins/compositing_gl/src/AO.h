#pragma once


//#include "geometry_calls/MultiParticleDataCall.h"
//#include "geometry_calls/VolumetricDataCall.h"

//#include "misc/MDAOVolumeGenerator.h"
#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmstd_gl/renderer/Renderer3DModuleGL.h"
#include "mmcore_gl/utility/ShaderFactory.h"
#include "geometry_calls/VolumetricDataCall.h"
#include "mmcore/param/BoolParam.h"
#include "glowl/glowl.h"
#include "mmcore/param/FloatParam.h"



#define _USE_MATH_DEFINES
#include <math.h>

#include <vector>

//#include "mmcore/CoreInstance.h"

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
    ///**
    // * TODO: Document
    // *
    // * @param t           ...
    // * @param outScaling  ...
    // *
    // * @return Pointer to MultiParticleDataCall ...
    // */
    //MultiParticleDataCall* getData(unsigned int t, float& out_scaling);

    /** The slot that requests the data. */
    core::CalleeSlot output_tex_slot_; // Ambient occlusion texture (left slot)

    core::CallerSlot voxels_tex_slot_; // VolumetricDataCall (right slot)

    GLuint texture_handle;
    GLuint voxel_handle;

    std::shared_ptr<glowl::Texture2D> color_tex;
    std::shared_ptr<glowl::Texture2D> depth_tex;
    std::shared_ptr<glowl::Texture2D> normal_tex;
    
    glm::mat4 cur_mvp_inv_;

    int cur_vp_width_;
    int cur_vp_height_;

    //geocalls::VolumetricDataCall::Metadata metadata;

    core::param::ParamSlot vol_size_slot_;

    //misc::MDAOVolumeGenerator* vol_gen_;

    GLuint vertex_array_;

    GLuint vbo_;

    std::unique_ptr<msf::ShaderFactoryOptionsOpenGL> shader_options_flags_;

    std::shared_ptr<glowl::GLSLProgram> lighting_prgm_;

    std::unique_ptr<glowl::BufferObject> ao_dir_ubo_;

    megamol::core::param::ParamSlot ao_cone_apex_slot_;
    megamol::core::param::ParamSlot enable_lighting_slot_;
    megamol::core::param::ParamSlot ao_offset_slot_;
    megamol::core::param::ParamSlot ao_strength_slot_;
    megamol::core::param::ParamSlot ao_cone_length_slot_;

    /** Slot for querying normals render target texture, i.e. a rhs connection */
    megamol::core::CallerSlot normals_tex_slot_;

    /** Slot for querying depth render target texture, i.e. a rhs connection */
    megamol::core::CallerSlot depth_tex_slot_;

    /** Slot for querying color render target texture, i.e. a rhs connection */
    megamol::core::CallerSlot color_tex_slot_;

    /** Slot for querying camera, i.e. a rhs connection */
    megamol::core::CallerSlot camera_slot_;

    void renderAmbientOcclusion();

    void generate3ConeDirections(std::vector<glm::vec4>& directions, float apex);

    bool updateVolumeData(const unsigned int frameID);
};

} // namespace megamol::compositing_gl
