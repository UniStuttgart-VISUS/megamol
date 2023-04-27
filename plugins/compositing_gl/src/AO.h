#pragma once


//#include "geometry_calls/MultiParticleDataCall.h"
//#include "geometry_calls/VolumetricDataCall.h"

//#include "misc/MDAOVolumeGenerator.h"
#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmstd_gl/ModuleGL.h"

#include "mmcore_gl/utility/ShaderFactory.h"

//#include "mmcore/CoreInstance.h"

namespace megamol::compositing_gl {

class AO : public mmstd_gl::ModuleGL {

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
    virtual ~AO(void);

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);

    /**
     * Implementation of 'Release'.
     */
    virtual void release(void);

    /**
     * Generates Voxels.
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool getDataCallback(core::Call& call);

    bool getMetadataCallback(core::Call& call);

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
    core::CalleeSlot output_tex_slot_; // VolumetricDataCall (left slot)

    core::CallerSlot voxels_tex_slot_; // MultiParticleDataCall (right slot)

    GLuint texture_handle;
    GLuint voxel_handle;

    //geocalls::VolumetricDataCall::Metadata metadata;

    core::param::ParamSlot vol_size_slot_;

    //misc::MDAOVolumeGenerator* vol_gen_;

    GLuint vertex_array_;

    GLuint vbo_;

    std::unique_ptr<msf::ShaderFactoryOptionsOpenGL> shader_options_flags_;

    std::shared_ptr<glowl::GLSLProgram> sphere_prgm_;
};

} // namespace megamol::compositing_gl
