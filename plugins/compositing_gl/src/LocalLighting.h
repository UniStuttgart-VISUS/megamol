/*
 * LocalLighting.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#pragma once

#include <glowl/glowl.h>

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmstd_gl/renderer/CallRender3DGL.h"
#include "mmstd_gl/renderer/Renderer3DModuleGL.h"

namespace megamol::compositing_gl {

class LocalLighting : public core::Module {
public:
    struct LightParams {
        float x, y, z, intensity;
    };

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "LocalLighting";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Compositing module that computes local lighting";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    LocalLighting();
    ~LocalLighting();

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create();

    /**
     * Implementation of 'Release'.
     */
    void release();

    /**
     * TODO
     */
    bool getDataCallback(core::Call& caller);

    /**
     * TODO
     */
    bool getMetaDataCallback(core::Call& caller);

private:
    uint32_t m_version;

    /** Shader program for texture add (Lambert Illumination) */
    std::unique_ptr<glowl::GLSLProgram> m_lambert_prgm;

    /** Shader program for texture add (Blinn-Phong Illumination) */
    std::unique_ptr<glowl::GLSLProgram> m_phong_prgm;

    /** Shader program for texture add (Blinn-Phong Illumination) */
    std::unique_ptr<glowl::GLSLProgram> m_toon_prgm;

    /** Texture that the lighting result will be written to */
    std::shared_ptr<glowl::Texture2D> m_output_texture;

    //TODO add same thing for Ambient Light as for point & distant light

    /** GPU buffer object for making active (point)lights available in during shading pass */
    std::unique_ptr<glowl::BufferObject> m_point_lights_buffer;

    /** GPU buffer object for making active (distant)lights available in during shading pass */
    std::unique_ptr<glowl::BufferObject> m_distant_lights_buffer;

    /** buffered light information */
    std::vector<LightParams> m_point_lights, m_distant_lights;


    /**Parameter for different illuminations e.g. Lambertian, Phong */
    megamol::core::param::ParamSlot m_illuminationmode;

    /**Parameters for Blinn-Phong Illumination*/
    megamol::core::param::ParamSlot m_phong_ambientColor;
    megamol::core::param::ParamSlot m_phong_diffuseColor;
    megamol::core::param::ParamSlot m_phong_specularColor;

    megamol::core::param::ParamSlot m_phong_k_ambient;
    megamol::core::param::ParamSlot m_phong_k_diffuse;
    megamol::core::param::ParamSlot m_phong_k_specular;
    megamol::core::param::ParamSlot m_phong_k_exp;

    megamol::core::param::ParamSlot m_toon_exposure_avg_intensity;
    megamol::core::param::ParamSlot m_toon_roughness;

    /** Slot for requesting the output textures from this module, i.e. lhs connection */
    megamol::core::CalleeSlot m_output_tex_slot;

    /** Slot for querying albedo color buffer texture, i.e. a rhs connection */
    megamol::core::CallerSlot m_albedo_tex_slot;

    /** Slot for querying normal buffer texture, i.e. a rhs connection */
    megamol::core::CallerSlot m_normal_tex_slot;

    /** Slot for querying normal buffer texture, i.e. a rhs connection */
    megamol::core::CallerSlot m_depth_tex_slot;

    /** Slot for querying texture that contain roughness and metalness channel, i.e. a rhs connection */
    megamol::core::CallerSlot m_roughness_metalness_tex_slot;

    /** Slot to retrieve the light information */
    megamol::core::CallerSlot m_lightSlot;

    /** Slot for querying camera, i.e. a rhs connection */
    megamol::core::CallerSlot m_camera_slot;
};

} // namespace megamol::compositing_gl
