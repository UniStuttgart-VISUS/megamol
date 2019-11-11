/*
 * LocalLighting.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef LOCAL_LIGHTING_H_INCLUDED
#define LOCAL_LIGHTING_H_INCLUDED

#include "compositing/compositing_gl.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/view/CallRender3D_2.h"
#include "mmcore/view/Renderer3DModule_2.h"

#include "vislib/graphics/gl/GLSLComputeShader.h"

#include "glowl/BufferObject.hpp"
#include "glowl/Texture2D.hpp"

namespace megamol {
namespace compositing {

class COMPOSITING_GL_API LocalLighting : public core::Module {
public:
    struct LightParams {
        float x, y, z, intensity;
    };

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() { return "LocalLighting"; }

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
    static bool IsAvailable() { return true; }

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

    /**
     * Receives the current lights from the light call and writes them to the lightMap
     *
     * @return True if any light has changed, false otherwise.
     */
    bool GetLights(void);

private:
    typedef vislib::graphics::gl::GLSLComputeShader GLSLComputeShader;

    /** Shader program for texture add */
    std::unique_ptr<GLSLComputeShader> m_lighting_prgm;

    /** Texture that the lighting result will be written to */
    std::shared_ptr<glowl::Texture2D> m_output_texture;

    /** GPU buffer object for making active (point)lights available in during shading pass */
    std::unique_ptr<glowl::BufferObject> m_point_lights_buffer;

    /** GPU buffer object for making active (distant)lights available in during shading pass */
    std::unique_ptr<glowl::BufferObject> m_distant_lights_buffer;

    /** map to store the called lights */
    core::view::light::LightMap m_light_map;

    /** buffered light information */
    std::vector<LightParams> m_point_lights, m_distant_lights;


    //TODO mode slot for LAMBERT, PHONG, etc.

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

} // namespace compositing
} // namespace megamol


#endif // !LOCAL_LIGHTING_H_INCLUDED
