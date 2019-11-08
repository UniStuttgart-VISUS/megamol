/*
 * ProbeBillboardGlyphMaterial.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef PROBE_BILLBOARD_GLYPH_MATERIAL_H_INCLUDED
#define PROBE_BILLBOARD_GLYPH_MATERIAL_H_INCLUDED

#include "mesh/AbstractGPUMaterialDataSource.h"

namespace megamol {
namespace probe_gl {

class ProbeBillboardGlyphMaterial : public mesh::AbstractGPUMaterialDataSource {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "ProbeBillboardGlyphMaterial"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "..."; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

    ProbeBillboardGlyphMaterial();
    ~ProbeBillboardGlyphMaterial();

protected:
    bool create();

    bool getDataCallback(core::Call& caller);

    bool getMetaDataCallback(core::Call& caller);

private:
    typedef glowl::GLSLProgram ShaderProgram;

    /** Shader program for deferred shading pass */
    std::shared_ptr<ShaderProgram> m_billboard_glyph_prgm;

    core::CallerSlot m_glyph_images_slot;
    size_t m_glyph_images_slot_cached_hash;
};


} // probe_gl
} // namespace megamol


#endif // !PROBE_BILLBOARD_GLYPH_MATERIAL_H_INCLUDED
