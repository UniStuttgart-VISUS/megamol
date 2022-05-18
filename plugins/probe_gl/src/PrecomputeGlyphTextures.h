/*
 * InjectClusterID.h
 * Copyright (C) 2020 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#ifndef PRECOMPUTE_GLYPH_TEXTURES_H_INCLUDED
#define PRECOMPUTE_GLPYH_TEXTURES_H_INCLUDED

#include "glowl/GLSLProgram.hpp"

#include "mesh/MeshCalls.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"

namespace megamol {
namespace probe_gl {

class PrecomputeGlyphTextures : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "PrecomputeGlyphTextures";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "...";
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
    PrecomputeGlyphTextures(void);

    /** Dtor. */
    virtual ~PrecomputeGlyphTextures(void);

protected:
    virtual bool create();
    virtual void release();

    core::CallerSlot _probes_rhs_slot;
    core::CalleeSlot _textures_lhs_slot;

private:
    bool getMetaData(core::Call& call);
    bool getData(core::Call& call);

    std::shared_ptr<mesh::ImageDataAccessCollection> _glyph_texture_data;

    std::shared_ptr<glowl::GLSLProgram> _glyph_texture_compute_shader;

    uint32_t _version = 0;
};

} // namespace probe_gl
} // namespace megamol

#endif // !PRECOMPUTE_GLYPH_TEXTURES_H_INCLUDED
