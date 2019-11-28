/*
 * ProbeBillboardGlyphRenderTasks.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef PROBE_BILLBOARD_GLYPH_RENDER_TASK_H_INCLUDED
#define PROBE_BILLBOARD_GLYPH_RENDER_TASK_H_INCLUDED

#include "mesh/AbstractGPURenderTaskDataSource.h"

namespace megamol {
namespace probe_gl {

class ProbeBillboardGlyphRenderTasks : public mesh::AbstractGPURenderTaskDataSource {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "ProbeBillboardGlyphRenderTasks"; }

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

    ProbeBillboardGlyphRenderTasks();
    ~ProbeBillboardGlyphRenderTasks();

protected:
    bool getDataCallback(core::Call& caller);

    bool getMetaDataCallback(core::Call& caller);

    core::param::ParamSlot m_billboard_size_slot;

private:
    core::CallerSlot m_probes_slot;
    size_t m_probes_cached_hash;

    std::shared_ptr<glowl::Mesh> m_billboard_dummy_mesh;
};

} // namespace probe_gl
} // namespace megamol


#endif // !PROBE_BILLBOARD_GLYPH_RENDER_TASK_H_INCLUDED
