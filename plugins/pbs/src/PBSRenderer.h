/*
* PBSRenderer.h
*
* Copyright (C) 2017 by VISUS (Universitaet Stuttgart)
* Alle Rechte vorbehalten.
*/

#ifndef MEGAMOL_PBS_PBSRENDERER_H_INCLUDED
#define MEGAMOL_PBS_PBSRENDERER_H_INCLUDED

#include "mmcore/Call.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/view/Renderer3DModule.h"
#include "mmcore/param/ParamSlot.h"

#include "glad/glad.h"

namespace megamol {
namespace pbs {

class PBSRenderer : public core::view::Renderer3DModule {
public:
    /**
    * Answer the name of this module.
    *
    * @return The name of this module.
    */
    static const char *ClassName(void) {
        return "PBSRenderer";
    }

    /**
    * Answer a human readable description of this module.
    *
    * @return A human readable description of this module.
    */
    static const char *Description(void) {
        return "Renderer for sphere glyphs from PBS files.";
    }

    /**
    * Answers whether this module is available on the current system.
    *
    * @return 'true' if the module is available, 'false' otherwise.
    */
    static bool IsAvailable(void) {
        return gladLoadGL();
    }

    /** Ctor. */
    PBSRenderer(void);

    /** Dtor. */
    virtual ~PBSRenderer(void);

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
    * The render callback.
    *
    * @param call The calling call.
    *
    * @return The return value of the function.
    */
    virtual bool Render(core::Call &call);

    // Inherited via Renderer3DModule
    virtual bool GetExtents(core::Call &call) override;

private:
    bool printShaderInfoLog(GLuint shader) const;

    bool printProgramInfoLog(GLuint shaderProg) const;

    core::CallerSlot getDataSlot;

    core::param::ParamSlot radiusParamSlot;

    GLuint x_buffer, y_buffer, z_buffer;

    GLint max_ssbo_size = 0;

    const GLuint x_buffer_base = 1, y_buffer_base = 2, z_buffer_base = 3;

    GLuint shader;
}; /* end class PBSRenderer */

} /* end namespace pbs */
} /* end namespace megamol */

#endif /* end ifndef MEGAMOL_PBS_PBSRENDERER_H_INCLUDED */