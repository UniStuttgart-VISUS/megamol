/*
 * AbstractMultiShaderQuartzRenderer.h
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "AbstractQuartzRenderer.h"
#include "vislib_gl/graphics/gl/GLSLShader.h"


namespace megamol {
namespace demos_gl {

/**
 * AbstractQuartzRenderer
 */
class AbstractMultiShaderQuartzRenderer : public AbstractQuartzRenderer {
public:
    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    /**
     * Ctor
     */
    AbstractMultiShaderQuartzRenderer(void);

    /**
     * Dtor
     */
    virtual ~AbstractMultiShaderQuartzRenderer(void);

protected:
    /**
     * Answer the crystalite data from the connected module
     *
     * @return The crystalite data from the connected module or NULL if no
     *         data could be received
     */
    virtual CrystalDataCall* getCrystaliteData(void);

    /**
     * Releases all shader objects
     */
    void releaseShaders(void);

    /**
     * Creates a raycasting shader for the specified crystalite
     *
     * @param c The crystalite
     *
     * @return The shader
     */
    virtual vislib_gl::graphics::gl::GLSLShader* makeShader(const CrystalDataCall::Crystal& c) = 0;

    /** The number of shader slots */
    unsigned int cntShaders;

    /** The crystalite shaders */
    vislib_gl::graphics::gl::GLSLShader** shaders;

    /** The error shader indicating that the correct shader is not yet loaded */
    vislib_gl::graphics::gl::GLSLShader errShader;
};

} // namespace demos_gl
} /* end namespace megamol */
