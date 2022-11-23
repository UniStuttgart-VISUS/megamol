/*
 * AbstractMultiShaderQuartzRenderer.h
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include <memory>

#include <glowl/glowl.h>

#include "AbstractQuartzRenderer.h"


namespace megamol::demos_gl {

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
    static bool IsAvailable() {
        return true;
    }

    /**
     * Ctor
     */
    AbstractMultiShaderQuartzRenderer();

    /**
     * Dtor
     */
    virtual ~AbstractMultiShaderQuartzRenderer();

protected:
    /**
     * Answer the crystalite data from the connected module
     *
     * @return The crystalite data from the connected module or NULL if no
     *         data could be received
     */
    virtual CrystalDataCall* getCrystaliteData();

    /**
     * Releases all shader objects
     */
    void releaseShaders();

    /**
     * Creates a raycasting shader for the specified crystalite
     *
     * @param c The crystalite
     *
     * @return The shader
     */
    virtual std::shared_ptr<glowl::GLSLProgram> makeShader(const CrystalDataCall::Crystal& c) = 0;

    /** The crystalite shaders */
    std::vector<std::shared_ptr<glowl::GLSLProgram>> shaders;

    /** The error shader indicating that the correct shader is not yet loaded */
    std::shared_ptr<glowl::GLSLProgram> errShader;
};

} // namespace megamol::demos_gl
