/*
 * AbstractMultiShaderQuartzRenderer.h
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "AbstractQuartzRenderer.h"
#include "vislib/graphics/gl/GLSLShader.h"


namespace megamol {
namespace demos {

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
            return vislib::graphics::gl::GLSLShader::AreExtensionsAvailable();
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
        virtual CrystalDataCall *getCrystaliteData(void);

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
        virtual vislib::graphics::gl::GLSLShader* makeShader(const CrystalDataCall::Crystal& c) = 0;

        /** The number of shader slots */
        unsigned int cntShaders;

        /** The crystalite shaders */
        vislib::graphics::gl::GLSLShader **shaders;

        /** The error shader indicating that the correct shader is not yet loaded */
        vislib::graphics::gl::GLSLShader errShader;

    };

} /* end namespace demos */
} /* end namespace megamol */
