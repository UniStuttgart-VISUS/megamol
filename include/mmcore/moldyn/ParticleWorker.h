/*
 * ParticleWorker.h
 *
 * Copyright (C) 2013 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/moldyn/DirectionalParticleDataCall.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "vislib/RawStorage.h"
#include "vislib/types.h"

#include "vislib/graphics/gl/IncludeAllGL.h"
#include <GL/glu.h>

#include "vislib/graphics/gl/GLSLComputeShader.h"


namespace megamol {
namespace core {
namespace moldyn {


    /**
     * Module to filter calls with multiple particle lists (currently directional and spherical) by list index
     */
	class ParticleWorker : public Module {
    public:

		class VAOUnlocker : public AbstractGetDataCall::Unlocker
		{
		public:
			VAOUnlocker() { };
			virtual ~VAOUnlocker() { };
			void Unlock()
			{
				glBindVertexArray(0);
				glBindBufferARB (GL_SHADER_STORAGE_BUFFER, 0);
			};
		};

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "ParticleWorker";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Modify incoming particles";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
            return vislib::graphics::gl::GLSLShader::AreExtensionsAvailable()
                && ogl_IsVersionGEQ(4, 3);
        }

        /**
         * Disallow usage in quickstarts
         *
         * @return false
         */
        static bool SupportQuickstart(void) {
            return false;
        }

        /** Ctor. */
        ParticleWorker(void);

        /** Dtor. */
        virtual ~ParticleWorker(void);

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

    private:

        /**
         * Callback publishing the gridded data
         *
         * @param call The call requesting the gridded data
         *
         * @return 'true' on success, 'false' on failure
         */
        bool getDataCallback(Call& call);

        /**
         * Callback publishing the extend of the data
         *
         * @param call The call requesting the extend of the data
         *
         * @return 'true' on success, 'false' on failure
         */
        bool getExtentCallback(Call& call);

        CallerSlot inParticlesDataSlot;

        CalleeSlot outParticlesDataSlot;

		vislib::Array<GLuint> glVAO;
		vislib::Array<GLuint> glVB;
		vislib::Array<GLuint> glCB;
		
		GLuint glClusterInfos;
		vislib::graphics::gl::GLSLComputeShader shaderOnClusterComputation;

		/*
		GLuint glParticleList;
		GLuint glPrefixIn;
		GLuint glPrefixOut;
		
		vislib::graphics::gl::GLSLComputeShader shaderComputeInitParticleList;
		vislib::graphics::gl::GLSLComputeShader shaderComputeMakeParticleList;
		vislib::graphics::gl::GLSLComputeShader shaderComputeCompactToClusterList;
		vislib::graphics::gl::GLSLComputeShader shaderComputeGrid;
		vislib::graphics::gl::GLSLComputeShader shaderComputeGriddify;
		vislib::graphics::gl::GLSLComputeShader shaderComputePrefixSum;
		*/
    };

} /* end namespace moldyn */
} /* end namespace core */
} /* end namespace megamol */

