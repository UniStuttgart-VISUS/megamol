/*
 * CartoonRenderer.h
 *
 * Copyright (C) 2010 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef MMPROTEINPLUGIN_CARTOONRENDERER_H_INCLUDED
#define MMPROTEINPLUGIN_CARTOONRENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/moldyn/MolecularDataCall.h"
#include "mmcore/moldyn/BindingSiteCall.h"
#include "Color.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/view/Renderer3DModuleDS.h"
#include "mmcore/view/AbstractCallRender3D.h"

#include "vislib/graphics/gl/GLSLShader.h"
#include "vislib/graphics/gl/GLSLGeometryShader.h"


namespace megamol {
namespace protein {

    /*
     * Cartoon Renderer class
     */

    class CartoonRenderer : public megamol::core::view::Renderer3DModuleDS
    {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void)
        {
            return "CartoonRenderer";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void)
        {
            return "Offers cartoon renderings for biomolecules.";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void)
        {
            return true;
        }

        /** Ctor. */
        CartoonRenderer(void);

        /** Dtor. */
        virtual ~CartoonRenderer(void);

    protected:

        /**
         * Implementation of 'Create'.
         *
         * @return 'true' on success, 'false' otherwise.
         */
        virtual bool create(void);

        /**
         * Implementation of 'release'.
         */
        virtual void release(void);

    private:

       /**********************************************************************
        * 'render'-functions
        **********************************************************************/

        /**
         * The get capabilities callback. The module should set the members
         * of 'call' to tell the caller its capabilities.
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        virtual bool GetCapabilities( megamol::core::Call& call);

        /**
         * The get extents callback. The module should set the members of
         * 'call' to tell the caller the extents of its data (bounding boxes
         * and times).
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        virtual bool GetExtents( megamol::core::Call& call);

        /**
         * The Open GL Render callback.
         *
         * @param call The calling call.
         * @return The return value of the function.
         */
        virtual bool Render( megamol::core::Call& call);

        /**
         * Update all parameter slots.
         *
         * @param mol   Pointer to the data call.
         */
		void UpdateParameters(const megamol::core::moldyn::MolecularDataCall *mol, const core::moldyn::BindingSiteCall *bs = 0);


        /**********************************************************************
         * variables
         **********************************************************************/
        
        /** MolecularDataCall caller slot */
        megamol::core::CallerSlot molDataCallerSlot;
        /** BindingSiteCall caller slot */
        megamol::core::CallerSlot bsDataCallerSlot;

        /** camera information */
        vislib::SmartPtr<vislib::graphics::CameraParameters> cameraInfo;

        /** shader for the spheres (raycasting view) */
        vislib::graphics::gl::GLSLShader sphereShader;
    };


} /* end namespace protein */
} /* end namespace megamol */

#endif // MMPROTEINPLUGIN_CARTOONRENDERER_H_INCLUDED
