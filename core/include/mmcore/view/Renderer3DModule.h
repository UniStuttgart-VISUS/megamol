/*
 * Renderer3DModule.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_RENDERER3DMODULE_H_INCLUDED
#define MEGAMOLCORE_RENDERER3DMODULE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/view/RendererModule.h"
#include "mmcore/view/CallRender3D.h"

namespace megamol {
namespace core {
namespace view {


    /**
     * Base class of rendering graph 3D renderer modules.
     */
    class MEGAMOLCORE_API Renderer3DModule : public RendererModule<CallRender3D>{
    public:

        /** Ctor. */
        Renderer3DModule(void);

        /** Dtor. */
        virtual ~Renderer3DModule(void) = default;

    protected:
		/**
         * The get extents callback. The module should set the members of
         * 'call' to tell the caller the extents of its data (bounding boxes
         * and times).
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        [[deprecated("Use CallRender3D version instead")]]
        virtual bool GetExtents(Call& call) { return false; };

		virtual bool GetExtents(CallRender3D& call) override { return this->GetExtents(static_cast<Call&>(call)); }

        /**
         * The render callback.
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
		[[deprecated("Use CallRender3D version instead")]]
        virtual bool Render(Call& call) { return false; };

		virtual bool Render(CallRender3D& call) override { return this->Render(static_cast<Call&>(call)); }
    };


} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_RENDERERMODULE_H_INCLUDED */
