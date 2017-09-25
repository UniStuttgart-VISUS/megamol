/*
 * AbstractRendererDeferred3D.h
 *
 * Copyright (C) 2011 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten..
 */

#ifndef MEGAMOLCORE_ABSTRACTRENDERERDEFERRED3D_H_INCLUDED
#define MEGAMOLCORE_ABSTRACTRENDERERDEFERRED3D_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */


#include "Renderer3DModule.h"
#include "mmcore/CallerSlot.h"


namespace megamol {
namespace core {
namespace view {


    /*
     * Base class of rendering graph renderer modules using deferred shading.
     */
    class MEGAMOLCORE_API AbstractRendererDeferred3D : public Renderer3DModule {
    public:

        /** Ctor. */
        AbstractRendererDeferred3D(void);

        /** Dtor. */
        virtual ~AbstractRendererDeferred3D(void);

    protected:

        /** Slot to call another renderer to render to the provided target. */
        CallerSlot rendererSlot;

    };


} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_ABSTRACTRENDERERDEFERRED3D_H_INCLUDED */
