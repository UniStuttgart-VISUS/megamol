/*
 * Renderer2DModule.h
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_RENDERER2DMODULE_H_INCLUDED
#define MEGAMOLCORE_RENDERER2DMODULE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "CallRender2DGL.h"
#include "mmcore/view/RendererModule.h"


namespace megamol {
namespace core {
namespace view {

/**
 * Base class of rendering graph 2D renderer modules.
 */
class MEGAMOLCORE_API Renderer2DModule : public RendererModule<CallRender2DGL> {
public:
    /** Ctor. */
    Renderer2DModule() :  RendererModule<CallRender2DGL>() {
	    this->MakeSlotAvailable(&this->renderSlot);
	}

    /** Dtor. */
    virtual ~Renderer2DModule(void) = default;
};

} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_RENDERERMODULE_H_INCLUDED */
