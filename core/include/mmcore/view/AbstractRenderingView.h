/*
 * AbstractRenderingView.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_ABSTRACTRENDERINGVIEW_H_INCLUDED
#define MEGAMOLCORE_ABSTRACTRENDERINGVIEW_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include <glm/glm.hpp>

#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/AbstractView.h"
#include "vislib/sys/CriticalSection.h"
#include "vislib/graphics/FpsCounter.h"


namespace megamol {
namespace core {
namespace view {


/**
 * Abstract base class of rendering views
 */
class MEGAMOLCORE_API AbstractRenderingView : public AbstractView {
public:

    /**
     * Interface definition
     */
    class MEGAMOLCORE_API AbstractTitleRenderer {
    public:

        /** Ctor */
        AbstractTitleRenderer(void);

        /** Dtor */
        virtual ~AbstractTitleRenderer(void);

        /**
         * Create the renderer and allocates all resources
         *
         * @return True on success
         */
        virtual bool Create(void) = 0;

        /**
         * Renders the title scene
         *
         * @param tileX The view tile x coordinate
         * @param tileY The view tile y coordinate
         * @param tileW The view tile width
         * @param tileH The view tile height
         * @param virtW The virtual view width
         * @param virtH The virtual view height
         * @param stereo Flag if stereo rendering is to be performed
         * @param leftEye Flag if the stereo rendering is done for the left eye view
         * @param instTime The instance time code
         * @param core The core
         */
        virtual void Render(float tileX, float tileY, float tileW, float tileH,
            float virtW, float virtH, bool stereo, bool leftEye, double instTime,
            class ::megamol::core::CoreInstance *core) = 0;

        /**
         * Releases the renderer and all of its resources
         */
        virtual void Release(void) = 0;

    };

    /** Ctor. */
    AbstractRenderingView(void);

    /** Dtor. */
    virtual ~AbstractRenderingView(void);

    /**
    * Answer the background colour for the view
    *
    * @return The background colour for the view
    */
    glm::vec4 BkgndColour(void) const;

protected:

    /** Pointer to the override background colour */
    glm::vec4 overrideBkgndCol;

    glm::vec4 overrideViewport;

    /** The background colour for the view */
    mutable param::ParamSlot bkgndColSlot;

private:

    /** The background colour for the view */
    mutable glm::vec4 bkgndCol;

};


} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_ABSTRACTRENDERINGVIEW_H_INCLUDED */
