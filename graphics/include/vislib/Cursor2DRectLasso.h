/*
 * Cursor2DRectLasso.h
 *
 * Copyright (C) 2006 - 2009 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_CURSOR2DRECTLASSO_H_INCLUDED
#define VISLIB_CURSOR2DRECTLASSO_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/AbstractCursor2DEvent.h"
#include "vislib/graphicstypes.h"


namespace vislib {
namespace graphics {


    /**
     * A Cursor2DEvent implementing a selection lasso for rectangular regions.
     */
    class Cursor2DRectLasso : public AbstractCursor2DEvent {

    public:

        /** Ctor. */
        Cursor2DRectLasso(void);

        /** Dtor. */
        ~Cursor2DRectLasso(void);

        /**
         * Clears the rectangle.
         */
        void Clear(void);

        /**
         * Gets the rectangle.
         *
         * @return The rectangle
         */
        inline const ImageSpaceRectangle& Rectangle(void) const {
            return this->rect;
        }

        /**
         * Is called by an AbstractCursor which has this event in it's observer
         * list, if an event occured which is of interest to this 
         * AbstractCursorEvent because of the succeeded tests.
         *
         * @param caller The AbstractCursor object which calles this methode.
         * @param reason The reason why for this call.
         * @param param A reason depending parameter.
         */
        virtual void Trigger(AbstractCursor *caller, TriggerReason reason,
            unsigned int param);

    private:

        /** Flag whether the mouse draggs. */
        bool drag;

        /** The selected rectangle */
        ImageSpaceRectangle rect;

    };
    
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_CURSOR2DRECTLASSO_H_INCLUDED */

