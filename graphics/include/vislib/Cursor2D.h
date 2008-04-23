/*
 * Cursor2D.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_CURSOR2D_H_INCLUDED
#define VISLIB_CURSOR2D_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/AbstractCursor.h"
#include "vislib/CameraParameters.h"
#include "vislib/graphicstypes.h"
#include "vislib/SmartPtr.h"


namespace vislib {
namespace graphics {

    /* forward declarations */
    class AbstractCursorEvent;
    class AbstractCursor2DEvent;


    /**
     * Class modelling a two dimensional cursor, like a pc mouse.
     */
    class Cursor2D: public AbstractCursor {
    public:

        /** ctor */
        Cursor2D(void);

        /**
         * copy ctor
         *
         * @param rhs Sourc object.
         */
        Cursor2D(const Cursor2D& rhs);

        /** Dtor. */
        virtual ~Cursor2D(void);

        /**
         * Sets the position of the cursor in virtual image space of the 
         * associated camera. The previous position of the cursor is also 
         * changed.
         *
         * @param x The new x coordinate
         * @param y The new y coordinate
         * @param flipY Indicates whether to flip the y coordinate accordingly
         *              to the associated camera. This is used to ensure the
         *              origin is placed in the lower left corner of the 
         *              window.
         */
        void SetPosition(ImageSpaceType x, ImageSpaceType y, bool flipY);

        /**
         * Assignment operator
         *
         * @param rhs Sourc object.
         *
         * @return Reference to this.
         */
        Cursor2D& operator=(const Cursor2D& rhs);

        /**
         * Behaves like AbstractCursor::RegisterCursorEvent.
         *
         * @param cursorEvent The cursor event to be added.
         */
        virtual void RegisterCursorEvent(AbstractCursor2DEvent *cursorEvent);

        /**
         * Associates a camera parameters object with this cursor.
         *
         * @param cameraParams The camera parameters object.
         */
        void SetCameraParams(SmartPtr<CameraParameters> cameraParams);

        /**
         * Returns the associated camera parameters object.
         *
         * @return The associated camera parameters object.
         */
        inline SmartPtr<CameraParameters> CameraParams(void) {
            return this->camPams;
        }

        /**
         * Returns the x coordinate of the cursor.
         *
         * @return The x coordinate.
         */
        inline ImageSpaceType X(void) const {
            return this->x;
        }

        /**
         * Returns the y coordinate of the cursor.
         *
         * @return The y coordinate.
         */
        inline ImageSpaceType Y(void) const {
            return this->y;
        }

        /**
         * Returns the previous x coordinate of the cursor. The pervious 
         * position of the cursor is set by cursor movements.
         *
         * @return The previous x coordinate.
         */
        inline ImageSpaceType PreviousX(void) const {
            return this->prevX;
        }

        /**
         * Returns the previous y coordinate of the cursor. The pervious 
         * position of the cursor is set by cursor movements.
         *
         * @return The previous y coordinate.
         */
        inline ImageSpaceType PreviousY(void) const {
            return this->prevY;
        }

    private:

        /** x position of the cursor space */
        ImageSpaceType x;

        /** y position of the cursor space */
        ImageSpaceType y;

        /** previous x position of the cursor space */
        ImageSpaceType prevX;

        /** previous y position of the cursor space */
        ImageSpaceType prevY;

        /** The parameters object of the associated camera */
        SmartPtr<CameraParameters> camPams;


    };

} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_CURSOR2D_H_INCLUDED */
