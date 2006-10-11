/*
 * Cursor2D.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_CURSOR2D_H_INCLUDED
#define VISLIB_CURSOR2D_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


#include "vislib/AbstractCursor.h"
#include "vislib/graphicstypes.h"


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
         * Sets the size of the cursor space. Keep in mind, that the cursor 
         * position will not be clamped to these bounds, or otherwise changed.
         *
         * @param width The new width of the cursor space.
         * @param height The new height of the cursor space.
         */
        void SetSize(CursorSpaceType width, CursorSpaceType height);

        /**
         * Sets the position of the cursor in cursor space.
         *
         * @param x The new x coordinate
         * @param y The new y coordinate
         */
        void SetPosition(CursorSpaceType x, CursorSpaceType y);

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

    private:

        /** width of the cursor space */
        CursorSpaceType width;

        /** height of the cursor space */
        CursorSpaceType height;

        /** x position of the cursor space */
        CursorSpaceType x;

        /** y position of the cursor space */
        CursorSpaceType y;

    };

} /* end namespace graphics */
} /* end namespace vislib */

#endif /* VISLIB_CURSOR2D_H_INCLUDED */
