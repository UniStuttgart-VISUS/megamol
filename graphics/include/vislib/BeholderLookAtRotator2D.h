/*
 * BeholderLookAtRotator2D.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_BEHOLDERLOOKATROTATOR2D_H_INCLUDED
#define VISLIB_BEHOLDERLOOKATROTATOR2D_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */

#include "vislib/AbstractBeholderController.h"
#include "vislib/AbstractCursor2DEvent.h"
#include "vislib/graphicstypes.h"


namespace vislib {
namespace graphics {

    /**
     * Controller class translating cursor 2D movement to Beholder rotations
     * about the look at point of the beholder.
     * You should set up a button test and you should set the alternative 
     * rotation modifier if you use this class.
     * Normal rotation schema modelles rotations about the pitch and yaw axes, 
     * while the alternative rotation schema modelles rotations about the roll
     * axis.
     */
    class BeholderLookAtRotator2D : public AbstractCursor2DEvent, public AbstractBeholderController {

    public:

        /** ctor */
        BeholderLookAtRotator2D(void);

        /** Dtor. */
        virtual ~BeholderLookAtRotator2D(void);

        /**
         * Callback methode called by a Cursor2D this event is registered to 
         * when a cursor event occures. REASON_BUTTON_DOWN, REASON_MOVE and
         * REASON_BUTTON_UP control the mouse position tracking and the 
         * rotation of the beholder. Othere calling reasons have no effect.
         *
         * @param caller The AbstractCursor object which calles this methode.
         * @param reason The reason why for this call.
         * @param param A reason depending parameter.
         */
        virtual void Trigger(AbstractCursor *caller, TriggerReason reason, unsigned int param);

        /**
         * Sets the modifier for alternative rotation schema. The normal 
         * rotation schema modelles pitch and yaw rotations, while the 
         * alternative rotation schema modelles roll rotations.
         *
         * @param modifier The modifier.
         */
        inline void SetAltModifier(unsigned int modifier) {
            this->altMod = modifier;
        }

        /**
         * Returns the modifier for alternative rotation schema.
         *
         * @return The modifier.
         */
        inline unsigned int GetAltModifier(void) const {
            return this->altMod;
        }

    private:

        /** The modifier for alternate rotation */
        unsigned int altMod;

        /** Flag whether the mouse draggs. */
        bool drag;

        /** drag source x coordinate */
        ImageSpaceType dragX;

        /** drag source y coordinate */
        ImageSpaceType dragY;

    };
    
} /* end namespace graphics */
} /* end namespace vislib */

#endif /* VISLIB_BEHOLDERLOOKATROTATOR2D_H_INCLUDED */
