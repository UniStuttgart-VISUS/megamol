/*
 * BeholderRotator2D.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_BEHOLDERROTATOR2D_H_INCLUDED
#define VISLIB_BEHOLDERROTATOR2D_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/AbstractBeholderController.h"
#include "vislib/AbstractCursor2DEvent.h"
#include "vislib/graphicstypes.h"
#include "vislib/InputModifiers.h"


namespace vislib {
namespace graphics {


    /**
     * Controller class translating cursor 2D movement to Beholder rotations.
     * You should set up a button test and you should set the alternative 
     * rotation modifier if you use this class.
     * Normal rotation schema modelles rotations about the pitch and yaw axes, 
     * while the alternative rotation schema modelles rotations about the roll
     * axis.
     */
    class BeholderRotator2D: public AbstractCursor2DEvent, public AbstractBeholderController {

    public:

        /** ctor */
        BeholderRotator2D(void);

        /** Dtor. */
        virtual ~BeholderRotator2D(void);

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
        inline void SetAltModifier(InputModifiers::Modifier modifier) {
            this->altMod = modifier;
        }

        /**
         * Returns the modifier for alternative rotation schema.
         *
         * @return The modifier.
         */
        inline InputModifiers::Modifier GetAltModifier(void) const {
            return this->altMod;
        }

    private:

        /** The modifier for alternate rotation */
        InputModifiers::Modifier altMod;

        /** Flag whether the mouse draggs. */
        bool drag;

    };
    
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_BEHOLDERROTATOR2D_H_INCLUDED */
