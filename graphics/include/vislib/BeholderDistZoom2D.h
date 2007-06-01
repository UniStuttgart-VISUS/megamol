/*
 * BeholderDistZoom2D.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_BEHOLDERDISTZOOM2D_H_INCLUDED
#define VISLIB_BEHOLDERDISTZOOM2D_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/AbstractBeholderController.h"
#include "vislib/AbstractCursor2DEvent.h"
#include "vislib/graphicstypes.h"


namespace vislib {
namespace graphics {


    /**
     * Controller class translating cursor 2D movement along the y axis to 
     * distance changes of the Beholder position relative to the look ato 
     * position.
     */
    class BeholderDistZoom2D : public AbstractCursor2DEvent, public AbstractBeholderController {

    public:

        /** Ctor. */
        BeholderDistZoom2D(void);

        /** Dtor. */
        ~BeholderDistZoom2D(void);

        /**
         * Callback methode called by a Cursor2D this event is registered to 
         * when a cursor event occures. REASON_BUTTON_DOWN, REASON_MOVE and
         * REASON_BUTTON_UP control the mouse position tracking. Changes to
         * the y coordinate of the mouse position will change the distance 
         * angle of the associated camera. Othere calling reasons have no 
         * effect.
         *
         * @param caller The AbstractCursor object which calles this methode.
         * @param reason The reason why for this call.
         * @param param A reason depending parameter.
         */
        virtual void Trigger(AbstractCursor *caller, TriggerReason reason, unsigned int param);

        /**
         * Getter for the minimal distance to be set for the beholder.
         *
         * @return The minimal distance.
         */
        vislib::graphics::SceneSpaceType GetMinDist(void);

        /**
         * Answers the scaling factor from the mouse movement to the 
         * coordinates translation.
         *
         * @return The scaling factor.
         */
        inline float GetSpeedScaling(void) const {
            return this->scale;
        }

        /**
         * Setter for the minimal distance to be set for the beholder.
         *
         * @param dist The minimal distance.
         *
         * @throws IllegalParamException if dist is less or equal zero.
         */
        void SetMinDist(vislib::graphics::SceneSpaceType dist);

        /**
         * Sets a new scaling factor from the mouse movement to the coordinates
         * translation.
         *
         * @param speed The new scaling factor.
         */
        void SetSpeedScaling(float speed);

    private:

        /** Flag whether the mouse draggs. */
        bool drag;

        /** 
         * The square of the minimal distance from beholder position to look 
         * at point 
         */
        vislib::graphics::SceneSpaceType squareMinDist;

        /** The scaling speed */
        float scale;

    };
    
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_BEHOLDERDISTZOOM2D_H_INCLUDED */
