/*
 * AnimDataTimer.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef megamol_ANIMDATATIMER_H_INCLUDED
#define megamol_ANIMDATATIMER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "view/AnimDataModule.h"
#include "param/ParamSlot.h"
#include "vislib/PerformanceCounter.h"

namespace megamol {
namespace core {
namespace protein {

    /**
     * Utility class used to animate data from "AnimData" classes.
     */
	class AnimDataTimer { 
    public:

        /** Ctor. */
        AnimDataTimer(void);

        /** Dtor. */
        ~AnimDataTimer(void);

        /**
         * Answer the parameter for the flag whether or not the animation is 
         * running.
         *
         * @return The parameter for the flag whether or not the animation is 
         *         running.
         */
        inline param::ParamSlot& AnimationActiveParameter(void) {
            return this->m_animationParam;
        }

        /**
         * Answer the parameter for the speed factor of the animation in 
         * frames per second.
         *
         * @return The parameter for the speed factor of the animation in 
         *         frames per second.
         */
        inline param::ParamSlot& AnimationSpeedParameter(void) {
            return this->m_animSpeedParam;
        }

        /**
         * Answer whether or not this timer is running.
         *
         * @return whether or not this timer is running.
         */
        inline bool IsRunning(void) const {
            return this->m_runAnim;
        }

        /**
         * Sets the pointer to the 'AnimDataSource' object which will be 
         * animated using this timer. This method should not be used while the
         * timer is running.
         *
         * @param source The 'AnimDataSource' object which will be animated
         *               using this timer.
         */
        void SetSource(view::AnimDataModule *source) {
            this->m_source = source;
        }

        /** Starts the animation from the first frame. */
        void Start(void);

        /** Stops the animation at the current frame. */
        void Stop(void);

        /**
         * Answer the current time in frames of the animation timer.
         *
         * @return The current time in frames of the animation timer.
         */
        float Time(void) const;

    private:

        /** The data source which will be animated with this timer. */
        view::AnimDataModule * m_source;

        /** flag whether or not the animation is running */
        mutable bool m_runAnim;

        /** the performance counter used for the animation */
        mutable vislib::sys::PerformanceCounter m_animTimer;

        /** 
         * the number of the frame to show including the interpolation value 
         */
        mutable float m_animTime;

        /** parameter for the flag whether or not the animation is running */
		mutable param::ParamSlot m_animationParam;

        /** the speed factor of the animation in frames per second */
        param::ParamSlot m_animSpeedParam;
    };

} /* end namespace protein */
} /* end namespace core */
} /* end namespace megamol */

#endif /* megamol_ANIMDATATIMER_H_INCLUDED */
