/*
 * NVSwapGroupController.h
 *
 * Copyright (C) 2010 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLVIEWER_NVSWAPGROUPCONTROLLER_H_INCLUDED
#define MEGAMOLVIEWER_NVSWAPGROUPCONTROLLER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */


namespace megamol {
namespace viewer {


    /**
     * Class to hold and apply the NVSwapGroup settings
     */
    class NVSwapGroupController {
    public:

        /**
         * Returns the only singelton instance
         *
         * @return The only instance
         */
        static NVSwapGroupController& Instance(void);

        /**
         * Sets the settings
         *
         * @param group The group to join
         * @param barrier The barrier to join
         */
        inline void Set(unsigned int group, unsigned int barrier) {
            this->group = group;
            this->barrier = barrier;
        }

        /**
         * Sets the settings
         *
         * @param group The group to join
         */
        inline void SetGroup(unsigned int group) {
            this->group = group;
        }

        /**
         * Sets the settings
         *
         * @param barrier The barrier to join
         */
        inline void SetBarrier(unsigned int barrier) {
            this->barrier = barrier;
        }

        /**
         * Joins the current glut window into the swap group
         */
        void JoinGlutWindow(void);

    private:

        /** Ctor */
        NVSwapGroupController(void);

        /** Dtor */
        ~NVSwapGroupController(void);

        /** initializes the extensions, if required */
        void assertExtensions(void);

        /** The group to join */
        unsigned int group;

        /** The barrier to join */
        unsigned int barrier;

        /** Flag whether or not the extensions have been initialized */
        bool extensionInitialized;

    };


} /* end namespace viewer */
} /* end namespace megamol */

#endif /* MEGAMOLVIEWER_NVSWAPGROUPCONTROLLER_H_INCLUDED */
