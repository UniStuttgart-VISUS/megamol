/*
 * CallRender.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_CALLRENDER_H_INCLUDED
#define MEGAMOLCORE_CALLRENDER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "Call.h"
#include "CallAutoDescription.h"
#include "vislib/CameraParameters.h"
#include "vislib/SmartPtr.h"


namespace megamol {
namespace core {
namespace view {


    /**
     * Base class of rendering graph calls
     */
    class CallRender : public Call {
    public:

        /**
         * Answer the name of the objects of this description.
         *
         * @return The name of the objects of this description.
         */
        static const char *ClassName(void) {
            return "CallRender";
        }

        /**
         * Gets a human readable description of the module.
         *
         * @return A human readable description of the module.
         */
        static const char *Description(void) {
            return "Call for rendering a frame";
        }

        /**
         * Answer the number of functions used for this call.
         *
         * @return The number of functions used for this call.
         */
        static unsigned int FunctionCount(void) {
            return 1;
        }

        /**
         * Answer the name of the function used for this call.
         *
         * @param idx The index of the function to return it's name.
         *
         * @return The name of the requested function.
         */
        static const char * FunctionName(unsigned int idx) {
            switch (idx) {
                case 0: return "Render";
                case 1: return "GetBBox";
                default: return NULL;
            }
        }

        /** Ctor. */
        CallRender(void);

        /** Dtor. */
        virtual ~CallRender(void);

        /**
         * Gets the camera parameters pointer.
         *
         * @return The camera parameters pointer.
         */
        inline const vislib::SmartPtr<vislib::graphics::CameraParameters>&
        GetCameraParameters(void) const {
            return this->camParams;
        }

        /**
         * Sets the camera parameters pointer.
         *
         * @param camParams The new value for the camera parameters pointer.
         */
        inline void SetCameraParameters(const vislib::SmartPtr<
                vislib::graphics::CameraParameters>& camParams) {
            this->camParams = camParams;
        }

    private:

        /** The camera parameters */
        vislib::SmartPtr<vislib::graphics::CameraParameters> camParams;

    };


    /** Description class typedef */
    typedef CallAutoDescription<CallRender> CallRenderDescription;


} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_CALLRENDER_H_INCLUDED */
