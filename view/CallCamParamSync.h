/*
 * CallCamParamSync.h
 *
 * Copyright (C) 2014 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_CALLCAMPARAMSYNC_H_INCLUDED
#define MEGAMOLCORE_CALLCAMPARAMSYNC_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "Call.h"
#include "CallAutoDescription.h"

#include "api/MegaMolCore.std.h"

#include "vislib/CameraParameters.h"
#include "vislib/StackTrace.h"


namespace megamol {
namespace core {
namespace view {

    /**
     * A call that transports camera parameters from one object to another.
     */
    class MEGAMOLCORE_API CallCamParamSync : public Call {

    public:

        /** The type of parameters being transported. */
        typedef vislib::SmartPtr<vislib::graphics::CameraParameters> CamParams;

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static inline const char *ClassName(void) {
            return "CallCamParamSync";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static inline const char *Description(void) {
            return "Transports camera parameters from one view to another.";
        }

        /**
         * Answer the number of functions used for this call.
         *
         * @return The number of functions used for this call.
         */
        static unsigned int FunctionCount(void);

        /**
         * Answer the name of the function used for this call.
         *
         * @param idx The index of the function to return it's name.
         *
         * @return The name of the requested function.
         */
        static const char *FunctionName(unsigned int idx);

        /** Index of the function retrieving the data. */
        static const unsigned int IDX_GET_CAM_PARAMS;

        /** Initialises a new instance. */
        CallCamParamSync(void);

        /** Finalises the instance. */
        virtual ~CallCamParamSync(void);

        inline CamParams PeekCamParams(void) {
            VLAUTOSTACKTRACE;
            return this->camParams;
        }

        inline const vislib::graphics::CameraParameters& GetCamParams(
                void) const {
            VLAUTOSTACKTRACE;
            return *this->camParams;
        }

        inline void SetCamParams(const CamParams camParams) {
            VLAUTOSTACKTRACE;
            this->camParams = camParams;
        }

    private:

        /** The intents that are provided by the call. */
        static const char *INTENTS[1];

#ifdef _MSC_VER
#pragma warning(disable: 4251)
#endif /* _MSC_VER */
        /** The payload of the call. */
        CamParams camParams;
#ifdef _MSC_VER
#pragma warning(default: 4251)
#endif /* _MSC_VER */
    };

    /** Description class typedef. */
    typedef CallAutoDescription<CallCamParamSync> CallCamParamSyncDescription;

} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_CALLCAMPARAMSYNC_H_INCLUDED */
