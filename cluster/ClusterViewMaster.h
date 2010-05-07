/*
 * ClusterViewMaster.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_CLUSTERVIEWMASTER_H_INCLUDED
#define MEGAMOLCORE_CLUSTERVIEWMASTER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "CallerSlot.h"
#include "cluster/ClusterControllerClient.h"
#include "Module.h"
#include "param/ParamSlot.h"
#include "vislib/AbstractServerEndPoint.h"
#include "vislib/SmartRef.h"


namespace megamol {
namespace core {
namespace cluster {


    /**
     * Abstract base class of override rendering views
     */
    class ClusterViewMaster : public Module, public ClusterControllerClient {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "ClusterViewMaster";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Master view controller module for distributed, tiled rendering";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
            return true;
        }

        /** Ctor. */
        ClusterViewMaster(void);

        /** Dtor. */
        virtual ~ClusterViewMaster(void);

    protected:

        /**
         * Implementation of 'Create'.
         *
         * @return 'true' on success, 'false' otherwise.
         */
        virtual bool create(void);

        /**
         * Implementation of 'Release'.
         */
        virtual void release(void);

        /**
         * Reacts on changes of the view name parameter
         *
         * @param slot Must be 'viewNameSlot'
         *
         * @return 'true' to reset the dirty flag.
         */
        bool onViewNameChanged(param::ParamSlot& slot);

    private:

        /** The name of the view to be used */
        param::ParamSlot viewNameSlot;

        /** The slot connecting to the view to be used */
        CallerSlot viewSlot;

        /** The communication channel for control commands */
        vislib::SmartRef<vislib::net::AbstractServerEndPoint> commChnlCtrl;

        /** The communication channel for camera updates */
        vislib::SmartRef<vislib::net::AbstractServerEndPoint> commChnlCam;

    };


} /* end namespace cluster */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_CLUSTERVIEWMASTER_H_INCLUDED */
