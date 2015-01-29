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
#include "mmcore/cluster/ClusterControllerClient.h"
#include "mmcore/cluster/CommChannelServer.h"
#include "Module.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/param/ParamUpdateListener.h"
#include "vislib/net/AbstractCommEndPoint.h"
#include "vislib/net/CommServer.h"
#include "vislib/net/CommServerListener.h"
#include "vislib/sys/RunnableThread.h"
#include "vislib/SmartRef.h"
#include "vislib/sys/Thread.h"


namespace megamol {
namespace core {
namespace cluster {


    /**
     * Abstract base class of override rendering views
     */
    class ClusterViewMaster : public Module, protected ClusterControllerClient::Listener,
        protected CommChannelServer::Listener, protected param::ParamUpdateListener {
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

        /**
         * Disallow usage in quickstarts
         *
         * @return false
         */
        static bool SupportQuickstart(void) {
            return false;
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

        /**
         * A message has been received.
         *
         * @param sender The sending object
         * @param hPeer The peer which sent the message
         * @param msgType The type value of the message
         * @param msgBody The data of the message
         */
        virtual void OnClusterUserMessage(ClusterControllerClient& sender, const ClusterController::PeerHandle& hPeer, bool isClusterMember, const UINT32 msgType, const BYTE *msgBody);

        /**
         * A message has been received over the control channel.
         *
         * @param sender The sending object
         * @param channel The control channel
         * @param msg The received message
         */
        virtual void OnCommChannelMessage(CommChannelServer& server, CommChannel& channel, const vislib::net::AbstractSimpleMessage& msg);

        /**
         * Callback called when a parameter is updated
         *
         * @param slot The parameter updated
         */
        virtual void ParamUpdated(param::ParamSlot& slot);

    private:

        /**
         * The thread function for camera updates
         *
         * @param userData
         *
         * @return 0
         */
        static DWORD cameraUpdateThread(void *userData);

        /**
         * Answer the default server host of this machine, either IP-Address or computer name
         *
         * @return The default server host
         */
        vislib::TString defaultServerHost(void) const;

        /**
         * Answer the default server port of this machine
         *
         * @return The default server port
         */
        unsigned short defaultServerPort(void) const;

        /**
         * Answer the default server address of this machine
         *
         * @return The default server address
         */
        vislib::TString defaultServerAddress(void) const;

        /**
         * Answer the default server address of this machine
         *
         * @return The default server address
         */
        vislib::TString defaultVSyncServerAddress(void) const;

        /**
         * Callback when the server address is changed
         *
         * @param slot Must be serverAddressSlot
         *
         * @return True
         */
        bool onServerAddressChanged(param::ParamSlot& slot);

        /**
         * Sends a sanity check time request to all connected client nodes
         *
         * @param slot Must be sanityCheckTimeSlot
         *
         * @return True;
         */
        bool onDoSanityCheckTime(param::ParamSlot& slot);

        /**
         * Enters remote view pause mode
         *
         * @param slot Must be pauseRemoteViewSlot
         *
         * @return True;
         */
        bool onPauseRemoteViewClicked(param::ParamSlot& slot);

        /**
         * Resumes from remote view pause mode
         *
         * @param slot Must be resumeRemoteViewSlot
         *
         * @return True;
         */
        bool onResumeRemoteViewClicked(param::ParamSlot& slot);

        /**
         * Forces network v-sync on
         *
         * @param slot Must be forceNetVSyncOnSlot
         *
         * @return True;
         */
        bool onForceNetVSyncOnClicked(param::ParamSlot& slot);

        /**
         * Forces network v-sync off
         *
         * @param slot Must be forceNetVSyncOffSlot
         *
         * @return True;
         */
        bool onForceNetVSyncOffClicked(param::ParamSlot& slot);

        /** The cluster control client */
        ClusterControllerClient ccc;

        /** The control channel server */
        CommChannelServer ctrlServer;

        /** The name of the view to be used */
        param::ParamSlot viewNameSlot;

        /** The slot connecting to the view to be used */
        CallerSlot viewSlot;

        /** The TCP/IP address of the server including the port */
        param::ParamSlot serverAddressSlot;

        /** Endpoint for the server */
        vislib::net::IPEndPoint serverEndPoint;

        /** Performs a sanitycheck of the times on all cluster nodes */
        param::ParamSlot sanityCheckTimeSlot;

        /** The thread to update the camera settings */
        vislib::sys::Thread camUpdateThread;

        /** The slot to enter view pause */
        param::ParamSlot pauseRemoteViewSlot;

        /** The slot to resume from view pause */
        param::ParamSlot resumeRemoteViewSlot;

        /** The slot to force the network v-sync on */
        param::ParamSlot forceNetVSyncOnSlot;

        /** The slot to force the network v-sync off */
        param::ParamSlot forceNetVSyncOffSlot;

    };


} /* end namespace cluster */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_CLUSTERVIEWMASTER_H_INCLUDED */
