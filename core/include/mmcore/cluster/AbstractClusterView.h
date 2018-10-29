/*
 * AbstractClusterView.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_ABSTRACTCLUSTERVIEW_H_INCLUDED
#define MEGAMOLCORE_ABSTRACTCLUSTERVIEW_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/view/AbstractTileView.h"
#include "mmcore/cluster/ClusterControllerClient.h"
#include "mmcore/cluster/CommChannel.h"
#include "mmcore/cluster/InfoIconRenderer.h"
#include "vislib/net/AbstractCommEndPoint.h"
#include "vislib/sys/CriticalSection.h"
#include "vislib/SmartRef.h"
#include "vislib/String.h"
#include "vislib/sys/Thread.h"


namespace megamol {
namespace core {
namespace cluster {


    /**
     * Abstract base class of override rendering views
     */
    class AbstractClusterView : public view::AbstractTileView,
        protected ClusterControllerClient::Listener, protected CommChannel::Listener {
    public:

        /** Possible setup states */
        enum SetupState {
            SETUP_UNKNOWN,
            SETUP_TIME,
            SETUP_GRAPH,
            SETUP_CAMERA,
            SETUP_COMPLETE
        };

        /** Ctor. */
        AbstractClusterView(void);

        /** Dtor. */
        virtual ~AbstractClusterView(void);

        /**
         * Resets the view. This normally sets the camera parameters to
         * default values.
         */
        virtual void ResetView(void);

    protected:

        /**
         * Initializes parameter values on 'create'
         */
        void initClusterViewParameters(void);

        /**
         * A ping function to be called at least once per second
         */
        void commPing(void);

        /**
         * Renders a fallback view holding information about the cluster
         */
        void renderFallbackView(void);

        /**
         * Gets the info message and icon for the fallback view
         *
         * @param outMsg The message to be shows in the fallback view
         * @param outState The state icon to be shows in the fallback view
         */
        virtual void getFallbackMessageInfo(vislib::TString& outMsg,
            InfoIconRenderer::IconState& outState);

        /**
         * A message has been received.
         *
         * @param sender The sending object
         * @param hPeer The peer which sent the message
         * @param msgType The type value of the message
         * @param msgBody The data of the message
         */
        void OnClusterUserMessage(ClusterControllerClient& sender, const ClusterController::PeerHandle& hPeer,
            bool isClusterMember, const UINT32 msgType, const BYTE *msgBody);

        /**
         * Informs that the control channel is now connected an can send and receive messages
         *
         * @param sender The sending object
         */
        virtual void OnCommChannelConnect(CommChannel& sender);

        /**
         * Informs that the control channel is no longer connected.
         *
         * @param sender The sending object
         */
        virtual void OnCommChannelDisconnect(CommChannel& sender);

        /**
         * A message has been received over the control channel.
         *
         * @param sender The sending object
         * @param msg The received message
         */
        virtual void OnCommChannelMessage(CommChannel& sender,
            const vislib::net::AbstractSimpleMessage& msg);

        /** The cluster control client */
        ClusterControllerClient ccc;

        /** The control channel */
        CommChannel ctrlChannel;

    private:

        /**
         * Utility class to initialize the camera
         */
        class InitCameraHookHandler : public view::AbstractView::Hooks {
        public:

            /**
             * Empty ctor.
             */
            InitCameraHookHandler(cluster::CommChannel *channel);

            /**
             * Empty but virtual dtor.
             */
            virtual ~InitCameraHookHandler(void);

            /**
             * Hook method to be called before the view is rendered.
             *
             * @param view The calling view
             */
            virtual void BeforeRender(AbstractView *view);

        private:

            /** The communication channel */
            cluster::CommChannel *channel;

            /** counter */
            unsigned int frameCnt;

        };

        /**
         * Callback when the server address is changed
         *
         * @param slot Must be serverAddressSlot
         *
         * @return True
         */
        bool onServerAddressChanged(param::ParamSlot& slot);

        /**
         * Continues the setup
         *
         * @param state The setup state to perform next
         */
        void continueSetup(SetupState state = SETUP_UNKNOWN);

        /** The ping counter */
        unsigned int lastPingTime;

        /** The TCP/IP address of the server including the port */
        param::ParamSlot serverAddressSlot;

        /** The current setup state */
        SetupState setupState;

        /** Data received from the network to setup the module graph */
        vislib::net::SimpleMessage *graphInitData;

    };


} /* end namespace cluster */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_ABSTRACTCLUSTERVIEW_H_INCLUDED */
