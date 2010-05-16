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

#include "view/AbstractTileView.h"
#include "cluster/ClusterControllerClient.h"
#include "cluster/ControlChannel.h"
#include "cluster/InfoIconRenderer.h"
#include "vislib/AbstractClientEndPoint.h"
#include "vislib/CriticalSection.h"
#include "vislib/SmartRef.h"
#include "vislib/String.h"
#include "vislib/Thread.h"


namespace megamol {
namespace core {
namespace cluster {


    /**
     * Abstract base class of override rendering views
     */
    class AbstractClusterView : public view::AbstractTileView,
        protected ClusterControllerClient::Listener, protected ControlChannel::Listener {
    public:

        /** Ctor. */
        AbstractClusterView(void);

        /** Dtor. */
        virtual ~AbstractClusterView(void);

        /**
         * Resets the view. This normally sets the camera parameters to
         * default values.
         */
        virtual void ResetView(void);

        /**
         * Sets the button state of a button of the 2d cursor. See
         * 'vislib::graphics::Cursor2D' for additional information.
         *
         * @param button The button.
         * @param down Flag whether the button is pressed, or not.
         */
        virtual void SetCursor2DButtonState(unsigned int btn, bool down);

        /**
         * Sets the position of the 2d cursor. See 'vislib::graphics::Cursor2D'
         * for additional information.
         *
         * @param x The x coordinate
         * @param y The y coordinate
         */
        virtual void SetCursor2DPosition(float x, float y);

        /**
         * Sets the state of an input modifier.
         *
         * @param mod The input modifier to be set.
         * @param down The new state of the input modifier.
         */
        virtual void SetInputModifier(mmcInputModifier mod, bool down);

    protected:

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
        void OnClusterUserMessage(ClusterControllerClient& sender, const ClusterController::PeerHandle& hPeer, bool isClusterMember, const UINT32 msgType, const BYTE *msgBody);

        /**
         * Informs that the control channel is now connected an can send and receive messages
         *
         * @param sender The sending object
         */
        virtual void OnControlChannelConnect(ControlChannel& sender);

        /**
         * Informs that the control channel is no longer connected.
         *
         * @param sender The sending object
         */
        virtual void OnControlChannelDisconnect(ControlChannel& sender);

        /** The cluster control client */
        ClusterControllerClient ccc;

        /** The control channel */
        ControlChannel ctrlChannel;

    private:

        /**
         * Callback when the server address is changed
         *
         * @param slot Must be serverAddressSlot
         *
         * @return True
         */
        bool onServerAddressChanged(param::ParamSlot& slot);

        /** The ping counter */
        unsigned int lastPingTime;

        /** The TCP/IP address of the server including the port */
        param::ParamSlot serverAddressSlot;

    };


} /* end namespace cluster */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_ABSTRACTCLUSTERVIEW_H_INCLUDED */
