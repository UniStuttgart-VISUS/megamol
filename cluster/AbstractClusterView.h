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
#include "cluster/InfoIconRenderer.h"
#include "vislib/AbstractClientEndPoint.h"
#include "vislib/AbstractSimpleMessage.h"
#include "vislib/CriticalSection.h"
#include "vislib/SimpleMessageDispatcher.h"
#include "vislib/SimpleMessageDispatchListener.h"
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
        public ClusterControllerClient,
        public vislib::net::SimpleMessageDispatchListener {
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

        /**
         * Informs the client that the cluster is now available.
         */
        virtual void OnClusterAvailable(void);

        /**
         * Informs the client that the cluster is no longer available.
         */
        virtual void OnClusterUnavailable(void);

        /**
         * A message has been received.
         *
         * @param hPeer The peer which sent the message
         * @param msgType The type value of the message
         * @param msgBody The data of the message
         */
        virtual void OnUserMsg(const ClusterController::PeerHandle& hPeer,
            const UINT32 msgType, const BYTE *msgBody);

        /**
         * This method is called every time a message is received.
         *
         * This method should return very quickly and should not perform
         * excessive work as it is executed in the discovery thread.
         *
         * The return value of the method can be used to stop the message
         * dispatcher, e. g. if an exit message was received. 
         *
         * Note that the dispatcher will stop if any of the registered listeners
         * returns false.
         *
         * @param src The SimpleMessageDispatcher that received the message.
         * @param msg The message that was received.
         *
         * @return true in order to make the SimpleMessageDispatcher continue
         *         receiving messages, false will cause the dispatcher to
         *         exit.
         */
        virtual bool OnMessageReceived(
            vislib::net::SimpleMessageDispatcher& src,
            const vislib::net::AbstractSimpleMessage& msg) throw();

    protected:

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
         * Answer if this node is connected to the head node
         *
         * @return 'true' if this node is connected to the head node
         */
        bool isConnectedToHead(void) const;

    private:

        /** Possible setup states */
        enum SetupState {
            SETUPSTATE_ERROR,
            SETUPSTATE_PRECONNECT,
            SETUPSTATE_CONNECTED,
            SETUPSTATE_DISCONNECTED
        };

        /**
         * The code controlling the setup procedure
         *
         * @param userData Pointer to this AbstractClusterView
         *
         * @return The return code of the setup procedure
         */
        static DWORD setupProcedure(void *userData);

        /** The thread controlling the setup procedure */
        vislib::sys::Thread setupThread;

        /** The current setup state */
        SetupState setupState;

        /** The setup state variable lock */
        mutable vislib::sys::CriticalSection setupStateLock;

        /** The client end point connected to the head node view for control commands */
        vislib::SmartRef<vislib::net::AbstractClientEndPoint> commChnlCtrl;

        /** The client end point connected to the head node view for camera updates */
        vislib::SmartRef<vislib::net::AbstractClientEndPoint> commChnlCam;

        /** Message dispatcher for control commands */
        vislib::net::SimpleMessageDispatcher ctrlMsgDispatch;

        /** Message dispatcher for camera updates */
        vislib::net::SimpleMessageDispatcher camMsgDispatch;

    };


} /* end namespace cluster */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_ABSTRACTCLUSTERVIEW_H_INCLUDED */
