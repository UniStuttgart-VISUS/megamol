/*
 * RenderSlave.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_RENDERSLAVE_H_INCLUDED
#define MEGAMOLCORE_RENDERSLAVE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#if (!defined(_WIN32)) || defined(_WIN64)
#define USE_INTERLOCKED_WORKAROUND 1
#endif /* !defined(_WIN32) || defined(_WIN64) */

#include "view/AbstractView.h"
#include "Module.h"
#include "CallerSlot.h"
#include "ViewDescription.h"
#include "param/ParamSlot.h"
#include "special/ClusterControllerClient.h"
#include "special/ClusterDisplayPlane.h"
#include "special/ClusterDisplayTile.h"
#include "special/RenderNetMsg.h"
#ifdef USE_INTERLOCKED_WORKAROUND
#include "vislib/CriticalSection.h"
#endif /* !_WIN32 */
#include "vislib/Socket.h"
#include "vislib/Thread.h"


namespace megamol {
namespace core {
namespace special {


    /**
     * Base class for render slaves (cluster display and visplay client)
     */
    class RenderSlave : public view::AbstractView, public Module,
        public ClusterControllerClient, public AbstractSlot::Listener {
    public:

        /**
         * Dtor.
         */
        virtual ~RenderSlave(void);

        /**
         * Renders this AbstractView3D in the currently active OpenGL context.
         */
        virtual void Render(void);

        /**
         * Resizes the AbstractView3D.
         *
         * @param width The new width.
         * @param height The new height.
         */
        virtual void Resize(unsigned int width, unsigned int height);

    protected:

        /**
         * Callback requesting a rendering of this view
         *
         * @param call The calling call
         *
         * @return The return value
         */
        virtual bool onRenderView(Call& call);

        /**
         * Implementation of 'Module::Create'.
         *
         * @return 'true' on success, 'false' otherwise.
         */
        virtual bool create(void);

        /**
         * Implementation of 'Module::Release'.
         */
        virtual void release(void);

        /**
         * This method is called when an object connects to the slot.
         *
         * @param slot The slot that triggered this event.
         */
        virtual void OnConnect(AbstractSlot& slot);

        /**
         * Ctor.
         */
        RenderSlave(void);

        /**
         * Sets the cluster display tile information to be used.
         *
         * @param plane The cluster display plane to be used
         * @param tile The cluster display tile to be used
         */
        void setClusterDisplayTile(const ClusterDisplayPlane &plane,
            const ClusterDisplayTile &tile);

        /**
         * Resets the cluster display tile information.
         */
        void resetClusterDisplayTile(void);

        /**
         * Answer the cluster display plane stored.
         *
         * @return The cluster display plane stored
         */
        inline const ClusterDisplayPlane &displayPlane(void) const {
            return this->plane;
        }

        /**
         * Answer the cluster display tile stored.
         *
         * @return The cluster display tile stored
         */
        inline const ClusterDisplayTile &displayTile(void) const {
            return this->tile;
        }

        /**
         * Answer the width of the actual viewport in pixels.
         *
         * @return The width of the actual viewport in pixels
         */
        inline unsigned int viewWidth(void) const {
            return this->viewportWidth;
        }

        /**
         * Answer the height of the actual viewport in pixels.
         *
         * @return The height of the actual viewport in pixels
         */
        inline unsigned int viewHeight(void) const {
            return this->viewportHeight;
        }

        /**
         * Handles an incoming network message
         *
         * @param msg The incoming message
         *
         * @return 'true' if the message was handled, 'false' if not.
         */
        virtual bool HandleMessage(RenderNetMsg &msg);

        /**
         * Freezes, updates, or unfreezes the view onto the scene (not the
         * rendering, but camera settings, timing, etc).
         *
         * @param freeze true means freeze or update freezed settings,
         *               false means unfreeze
         */
        virtual void UpdateFreeze(bool freeze);

    private:

        /**
         * The receiver thread runnable
         */
        class Receiver : public vislib::sys::Runnable {
        public:

            /**
             * Ctor.
             */
            Receiver(void);

            /**
             * Dtor.
             */
            virtual ~Receiver(void);

            /**
             * Sets the receiver up before starting.
             *
             * @param owner The owning object
             * @param socket The socket to use
             */
            void Setup(RenderSlave *owner, vislib::net::Socket *socket);

            /**
             * The loop receiving data from the socket.
             *
             * @param userData Not used
             *
             * @return 0 (Not used)
             */
            virtual DWORD Run(void *userData);

            /**
             * Signels the receiver loop to terminate. The socket must be
             * closed before calling this method.
             *
             * @return true.
             */
            virtual bool Terminate(void);

        private:

            /** The owning object */
            RenderSlave *owner;

            /** The socket to use */
            vislib::net::Socket *socket;

        };

        /** To deliver the received data */
        friend class Receiver;

        /**
         * Closes the connection.
         *
         * @param andThread Also terminates the receiver thread
         */
        void closeConnection(bool andThread = true);

        /**
         * Sets up the module graph based on the given description
         *
         * @param data The description of the module graph to set up
         * @param size The size of 'data' in bytes
         */
        void setupModuleGraph(const void* data, SIZE_T size);

#ifdef USE_INTERLOCKED_WORKAROUND

        /** Critical section as workaround for missing 64 bit interlocked support */
        vislib::sys::CriticalSection critSecInterlocked;

#endif /* USE_INTERLOCKED_WORKAROUND */

        /** The width of the actual viewport in pixels */
        unsigned int viewportWidth;

        /** The height of the actual viewport in pixels */
        unsigned int viewportHeight;

        /** The plane of the cluster display */
        ClusterDisplayPlane plane;

        /** The tile of the cluster display */
        ClusterDisplayTile tile;

        /** caller slot connected to the cluster controller */
        CallerSlot controllerSlot;

        /** caller slot connected to the view to be rendered */
        CallerSlot renderViewSlot;

        /** The slot for the server network address */
        param::ParamSlot serverAddrSlot;

        /** The slot for the server network port */
        param::ParamSlot serverPortSlot;

        /** Flag slot controlling the state of the server connection */
        param::ParamSlot serverConnectedSlot;

        /** Flag slot if the view should close when server disconnects */
        param::ParamSlot closeOnDisconnectSlot;

        /** The socket connecting to the master */
        vislib::net::Socket socket;

        /** The thread to receive data from the master */
        vislib::sys::Thread receiveThread;

        /** The view description to be instantiated */
        ViewDescription *viewDesc;

    };


} /* end namespace special */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_RENDERSLAVE_H_INCLUDED */
