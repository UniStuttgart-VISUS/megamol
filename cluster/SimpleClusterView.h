/*
 * SimpleClusterView.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_SIMPLECLUSTERVIEW_H_INCLUDED
#define MEGAMOLCORE_SIMPLECLUSTERVIEW_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "view/AbstractTileView.h"
#include "CallerSlot.h"
#include "param/ParamSlot.h"
#include "vislib/AbstractSimpleMessage.h"
#include "vislib/Serialiser.h"
#include "vislib/String.h"


namespace megamol {
namespace core {
namespace cluster {


    /**
     * Abstract base class of override rendering views
     */
    class SimpleClusterView : public view::AbstractTileView {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "SimpleClusterView";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Simple Powerwall-Fusion View Module";
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
        SimpleClusterView(void);

        /** Dtor. */
        virtual ~SimpleClusterView(void);

        /**
         * Renders this AbstractView3D in the currently active OpenGL context.
         */
        virtual void Render(void);

        /**
         * Unregisters from the specified client
         *
         * @param client The client to unregister from
         */
        void Unregister(class SimpleClusterClient *client);

        /**
         * Disconnect the view call
         */
        void DisconnectViewCall(void);

        /**
         * Set the module graph setup message
         *
         * @return msg The message
         */
        void SetSetupMessage(const vislib::net::AbstractSimpleMessage& msg);

        /**
         * Sets a initialization message for the camera parameters
         */
        void SetCamIniMessage(void);

        /**
         * Connects this view to another view
         *
         * @param toName The slot to connect to
         */
        void ConnectView(const vislib::StringA toName);

        /**
         * Answer the connected view
         *
         * @return The connected view or NULL if no view is connected
         */
        inline view::AbstractView *GetConnectedView(void) const {
            return this->getConnectedView();
        }

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
         * Renders a fallback view holding information about the cluster
         */
        void renderFallbackView(void);

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
         * Loads the configuration
         *
         * @param name The name to load the configuration for
         *
         * @return True on success
         */
        bool loadConfiguration(const vislib::StringA& name);

        /** Flag to identify the first frame */
        bool firstFrame;

        /** Flag if everything is frozen */
        bool frozen;

        /** The frozen time */
        double frozenTime;

        /** frozen camera parameters */
        vislib::Serialiser *frozenCam;

        /** The slot registering this view */
        CallerSlot registerSlot;

        /** The client end */
        class SimpleClusterClient *client;

        /** The initialization message */
        vislib::net::AbstractSimpleMessage *initMsg;

        param::ParamSlot heartBeatPortSlot;
        param::ParamSlot heartBeatServerSlot;

    };


} /* end namespace cluster */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_SIMPLECLUSTERVIEW_H_INCLUDED */
