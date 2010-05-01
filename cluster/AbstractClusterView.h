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
#include "vislib/AbstractFont.h"
#include "vislib/String.h"


namespace megamol {
namespace core {
namespace cluster {


    /**
     * Abstract base class of override rendering views
     */
    class AbstractClusterView : public view::AbstractTileView,
        public ClusterControllerClient {
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
         * Renders a fallback view holding information about the cluster
         */
        void renderFallbackView(void);

    private:

        /** possible icon states */
        enum IconState {
            ICONSTATE_ERROR,
            ICONSTATE_OK,
            ICONSTATE_WAIT,
            ICONSTATE_WORK
        };

        /** The number of vertices to be used to render the corners */
        const static int cornerPtCnt;

        /** The radius for the big corners */
        const static float cornerBigRad;

        /** The radius for the medium corners */
        const static float cornerMidRad;

        /** The radius for the small corners */
        const static float cornerSmlRad;

        /** The size of the borders */
        const static float borderSize;

        /**
         * Answers the font to be used for the info text messages
         *
         * @return The font to be used for the info text messages
         */
        static const vislib::graphics::AbstractFont& infoFont(void);

        /**
         * Creates the caption for the info icon border
         *
         * @return The caption for the info icon border
         */
        static vislib::TString infoIconBorderCaption(void);

        /**
         * Renders the error info icon with the specified colour
         *
         * @param colR The red colour component
         * @param colG The green colour component
         * @param colB The blue colour component
         * @param message The message to be displayed along with the icon
         */
        static void renderErrorInfoIcon(unsigned char colR,
            unsigned char colG, unsigned char colB,
            const vislib::TString& message = vislib::TString::EMPTY);

        /**
         * Renders the info icon border with the specified colour
         *
         * @param colR The red colour component
         * @param colG The green colour component
         * @param colB The blue colour component
         */
        static void renderInfoIconBorder(unsigned char colR,
            unsigned char colG, unsigned char colB);

        /**
         * Renders the OK info icon with the specified colour
         *
         * @param colR The red colour component
         * @param colG The green colour component
         * @param colB The blue colour component
         * @param message The message to be displayed along with the icon
         */
        static void renderOKInfoIcon(unsigned char colR,
            unsigned char colG, unsigned char colB,
            const vislib::TString& message = vislib::TString::EMPTY);

        /**
         * Renders the wait info icon with the specified colour
         *
         * @param colR The red colour component
         * @param colG The green colour component
         * @param colB The blue colour component
         * @param message The message to be displayed along with the icon
         */
        static void renderWaitInfoIcon(unsigned char colR,
            unsigned char colG, unsigned char colB,
            const vislib::TString& message = vislib::TString::EMPTY);

        /**
         * Renders the working info icon with the specified colour
         *
         * @param colR The red colour component
         * @param colG The green colour component
         * @param colB The blue colour component
         * @param message The message to be displayed along with the icon
         */
        static void renderWorkingInfoIcon(unsigned char colR,
            unsigned char colG, unsigned char colB,
            const vislib::TString& message = vislib::TString::EMPTY);

        /**
         * Sets up all important OpenGL rendering states
         */
        static void setupRendering(void);

    };


} /* end namespace cluster */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_ABSTRACTCLUSTERVIEW_H_INCLUDED */
