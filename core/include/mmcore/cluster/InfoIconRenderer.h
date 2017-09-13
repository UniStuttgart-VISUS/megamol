/*
 * InfoIconRenderer.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_INFOICONRENDERER_H_INCLUDED
#define MEGAMOLCORE_INFOICONRENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "vislib/graphics/AbstractFont.h"
#include "vislib/String.h"


namespace megamol {
namespace core {
namespace cluster {

    /**
     * Static utility class for rendering info icons for the cluster fallback
     * view
     */
    class InfoIconRenderer {
    public:

        /** possible icon states */
        enum IconState {
            ICONSTATE_UNKNOWN,
            ICONSTATE_ERROR,
            ICONSTATE_OK,
            ICONSTATE_WAIT,
            ICONSTATE_WORK
        };

        /**
         * Renders the error info icon with the specified colour
         *
         * @param colR The red colour component
         * @param colG The green colour component
         * @param colB The blue colour component
         * @param message The message to be displayed along with the icon
         */
        static void RenderErrorInfoIcon(unsigned char colR,
            unsigned char colG, unsigned char colB,
            const vislib::TString& message = vislib::TString::EMPTY);

        /**
         * Renders the info icon
         *
         * @param icon The icon to show
         * @param message The message to be displayed along with the icon
         */
        static void RenderInfoIcon(IconState icon,
            const vislib::TString& message = vislib::TString::EMPTY);

        /**
         * Renders the info icon
         *
         * @param icon The icon to show
         * @param colR The red colour component
         * @param colG The green colour component
         * @param colB The blue colour component
         * @param message The message to be displayed along with the icon
         */
        static void RenderInfoIcon(IconState icon, unsigned char colR,
            unsigned char colG, unsigned char colB,
            const vislib::TString& message = vislib::TString::EMPTY);

        /**
         * Renders the info icon border with the specified colour
         *
         * @param colR The red colour component
         * @param colG The green colour component
         * @param colB The blue colour component
         */
        static void RenderInfoIconBorder(unsigned char colR,
            unsigned char colG, unsigned char colB);

        /**
         * Renders the OK info icon with the specified colour
         *
         * @param colR The red colour component
         * @param colG The green colour component
         * @param colB The blue colour component
         * @param message The message to be displayed along with the icon
         */
        static void RenderOKInfoIcon(unsigned char colR,
            unsigned char colG, unsigned char colB,
            const vislib::TString& message = vislib::TString::EMPTY);

        /**
         * Renders the info icon for unknown state with the specified colour
         *
         * @param colR The red colour component
         * @param colG The green colour component
         * @param colB The blue colour component
         * @param message The message to be displayed along with the icon
         */
        static void RenderUnknownStateInfoIcon(unsigned char colR,
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
        static void RenderWaitInfoIcon(unsigned char colR,
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
        static void RenderWorkingInfoIcon(unsigned char colR,
            unsigned char colG, unsigned char colB,
            const vislib::TString& message = vislib::TString::EMPTY);

    private:

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
         * Creates the caption for the info icon border
         *
         * @return The caption for the info icon border
         */
        static vislib::TString infoIconBorderCaption(void);

        /**
         * Sets up all important OpenGL rendering states
         */
        static void setupRendering(void);

        /**
         * Private ctor to disallow instances
         */
        InfoIconRenderer(void);

    };

} /* end namespace cluster */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_INFOICONRENDERER_H_INCLUDED */
