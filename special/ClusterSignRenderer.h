/*
 * ClusterSignRenderer.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_CLUSTERSIGNRENDERER_H_INCLUDED
#define MEGAMOLCORE_CLUSTERSIGNRENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */


namespace megamol {
namespace core {
namespace special {


    /**
     * Utility class managing a display tile
     */
    class ClusterSignRenderer {
    public:

        /**
         * Renders a sign illustrating a broken/missconfigured state
         *
         * @param width The width of the viewport in pixel
         * @param height The height of the viewport in pixel
         * @param stereo Flag if this is a stereo projection
         * @param rightEye Flag if this is a right eye
         */
        static void RenderBroken(int width, int height, bool stereo = false, bool rightEye = false);

        /**
         * Renders a sign illustration a not-OK-state
         *
         * @param width The width of the viewport in pixel
         * @param height The height of the viewport in pixel
         * @param stereo Flag if this is a stereo projection
         * @param rightEye Flag if this is a right eye
         */
        static void RenderNo(int width, int height, bool stereo, bool rightEye);

        /**
         * Renders a sign illustration a OK-state
         *
         * @param width The width of the viewport in pixel
         * @param height The height of the viewport in pixel
         * @param stereo Flag if this is a stereo projection
         * @param rightEye Flag if this is a right eye
         */
        static void RenderYes(int width, int height, bool stereo, bool rightEye);

    private:

        /**
         * Sets the matrices up
         *
         * @param width The width of the viewport in pixel
         * @param height The height of the viewport in pixel
         * @param stereo Flag if this is a stereo projection
         * @param rightEye Flag if this is a right eye
         */
        static void setupMatrices(int width, int height, bool stereo, bool rightEye);

        /**
         * Sets up the scene
         */
        static void setupScene(void);

        /**
         * Cleans up the scene
         */
        static void cleanupScene(void);

        /**
         * Renders the border
         *
         * @param c1 The border color as 32bit 0xAABBGGRR (alpha is ignored!)
         * @param c2 The gap color as 32bit 0xAABBGGRR (alpha is ignored!)
         * @param c3 The background color as 32bit 0xAABBGGRR (alpha is ignored!)
         */
        static void renderBorder(unsigned int c1, unsigned int c2,
            unsigned int c3);

        /**
         * Renders the cross
         *
         * @param col The color as 32bit 0xAABBGGRR (alpha is ignored!)
         */
        static void renderCross(unsigned int col);

        /**
         * Renders the check
         *
         * @param col The color as 32bit 0xAABBGGRR (alpha is ignored!)
         */
        static void renderCheck(unsigned int col);

        /**
         * Answers the animation tick
         *
         * @return The animation tick
         */
        static inline bool tick(void);

        /**
         * Ctor.
         */
        ClusterSignRenderer(void);

        /**
         * Dtor.
         */
        ~ClusterSignRenderer(void);

    };


} /* end namespace special */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_CLUSTERSIGNRENDERER_H_INCLUDED */
