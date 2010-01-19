/*
 * CallRenderView.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_CALLRENDERVIEW_H_INCLUDED
#define MEGAMOLCORE_CALLRENDERVIEW_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "Call.h"
#include "CallAutoDescription.h"
#include "vislib/CameraParameters.h"
#include "vislib/deprecated.h"
#include "vislib/graphicstypes.h"


namespace megamol {
namespace core {
namespace view {

    /**
     * Call for registering a module at the cluster display
     */
    class CallRenderView : public Call {
    public:

        /**
         * Answer the name of the objects of this description.
         *
         * @return The name of the objects of this description.
         */
        static const char *ClassName(void) {
            return "CallRenderView";
        }

        /**
         * Gets a human readable description of the module.
         *
         * @return A human readable description of the module.
         */
        static const char *Description(void) {
            return "Call for registering a module at the cluster display";
        }

        /**
         * Answer the number of functions used for this call.
         *
         * @return The number of functions used for this call.
         */
        static unsigned int FunctionCount(void) {
            return 3;
        }

        /**
         * Answer the name of the function used for this call.
         *
         * @param idx The index of the function to return it's name.
         *
         * @return The name of the requested function.
         */
        static const char * FunctionName(unsigned int idx) {
            switch (idx) {
                case 0: return "render";
                case 1: return "freeze";
                case 2: return "unfreeze";
                default: return NULL;
            }
        }

        /**
         * Ctor.
         */
        CallRenderView(void);

        /**
         * Copy ctor.
         *
         * @param src Object to clone
         */
        CallRenderView(const CallRenderView& src);

        /**
         * ~Dtor.
         */
        virtual ~CallRenderView(void);

        /**
         * Answer the blue colour component of the background
         *
         * @return The blue colour component
         */
        inline unsigned char BackgroundBlue(void) const {
            return this->bkgndB;
        }

        /**
         * Answer the green colour component of the background
         *
         * @return The green colour component
         */
        inline unsigned char BackgroundGreen(void) const {
            return this->bkgndG;
        }

        /**
         * Answer the red colour component of the background
         *
         * @return The red colour component
         */
        inline unsigned char BackgroundRed(void) const {
            return this->bkgndR;
        }

        /**
         * Answer the stereo projection eye
         *
         * @return the stereo projection eye
         */
        inline vislib::graphics::CameraParameters::StereoEye GetEye(void) const {
            return this->eye;
        }

        /**
         * Answer the stereo projection type
         *
         * @return the stereo projection type
         */
        inline vislib::graphics::CameraParameters::ProjectionType GetProjectionType(void) const {
            return this->projType;
        }

        /**
         * Answers the flag indicating that the background information has been set
         *
         * @return 'true' if the background information has been set
         */
        inline bool IsBackgroundSet(void) const {
            return this->flagBkgnd;
        }

        /**
         * Answers the flag indicating that the projection information has been set
         *
         * @return 'true' if the projection information has been set
         */
        inline bool IsProjectionSet(void) const {
            return this->flagProj;
        }

        /**
         * Answers the flag indicating that the tile information has been set
         *
         * @return 'true' if the tile information has been set
         */
        inline bool IsTileSet(void) const {
            return this->flagTile;
        }

        /**
         * Answers the flag indicating that the viewport information has been set
         *
         * @return 'true' if the viewport information has been set
         */
        inline bool IsViewportSet(void) const {
            return this->flagVP;
        }

        /**
         * Resets all flags
         */
        inline void ResetAll(void) {
            this->flagBkgnd = false;
            this->flagProj = false;
            this->flagTile = false;
            this->flagVP = false;
        }

        /**
         * Resets the flag indicating that the background had been set.
         */
        inline void ResetBackground(void) {
            this->flagBkgnd = false;
        }

        /**
         * Resets the flag indicating that the projection had been set.
         */
        inline void ResetProjection(void) {
            this->flagProj = false;
        }

        /**
         * Resets the flag indicating that the tile had been set.
         */
        inline void ResetTile(void) {
            this->flagTile = false;
        }

        /**
         * Resets the flag indicating that the viewport had been set.
         */
        inline void ResetViewport(void) {
            this->flagVP = false;
        }

        /**
         * Sets the background colour information
         *
         * @param r The red colour component
         * @param g The green colour component
         * @param b The blue colour component
         */
        inline void SetBackground(unsigned char r, unsigned char g, unsigned char b) {
            this->flagBkgnd = true;
            this->bkgndR = r;
            this->bkgndG = g;
            this->bkgndB = b;
        }

        /**
         * Sets the projection information
         *
         * @param p The type of projection
         * @param e The eye used for stereo projections
         */
        inline void SetProjection(vislib::graphics::CameraParameters::ProjectionType p,
                vislib::graphics::CameraParameters::StereoEye e = vislib::graphics::CameraParameters::RIGHT_EYE) {
            this->flagProj = true;
            this->projType = p;
            this->eye = e;
        }

        /**
         * Sets the tile information
         *
         * @param fw The full width of the virtual viewport
         * @param fh The full height of the virtual viewport
         * @param x The x coordinate of the tile
         * @param y The y coordinate of the tile
         * @param w The width of the tile
         * @param h The height of the tile
         */
        inline void SetTile(float fw, float fh, float x, float y, float w, float h) {
            this->flagTile = true;
            this->width = fw;
            this->height = fh;
            this->tileX = x;
            this->tileY = y;
            this->tileW = w;
            this->tileH = h;
        }

        /**
         * Sets the size of the viewport in pixel.
         *
         * @param w The width of the viewport in pixel
         * @param h The height of the viewport in pixel
         */
        inline void SetViewportSize(unsigned int w, unsigned int h) {
            this->flagVP = true;
            this->vpHeight = h;
            this->vpWidth = w;
        }

        /**
         * Answer the height of the rendering tile
         *
         * @return The height of the rendering tile
         */
        inline float TileHeight(void) const {
            return this->tileH;
        }

        /**
         * Answer the width of the rendering tile
         *
         * @return The width of the rendering tile
         */
        inline float TileWidth(void) const {
            return this->tileW;
        }

        /**
         * Answer the x coordinate of the rendering tile
         *
         * @return The x coordinate of the rendering tile
         */
        inline float TileX(void) const {
            return this->tileX;
        }

        /**
         * Answer the y coordinate of the rendering tile
         *
         * @return The y coordinate of the rendering tile
         */
        inline float TileY(void) const {
            return this->tileY;
        }

        /**
         * Gets the height of the viewport in pixel.
         *
         * @return The height of the viewport in pixel
         */
        inline unsigned int ViewportHeight(void) const {
            return this->vpHeight;
        }

        /**
         * Gets the width of the viewport in pixel.
         *
         * @return The width of the viewport in pixel
         */
        inline unsigned int ViewportWidth(void) const {
            return this->vpWidth;
        }

        /**
         * Answer the height of the virtual viewport
         *
         * @return The height of the virtual viewport
         */
        inline float VirtualHeight(void) const {
            return this->height;
        }

        /**
         * Answer the width of the virtual viewport
         *
         * @return The width of the virtual viewport
         */
        inline float VirtualWidth(void) const {
            return this->width;
        }

        /**
         * Assignment operator
         *
         * @param rhs The right hand side operand
         *
         * @return A reference to 'this'
         */
        CallRenderView& operator=(const CallRenderView& rhs);

    private:

        /** The blue component of the background colour */
        unsigned char bkgndB;

        /** The green component of the background colour */
        unsigned char bkgndG;

        /** The red component of the background colour */
        unsigned char bkgndR;

        /** The stereo projection eye */
        vislib::graphics::CameraParameters::StereoEye eye;

        /** Flag indicating that the background colour information has been set */
        bool flagBkgnd : 1;

        /** Flag indicating that the projection information has been set */
        bool flagProj : 1;

        /** Flag indicating that the tile information has been set */
        bool flagTile : 1;

        /** Flag indicating that the viewport information has been set */
        bool flagVP : 1;

        /** The height of the virtual viewport */
        float height;

        /** The stereo projection type */
        vislib::graphics::CameraParameters::ProjectionType projType;

        /** The height of the rendering tile */
        float tileH;

        /** The width of the rendering tile */
        float tileW;

        /** The x coordinate of the rendering tile */
        float tileX;

        /** The y coordinate of the rendering tile */
        float tileY;

        /** The viewport height in pixel */
        unsigned int vpHeight;

        /** The viewport width in pixel */
        unsigned int vpWidth;

        /** The width of the virtual viewport */
        float width;

    };


    /** Description class typedef */
    typedef CallAutoDescription<CallRenderView>
        CallRenderViewDescription;


} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_CALLRENDERVIEW_H_INCLUDED */
