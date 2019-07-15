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

#include "mmcore/api/MegaMolCore.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "mmcore/view/AbstractCallRender.h"
#include "mmcore/view/RenderOutputOpenGL.h" 
#include "mmcore/view/Input.h"
#include "vislib/graphics/CameraParameters.h"
#include "vislib/graphics/graphicstypes.h"


namespace megamol {
namespace core {
namespace view {

#ifdef _WIN32
#pragma warning(disable: 4250)  // I know what I am doing ...
#endif /* _WIN32 */
    /**
     * Call for rendering visual elements (from separate sources) into a single target, i.e.,
	 * FBO-based compositing and cluster display.
     */
    class MEGAMOLCORE_API CallRenderView : public AbstractCallRender, public RenderOutputOpenGL {
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
            return "Call for rendering visual elements into a single target";
        }

		/** Function index of 'render' */
        static const unsigned int CALL_RENDER = AbstractCallRender::FnRender;

        /** Function index of 'freeze' */
        static const unsigned int CALL_FREEZE = 7;

        /** Function index of 'unfreeze' */
        static const unsigned int CALL_UNFREEZE = 8;

        /** Function index of 'ResetView' */
        static const unsigned int CALL_RESETVIEW = 9;

        /**
         * Answer the number of functions used for this call.
         *
         * @return The number of functions used for this call.
         */
        static unsigned int FunctionCount(void) {
			ASSERT(CALL_FREEZE == AbstractCallRender::FunctionCount()
				&& "Enum has bad magic number");
			ASSERT(CALL_UNFREEZE == AbstractCallRender::FunctionCount() + 1
				&& "Enum has bad magic number");
			ASSERT(CALL_RESETVIEW  == AbstractCallRender::FunctionCount() + 2
				&& "Enum has bad magic number");
            return AbstractCallRender::FunctionCount() + 3;
        }

        /**
         * Answer the name of the function used for this call.
         *
         * @param idx The index of the function to return it's name.
         *
         * @return The name of the requested function.
         */
		static const char* FunctionName(unsigned int idx) {
            if (idx == CALL_FREEZE) {
                return "freeze";
            } else if (idx == CALL_UNFREEZE) {
                return "unfreeze";
            } else if (idx == CALL_RESETVIEW) {
                return "ResetView";
            } 
            return AbstractCallRender::FunctionName(idx);
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
         * Gets the input modifier
         *
         * @return The input modifier
         */
        inline Modifier InputModifier(void) const {
            return this->mod;
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
            return true;
        }

        /**
         * Gets the button.
         *
         * @return The button
         */
        inline unsigned int MouseButton(void) const {
            return this->btn;
        }

        /**
         * Gets the 'down' flag.
         *
         * @return The 'down' flag
         */
        inline bool MouseButtonDown(void) const{
            return this->down;
        }

        /**
         * Gets the x coordinate.
         *
         * @return The x coordinate
         */
        inline float MouseX(void) const {
            return this->x;
        }

        /**
         * Gets the y coordinate.
         *
         * @return The y coordinate
         */
        inline float MouseY(void) const {
            return this->y;
        }

        /**
         * Propagates the parameters controlled by the frontend via 
         * mmcRenderViewContext to the call.
         *
         * @param context The context to get the data from.
         */
        inline void PropagateContext(const mmcRenderViewContext& context) {
            this->SetGpuAffinity(context.GpuAffinity);
            this->SetInstanceTime(context.InstanceTime);
            this->SetTime(static_cast<float>(context.Time));
        }

        /**
         * Resets all flags
         */
        inline void ResetAll(void) {
            this->flagBkgnd = false;
            this->flagProj = false;
            this->flagTile = false;
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
         * Sets the input modifier info
         *
         * @param mod The input modifier
         * qparam down The down flag
         */
        [[deprecated("This is utterly bad design and to be replaced by something AbstractInputScope-y")]]
        inline void SetInputModifier(Modifier mod, bool down) {
            this->mod = mod;
            this->down = down;
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
         * Sets the mouse button info
         *
         * @param btn The mouse button
         * @param down The down flag
         */
		[[deprecated("This is utterly bad design and to be replaced by something AbstractInputScope-y")]]
        inline void SetMouseButton(unsigned int btn, bool down) {
            this->btn = btn;
            this->down = down;
        }

        /**
         * Sets the mouse position
         *
         * @param x The x coordinate of the mouse position
         * @param y The y coordinate of the mouse position
         */
		[[deprecated("This is utterly bad design and to be replaced by something AbstractInputScope-y")]]
        inline void SetMousePosition(float x, float y) {
            this->x = x;
            this->y = y;
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
            return this->GetViewport().Height();
        }

        /**
         * Gets the width of the viewport in pixel.
         *
         * @return The width of the viewport in pixel
         */
        inline unsigned int ViewportWidth(void) const {
            return this->GetViewport().Width();
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

        /** The width of the virtual viewport */
        float width;

        /** The button */
        unsigned int btn;

        /**
         * Flag whether the button is pressed, or not, or the new input
         * modifier state
         */
        bool down;

        /** The x coordinate */
        float x;

        /** The y coordinate */
        float y;

        /** The input modifier to be set */
        Modifier mod;

    };
#ifdef _WIN32
#pragma warning(default: 4250)
#endif /* _WIN32 */


    /** Description class typedef */
    typedef factories::CallAutoDescription<CallRenderView>
        CallRenderViewDescription;


} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_CALLRENDERVIEW_H_INCLUDED */
