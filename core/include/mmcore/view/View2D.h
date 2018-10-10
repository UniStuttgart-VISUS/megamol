/*
 * View2D.h
 *
 * Copyright (C) 2008 - 2010 by VISUS (Universitaet Stuttgart). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_VIEW2D_H_INCLUDED
#define MEGAMOLCORE_VIEW2D_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/AbstractRenderingView.h"
#include "mmcore/view/MouseFlags.h"
#include "mmcore/view/TimeControl.h"
#include "vislib/math/Rectangle.h"

namespace megamol {
namespace core {
namespace view {

    /*
     * Forward declaration of incoming render calls
     */
    class CallRenderView;


    /**
     * Base class of rendering graph calls
     */
    class View2D: public AbstractRenderingView {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "View2D";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "2D View Module";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
            return true;
        }

        /** Ctor. */
        View2D(void);

        /** Dtor. */
        virtual ~View2D(void);

        /**
         * Answer the default time for this view
         *
         * @param instTime the current instance time
         *
         * @return The default time
         */
        virtual float DefaultTime(double instTime) const {
            return this->timeCtrl.Time(instTime);
        }

        /**
         * Answer the camera synchronization number.
         *
         * @return The camera synchronization number
         */
        virtual unsigned int GetCameraSyncNumber(void) const;

        /**
         * Serialises the camera of the view
         *
         * @param serialiser Serialises the camera of the view
         */
        virtual void SerialiseCamera(vislib::Serialiser& serialiser) const;

        /**
         * Deserialises the camera of the view
         *
         * @param serialiser Deserialises the camera of the view
         */
        virtual void DeserialiseCamera(vislib::Serialiser& serialiser);

        /**
         * Renders this AbstractView3D in the currently active OpenGL context.
         *
         * @param context
         */
        virtual void Render(const mmcRenderViewContext& context);

        /**
         * Resets the view. This normally sets the camera parameters to
         * default values.
         */
        virtual void ResetView(void);

        /**
         * Resizes the AbstractView3D.
         *
         * @param width The new width.
         * @param height The new height.
         */
        virtual void Resize(unsigned int width, unsigned int height);

        /**
         * Callback requesting a rendering of this view
         *
         * @param call The calling call
         *
         * @return The return value
         */
        virtual bool OnRenderView(Call& call);

        /**
         * Freezes, updates, or unfreezes the view onto the scene (not the
         * rendering, but camera settings, timing, etc).
         *
         * @param freeze true means freeze or update freezed settings,
         *               false means unfreeze
         */
        virtual void UpdateFreeze(bool freeze);

	    virtual bool OnKey(Key key, KeyAction action, Modifiers mods) override;

        virtual bool OnChar(unsigned int codePoint) override;

        virtual bool OnMouseButton(MouseButton button, MouseButtonAction action, Modifiers mods) override;

        virtual bool OnMouseMove(double x, double y) override;

        virtual bool OnMouseScroll(double dx, double dy) override;

    protected:

        /**
         * Unpacks the mouse coordinates, which are relative to the virtual
         * viewport size.
         *
         * @param x The x coordinate of the mouse position
         * @param y The y coordinate of the mouse position
         */
        virtual void unpackMouseCoordinates(float &x, float &y);

    private:

		enum MouseMode : uint8_t { Propagate, Pan, Zoom };

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
         * Resets the view
         *
         * @param p Must be resetViewSlot
         *
         * @return true
         */
        bool onResetView(param::ParamSlot& p);

        /**
         * Flag if this is the first time an image gets created. Used for 
         * initial camera reset
         */
        bool firstImg;

        /** The viewport height */
        float height;

        /** The mouse drag mode */
        MouseMode mouseMode;

        /** The mouse x coordinate */
        float mouseX;

        /** The mouse y coordinate */
        float mouseY;

        /** Slot to call the renderer to render */
        CallerSlot rendererSlot;

        /** Triggers the reset of the view */
        param::ParamSlot resetViewSlot;

        /** whether to reset the view when the object bounding box changes */
        param::ParamSlot resetViewOnBBoxChangeSlot;

        /** Shows/hides the bounding box */
        param::ParamSlot showBBoxSlot;

        /** The colour of the bounding box */
        float bboxCol[4];

		/** Parameter slot for the bounding box colour */
        param::ParamSlot bboxColSlot;

        /** The view focus x coordinate */
        float viewX;

        /** The view focus y coordinate */
        float viewY;

        /** The view zoom factor */
        float viewZoom;

        /** the update counter for the view settings */
        unsigned int viewUpdateCnt;

        /** the viewport width */
        float width;

        /** the incoming rendering call */
        class CallRenderView *incomingCall;

        /**
         * 6 floats holding the override information for the viewing tile:
         *   tileX, tileY, tileW, tileH, fullW, fullH
         */
        float *overrideViewTile;

        /** The time control */
        TimeControl timeCtrl;

        /** cached bounding box */
        vislib::math::Rectangle<float> bbox;
    };


} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_VIEW2D_H_INCLUDED */
