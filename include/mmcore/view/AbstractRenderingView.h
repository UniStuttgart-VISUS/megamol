/*
 * AbstractRenderingView.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_ABSTRACTRENDERINGVIEW_H_INCLUDED
#define MEGAMOLCORE_ABSTRACTRENDERINGVIEW_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "api/MegaMolCore.std.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/AbstractView.h"
#include "vislib/sys/CriticalSection.h"
#include "vislib/graphics/FpsCounter.h"


namespace megamol {
namespace core {
namespace view {


    /**
     * Abstract base class of rendering views
     */
    class MEGAMOLCORE_API AbstractRenderingView : public AbstractView {
    public:

        /**
         * Interface definition
         */
        class MEGAMOLCORE_API AbstractTitleRenderer {
        public:

            /** Ctor */
            AbstractTitleRenderer(void);

            /** Dtor */
            virtual ~AbstractTitleRenderer(void);

            /**
             * Create the renderer and allocates all resources
             *
             * @return True on success
             */
            virtual bool Create(void) = 0;

            /**
             * Renders the title scene
             *
             * @param tileX The view tile x coordinate
             * @param tileY The view tile y coordinate
             * @param tileW The view tile width
             * @param tileH The view tile height
             * @param virtW The virtual view width
             * @param virtH The virtual view height
             * @param stereo Flag if stereo rendering is to be performed
             * @param leftEye Flag if the stereo rendering is done for the left eye view
             * @param instTime The instance time code
             * @param core The core
             */
            virtual void Render(float tileX, float tileY, float tileW, float tileH,
                float virtW, float virtH, bool stereo, bool leftEye, double instTime,
                class ::megamol::core::CoreInstance *core) = 0;

            /**
             * Releases the renderer and all of its resources
             */
            virtual void Release(void) = 0;

        };

        /** Ctor. */
        AbstractRenderingView(void);

        /** Dtor. */
        virtual ~AbstractRenderingView(void);

    protected:

        /**
         * Marks the beginning of a frame
         */
        void beginFrame(void);

        /**
         * Marks the end of a frame
         *
         * @param abort Flag if the frame is aborted
         */
        void endFrame(bool abort = false);

        /**
         * The number of seconds the last frame needed to be drawn
         *
         * @return The last frame rendering time
         */
        double lastFrameTime(void) const;

        /**
         * Answer the background colour for the view
         *
         * @return The background colour for the view
         */
        const float *bkgndColour(void) const;

        /**
         * Answer if the soft cursor should be shown
         *
         * @return 'true' if the soft cursor should be shown
         */
        bool showSoftCursor(void) const;

        /**
         * Renders the title scene
         *
         * @param tileX The view tile x coordinate
         * @param tileY The view tile y coordinate
         * @param tileW The view tile width
         * @param tileH The view tile height
         * @param virtW The virtual view width
         * @param virtH The virtual view height
         * @param stereo Flag if stereo rendering is to be performed
         * @param leftEye Flag if the stereo rendering is done for the left eye view
             * @param instTime The instance time code
         */
        void renderTitle(float tileX, float tileY, float tileW, float tileH,
            float virtW, float virtH, bool stereo, bool leftEye, double instTime) const;

        /**
         * Removes the title renderer
         */
        void removeTitleRenderer(void) const;

        /**
         * TODO: Document me or I will softly curse you
         */
        void toggleSoftCurse(void);

        /** Pointer to the override background colour */
        float *overrideBkgndCol;

        /** Pointer to the override viewport */
        int *overrideViewport;

    private:

        /**
         * Fallback class just clearing the screen
         */
        class EmptyTitleRenderer : public AbstractTitleRenderer {
        public:

            /** Ctor */
            EmptyTitleRenderer(void);

            /** Dtor */
            virtual ~EmptyTitleRenderer(void);

            /**
             * Create the renderer and allocates all resources
             *
             * @return True on success
             */
            virtual bool Create(void);

            /**
             * Renders the title scene
             *
             * @param tileX The view tile x coordinate
             * @param tileY The view tile y coordinate
             * @param tileW The view tile width
             * @param tileH The view tile height
             * @param virtW The virtual view width
             * @param virtH The virtual view height
             * @param stereo Flag if stereo rendering is to be performed
             * @param leftEye Flag if the stereo rendering is done for the left eye view
             * @param instTime The instance time code
             * @param core The core
             */
            virtual void Render(float tileX, float tileY, float tileW, float tileH,
                float virtW, float virtH, bool stereo, bool leftEye, double instTime,
                class ::megamol::core::CoreInstance *core);

            /**
             * Releases the renderer and all of its resources
             */
            virtual void Release(void);

        };

        /** The background colour for the view */
        mutable float bkgndCol[3];

        /** The background colour for the view */
        mutable param::ParamSlot bkgndColSlot;

        /** Bool flag to activate software cursor rendering */
        mutable bool softCursor;

        /** Bool flag to activate software cursor rendering */
        mutable param::ParamSlot softCursorSlot;

        /** The title renderer */
        mutable AbstractTitleRenderer* titleRenderer;

#ifdef _WIN32
#pragma warning (disable: 4251)
#endif /* _WIN32 */
        /** The frames per second counter */
        vislib::graphics::FpsCounter fpsCounter;

        /** Sync object for fps counter */
        vislib::sys::CriticalSection fpsLock;
#ifdef _WIN32
#pragma warning (default: 4251)
#endif /* _WIN32 */

        /** The ID of the thread currently */
        DWORD fpsThreadID;

        /** A timer managing the fps output */
        unsigned int fpsOutputTimer;

    };


} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_ABSTRACTRENDERINGVIEW_H_INCLUDED */
