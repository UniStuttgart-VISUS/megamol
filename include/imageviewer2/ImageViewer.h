/*
 * ImageViewer.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_IMAGEVIEWER_H_INCLUDED
#define MEGAMOLCORE_IMAGEVIEWER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/view/Renderer3DModule.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/graphics/gl/OpenGLTexture2D.h"
#include "vislib/Pair.h"
#include "vislib/math/Rectangle.h"
#include "vislib/SmartPtr.h"

/*
 * Copyright (C) 2010 by Sebastian Grottel.
 */
#include "vislib/graphics/AbstractBitmapCodec.h"
#include "vislib/RawStorage.h"

using namespace megamol::core;

namespace megamol {
namespace imageviewer2 {

    /**
     * Mesh-based renderer for bézier curve tubes
     */
    class ImageViewer : public view::Renderer3DModule {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "ImageViewer";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "A litte less simple Image Viewer";
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
        ImageViewer(void);

        /** Dtor. */
        virtual ~ImageViewer(void);

    protected:

        /**
         * Implementation of 'Create'.
         *
         * @return 'true' on success, 'false' otherwise.
         */
        virtual bool create(void);

        /**
         * The get capabilities callback. The module should set the members
         * of 'call' to tell the caller its capabilities.
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        virtual bool GetCapabilities(Call& call);

        /**
         * The get extents callback. The module should set the members of
         * 'call' to tell the caller the extents of its data (bounding boxes
         * and times).
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        virtual bool GetExtents(Call& call);

        /**
         * Implementation of 'Release'.
         */
        virtual void release(void);

        /**
         * The render callback.
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        virtual bool Render(Call& call);

    private:

        /**
         * Splits a line at the semicolon into a left and right part. If there
         * is no semicolon, defaultEye governs which one of the strings is set,
         * the other one is emptied.
         */
        void interpretLine(const vislib::TString source, vislib::TString& left,
                           vislib::TString& right);

        /**
         * Callback invoked when the user pastes a line containing 
         * <leftimg>[;<rightimg>]. The text is split using interpretLine
         * and assigned to (left|right)FilenameSlot.
         */
        bool onFilesPasted(param::ParamSlot &slot);

        /**
        * Callback invoked when the user pastes a text containing multiple
        * lines containing <leftimg>[;<rightimg>]. Splits text into single
        * lines and then behaves like onFilesPasted for each line.
        */
        bool onSlideshowPasted(param::ParamSlot &slot);

        /** Callback for going back to slide 0 */
        bool onFirstPressed(param::ParamSlot &slot);

        /** Callback for going back one slide */
        bool onPreviousPressed(param::ParamSlot &slot);

        /** Callback for going forward one slide */
        bool onNextPressed(param::ParamSlot &slot);

        /** Callback for going forward to the last slide */
        bool onLastPressed(param::ParamSlot &slot);

        /**
         * Callback that occurs on slide change. Copies file names from
         * leftFiles and rightFiles to leftFilenameSlot and 
         * rightFilenameSlot respectively based on currentSlot.
         */
        bool onCurrentSet(param::ParamSlot &slot);

        /** makes sure the image for the respective eye is loaded. */
        void assertImage(bool rightEye);

        /** The image file path slot */
        param::ParamSlot leftFilenameSlot;

        /** The image file path slot */
        param::ParamSlot rightFilenameSlot;

        /** Slot to receive both file names at once */
        param::ParamSlot pasteFilenamesSlot;

        /** Slot to receive a whole slideshow at once */
        param::ParamSlot pasteSlideshowSlot;

        /** slot for going back to slide 0 */
        param::ParamSlot firstSlot;

        /** slot for going back one slide */
        param::ParamSlot previousSlot;

        /** slide for setting the current slide index */
        param::ParamSlot currentSlot;

        /** slot for going forward one slide */
        param::ParamSlot nextSlot;

        /** slot for going forward to the last slide */
        param::ParamSlot lastSlot;

        param::ParamSlot defaultEye;

        /** The width of the image */
        unsigned int width;

        /** The height of the image */
        unsigned int height;

        /** The image tiles */
        vislib::Array<vislib::Pair<vislib::math::Rectangle<float>,
            vislib::SmartPtr<vislib::graphics::gl::OpenGLTexture2D> > > tiles;

        /** the slide show files for the left eye */
        vislib::Array<vislib::TString> leftFiles;

        /** the slide show files for the right eye */
        vislib::Array<vislib::TString> rightFiles;
    };

} /* end namespace imageviewer2 */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_IMAGEVIEWER_H_INCLUDED */
