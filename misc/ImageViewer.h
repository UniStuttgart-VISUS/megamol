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

#include "view/Renderer3DModule.h"
#include "param/ParamSlot.h"
#include "vislib/OpenGLTexture2D.h"
#include "vislib/Pair.h"
#include "vislib/Rectangle.h"
#include "vislib/SmartPtr.h"

/*
 * Copyright (C) 2010 by Sebastian Grottel.
 */
#include "vislib/AbstractBitmapCodec.h"
#include "vislib/RawStorage.h"
#include "misc/PngBitmapCodec.h"


namespace megamol {
namespace core {
namespace misc {

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
            return "Simple Image Viewer";
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
         * TODO: Document
         */
        bool onFilesPasted(param::ParamSlot &slot);

        void assertImage(bool rightEye);

        /** The image file path slot */
        param::ParamSlot leftFilenameSlot;

        /** The image file path slot */
        param::ParamSlot rightFilenameSlot;

        /** Slot to receive both file names at once */
        param::ParamSlot pasteFilenamesSlot;

        /** The width of the image */
        unsigned int width;

        /** The height of the image */
        unsigned int height;

        /** The image tiles */
        vislib::Array<vislib::Pair<vislib::math::Rectangle<float>, vislib::SmartPtr<vislib::graphics::gl::OpenGLTexture2D> > > tiles;

    };

} /* end namespace misc */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_IMAGEVIEWER_H_INCLUDED */
