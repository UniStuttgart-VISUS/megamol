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
namespace sg {
namespace graphics {

    /**
     * Bitmap codec for png images
     * Currently loading only
     */
    class PngBitmapCodec : public vislib::graphics::AbstractBitmapCodec {
    public:

        /** Ctor */
        PngBitmapCodec(void);

        /** Dtor */
        virtual ~PngBitmapCodec(void);

        /**
         * Autodetects if an image can be loaded by this codec by checking
         * preview data from the beginning of the image data.
         *
         * @param mem The preview data.
         * @param size The size of the preview data in bytes.
         *
         * @return 0 if the file cannot be loaded by this codec.
         *         -1 if the preview data was insufficient to determine the
         *            codec compatibility.
         *         1 if the file can be loaded by this codec (loading might
         *           still fail however, e.g. if file data is corrupt).
         */
        virtual int AutoDetect(const void *mem, SIZE_T size) const;

        /**
         * Answers whether this codec can autodetect if an image is supported
         * by checking preview data.
         *
         * @return 'true' if the codec can autodetect image compatibility.
         */
        virtual bool CanAutoDetect(void) const;

        /**
         * Answers whether this codec can load images from memory buffers.
         *
         * @return 'true' if this codec can load images from memory buffers.
         */
        virtual bool CanLoadFromMemory(void) const;

        /**
         * Answers whether this codec can save images to memory buffers.
         *
         * @return 'true' if this codec can save images to memory buffers.
         */
        virtual bool CanSaveToMemory(void) const;

        /**
         * Answer the file name extensions usually used for image files of
         * the type of this codec. Each file name extension includes the
         * leading period. Multiple file name extensions are separated by
         * semicolons.
         *
         * @return The file name extensions usually used for image files of
         *         the type of this codec.
         */
        virtual const char* FileNameExtsA(void) const;

        /**
         * Answer the file name extensions usually used for image files of
         * the type of this codec. Each file name extension includes the
         * leading period. Multiple file name extensions are separated by
         * semicolons.
         *
         * @return The file name extensions usually used for image files of
         *         the type of this codec.
         */
        virtual const wchar_t* FileNameExtsW(void) const;

        /**
         * Loads an image from a memory buffer.
         *
         * You must set 'Image' to a valid BitmapImage object before calling
         * this method.
         *
         * @param mem Pointer to the memory buffer holding the image data.
         * @param size The size of the memory buffer in bytes.
         *
         * @return 'true' if the file was successfully loaded.
         */
        virtual bool Load(const void *mem, SIZE_T size);

        /* keeping overloaded 'Load' methods */
        using AbstractBitmapCodec::Load;

        /**
         * Answer the human-readable name of the codec.
         *
         * @return The human-readable name of the codec.
         */
        virtual const char * NameA(void) const;

        /**
         * Answer the human-readable name of the codec.
         *
         * @return The human-readable name of the codec.
         */
        virtual const wchar_t * NameW(void) const;

        /**
         * Saves the image to a memory block.
         *
         * You must set 'Image' to a valid BitmapImage object before calling
         * this method.
         *
         * @param outmem The memory block to receive the image data. The image
         *               data will replace all data in the memory block.
         *
         * @return 'true' if the file was successfully saved.
         */
        virtual bool Save(vislib::RawStorage& outmem) const;

        /* keeping overloaded 'Save' methods */
        using AbstractBitmapCodec::Save;

    };


} /* end namespace graphics */
} /* end namespace sg */

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
