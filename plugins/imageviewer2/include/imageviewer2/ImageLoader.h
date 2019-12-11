/*
 * ImageLoader.h
 *
 * Copyright (C) 2019 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */
#ifndef MEGAMOLCORE_IMAGELOADER_H_INCLUDED
#define MEGAMOLCORE_IMAGELOADER_H_INCLUDED

#include "image_calls/Image2DCall_2.h"
#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

#include <fstream>

namespace megamol {
namespace imageviewer2 {

class ImageLoader : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "ImageLoader"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "A module that loads images from disk"; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

    /** Ctor. */
    ImageLoader(void);

    /** Dtor. */
    virtual ~ImageLoader(void);

protected:
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
     * The render callback.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool GetData(core::Call& call);

private:
    /** Callee slot requesting images from this module */
    core::CalleeSlot callRequestImage;

    /** Image file path slot */
    core::param::ParamSlot filenameSlot;

    /** Pointer to the vector containing the images */
    std::shared_ptr<image_calls::Image2DCall_2::ImageVector> imageData;

    /** hash value for the data */
    SIZE_T datahash;

    /**
     * Loads an image from the harddrive using the given path
     *
     * @param path The path of the image file.
     * @return True, if the image could be loaded, false otherwise.
     */
    bool loadImage(const std::filesystem::path& path);
};

} // namespace imageviewer2
} // namespace megamol

#endif // !MEGAMOLCORE_IMAGELOADER_H_INCLUDED
