/*
 * ImageLoader.h
 *
 * Copyright (C) 2019 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */
#ifndef MEGAMOLCORE_IMAGELOADER_H_INCLUDED
#define MEGAMOLCORE_IMAGELOADER_H_INCLUDED

#include "image_calls/Image2DCall.h"
#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

#include <fstream>
#include <mutex>
#include <queue>
#include <set>
#include <thread>
#include <atomic>

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
     * The get data callback.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool GetData(core::Call& call);

    /**
     * The get metadata callback.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool GetMetaData(core::Call& call);

    /**
     * The set wishlist callback.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool SetWishlist(core::Call& call);

private:
    /** Callee slot requesting images from this module */
    core::CalleeSlot callRequestImage;

    /** Image file path slot */
    core::param::ParamSlot filenameSlot;

    /** Boolean slot indicating whether all listed images should be loaded directly */
    core::param::ParamSlot loadEverythingSlot;

    /** Slot determining the maximum memory occupation of the image data */
    core::param::ParamSlot maximumMemoryOccupationSlot;

    /** Pointer to the vector containing the images */
    std::shared_ptr<image_calls::Image2DCall::ImageMap> imageData;

    /** List of all available image files */
    std::shared_ptr<std::vector<std::string>> availableFiles;

    /** Set of all elements in the queue to avoid duplication */
    std::set<std::string> queueElements;

    /** Queue containing the paths of the images to load */
    std::queue<std::string> imageLoadingQueue;

    /**  */
    std::thread loadingThread;

    /** Mutex to protect the queue from race conditions */
    std::mutex queueMutex;

    /** Flag telling the seperate loader thread when to stop */
    std::atomic_bool keepRunning = true;

    /** hash value for the data */
    SIZE_T datahash;

    /**
     * Loads an image from the harddrive using the given path
     *
     * @param path The path of the image file.
     * @return True, if the image could be loaded, false otherwise.
     */
    bool loadImage(const std::filesystem::path& path);

    /**
     * Loop responsible for loading all images that are present in the imageLoadingQueue
     */
    void loadingLoop(void);
};

} // namespace imageviewer2
} // namespace megamol

#endif // !MEGAMOLCORE_IMAGELOADER_H_INCLUDED
