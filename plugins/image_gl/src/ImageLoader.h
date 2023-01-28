/**
 * MegaMol
 * Copyright (c) 2019, MegaMol Dev Team
 * All rights reserved.
 */
#ifndef MEGAMOLCORE_IMAGELOADER_H_INCLUDED
#define MEGAMOLCORE_IMAGELOADER_H_INCLUDED

#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <queue>
#include <set>
#include <thread>

#include "image_calls/Image2DCall.h"
#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

//#define LOADED_MESSAGE

namespace megamol::image_gl {

class ImageLoader : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "ImageLoader";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "A module that loads images from disk";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    /** Ctor. */
    ImageLoader();

    /** Dtor. */
    ~ImageLoader() override;

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() override;

    /**
     * Implementation of 'Release'.
     */
    void release() override;

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

    /**
     * The wait for data callback.
     *
     * @param call The calling call.
     *
     * @return The return value of the function
     */
    virtual bool WaitForData(core::Call& call);

    /**
     * The delete data callback.
     *
     * @param call The calling call.
     *
     * @return The return value of the function
     */
    virtual bool DeleteData(core::Call& call);

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

    /**  */
    image_calls::Image2DCall::ImageMap newImageData;

    /** List of all available image files */
    std::shared_ptr<std::vector<std::string>> availableFiles;

    /** Set of all elements in the queue to avoid duplication */
    std::set<std::string> queueElements;

    /** Queue containing the paths of the images to load */
    std::queue<std::string> imageLoadingQueue;

    /** Thread responsible for image loading */
    std::thread loadingThread;

    /** Mutex to protect the queue from race conditions */
    std::mutex queueMutex;

    /**  */
    std::mutex imageMutex;

    /** Flag telling the seperate loader thread when to stop */
    std::atomic_bool keepRunning = true;

    /** hash value for the data */
    SIZE_T datahash;

    std::atomic_bool newImageAvailable = true;

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
    void loadingLoop();

#ifdef LOADED_MESSAGE
    static uint32_t loaded;
#endif

    std::condition_variable condvar;
};

} // namespace megamol::image_gl

#endif // !MEGAMOLCORE_IMAGELOADER_H_INCLUDED
