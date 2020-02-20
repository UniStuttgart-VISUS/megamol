/*
 * ImageLoader.cpp
 *
 * Copyright (C) 2019 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "imageviewer2/ImageLoader.h"
#include "imageviewer2/JpegBitmapCodec.h"
#include "mmcore/misc/PngBitmapCodec.h"
#include "vislib/graphics/BitmapCodecCollection.h"

#include <filesystem>
#include <functional>
#include <iostream>
#include "image_calls/Image2DCall.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FloatParam.h"

using namespace megamol;
using namespace megamol::core;
using namespace megamol::imageviewer2;

/*
 * ImageLoader::ImageLoader
 */
ImageLoader::ImageLoader(void)
    : Module()
    , callRequestImage("requestImage", "Slot that provides the data of the loaded images")
    , filenameSlot("filepath",
          "Path to the image file (*.png, *.bmp) that should be loaded. If the file to load has a *.txt extension the "
          "file will be treated as list of image paths. The module will then load all of the listed images.")
    , loadEverythingSlot("loadEverything", "Forces this module to ignore the maxMemory value and all loading wishes by "
                                           "callers by loading all given data by default.")
    , maximumMemoryOccupationSlot("maxMemory",
          "The maximum memory in Gigabyte that will be occupied by the loaded data. This value can be "
          "ignored by selecting the loadEverything option.")
    , imageData(std::make_shared<image_calls::Image2DCall::ImageMap>())
    , availableFiles(std::make_shared<std::vector<std::string>>())
    , datahash(0) {

    this->callRequestImage.SetCallback(image_calls::Image2DCall::ClassName(),
        image_calls::Image2DCall::FunctionName(image_calls::Image2DCall::CallForGetData), &ImageLoader::GetData);
    this->callRequestImage.SetCallback(image_calls::Image2DCall::ClassName(),
        image_calls::Image2DCall::FunctionName(image_calls::Image2DCall::CallForGetMetaData),
        &ImageLoader::GetMetaData);
    this->callRequestImage.SetCallback(image_calls::Image2DCall::ClassName(),
        image_calls::Image2DCall::FunctionName(image_calls::Image2DCall::CallForSetWishlist),
        &ImageLoader::SetWishlist);
    this->callRequestImage.SetCallback(image_calls::Image2DCall::ClassName(),
        image_calls::Image2DCall::FunctionName(image_calls::Image2DCall::CallForWaitForData),
        &ImageLoader::WaitForData);
    this->MakeSlotAvailable(&this->callRequestImage);

    this->filenameSlot.SetParameter(new param::FilePathParam(""));
    this->MakeSlotAvailable(&this->filenameSlot);

    this->loadEverythingSlot.SetParameter(new param::BoolParam(true));
    this->MakeSlotAvailable(&this->loadEverythingSlot);

    this->maximumMemoryOccupationSlot.SetParameter(new param::FloatParam(4.0f, 0.1f));
    this->MakeSlotAvailable(&this->maximumMemoryOccupationSlot);
}

/*
 * ImageLoader::~ImageLoader
 */
ImageLoader::~ImageLoader(void) { this->Release(); }

/*
 * ImageLoader::create
 */
bool ImageLoader::create(void) {
    vislib::graphics::BitmapCodecCollection::DefaultCollection().AddCodec(new sg::graphics::PngBitmapCodec());
    vislib::graphics::BitmapCodecCollection::DefaultCollection().AddCodec(new sg::graphics::JpegBitmapCodec());
    this->loadingThread = std::thread(&ImageLoader::loadingLoop, std::ref(*this)); // start loading thread
    return true;
}

/*
 * ImageLoader::release
 */
void ImageLoader::release(void) {
    this->keepRunning = false;
    if (this->loadingThread.joinable()) {
        this->loadingThread.join();
    }
}

/*
 * ImageLoader::GetData
 */
bool ImageLoader::GetData(core::Call& call) {
    image_calls::Image2DCall* ic = dynamic_cast<image_calls::Image2DCall*>(&call);
    if (ic == nullptr) return false;

    if (this->newImageAvailable) {
        this->imageMutex.lock();
        this->imageData->insert(this->newImageData.begin(), this->newImageData.end());
        this->newImageData.clear();
        this->imageMutex.unlock();
        this->newImageAvailable = false;
    }

    ic->SetImagePtr(this->imageData);
    ic->SetDataHash(this->datahash);

    return true;
}

/*
 * ImageLoader::GetMetaData
 */
bool ImageLoader::GetMetaData(core::Call& call) {
    image_calls::Image2DCall* ic = dynamic_cast<image_calls::Image2DCall*>(&call);
    if (ic == nullptr) return false;

    if (this->filenameSlot.IsDirty()) {
        this->filenameSlot.ResetDirty();
        this->imageData->clear();
        auto tpath = this->filenameSlot.Param<param::FilePathParam>()->Value();
        std::filesystem::path path(tpath.PeekBuffer());

        this->availableFiles->clear();

        // check path extension
        if (path.has_extension() && path.extension().string().compare(".txt") != 0) { // normal file
            this->availableFiles->push_back(path.string());
        } else { // list of files
            std::ifstream file(path);
            if (file.is_open()) {
                std::string line;
                while (std::getline(file, line)) {
                    this->availableFiles->push_back(line);
                }
                if (this->loadEverythingSlot.Param<param::BoolParam>()->Value()) {
                    this->queueMutex.lock();
                    this->queueElements.insert(this->availableFiles->begin(), this->availableFiles->end());
                    for (const auto& e : this->queueElements) {
                        this->imageLoadingQueue.push(e);
                    }
                    this->queueMutex.unlock();
                }
            } else {
                vislib::sys::Log::DefaultLog.WriteError("ImageLoader: The file \"%s\" could not be opened", path);
                return false;
            }
        }
        ++this->datahash;
    }
    ic->SetAvailablePathsPtr(this->availableFiles);
    ic->SetDataHash(this->datahash);
    return true;
}

/*
 * ImageLoader::SetWishlist
 */
bool ImageLoader::SetWishlist(core::Call& call) { 
    image_calls::Image2DCall* ic = dynamic_cast<image_calls::Image2DCall*>(&call);
    if (ic == nullptr) return false;
    const auto wishlist = ic->GetWishlistPtr();

    if (wishlist == nullptr) {
        for (const auto& e : *this->availableFiles) {
            this->queueMutex.lock();
            if (imageData->count(e) == 0 && this->queueElements.count(e) == 0) {
                this->imageLoadingQueue.push(e);
                this->queueElements.insert(e);
            }
            this->queueMutex.unlock();
        }
    } else {
        for (const auto& id : *wishlist) {
            if (id >= wishlist->size()) {
                vislib::sys::Log::DefaultLog.WriteError("There is no image with the id %u", static_cast<unsigned int>(id));
                continue;
            }
            const auto& e = this->availableFiles->at(id);
            this->queueMutex.lock();
            if (imageData->count(e) == 0 && this->queueElements.count(e) == 0) {
                this->imageLoadingQueue.push(e);
                this->queueElements.insert(e);
            }
            this->queueMutex.unlock();
        }
    }

    return true; 
}

/*
 * ImageLoader::WaitForData
 */
bool ImageLoader::WaitForData(core::Call& call) {
    std::mutex waitmutex; // this has to be different from the queueMutex to avoid deadlocks
    std::unique_lock<std::mutex> lock(this->queueMutex);
    std::condition_variable condvar;
    condvar.wait(lock, [this] { return queueElements.empty(); });
    return true;
}

/*
 * ImageLoader::loadImage
 */
bool ImageLoader::loadImage(const std::filesystem::path& path) {
    if (!std::filesystem::is_regular_file(path)) {
        vislib::sys::Log::DefaultLog.WriteError(
            "ImageLoader: Could not open the file \"%s\" because it is no regular file", path.c_str());
    }
    auto fileSize = std::filesystem::file_size(path);
    std::vector<uint8_t> loadedFile;
    loadedFile.resize(fileSize);
    std::ifstream file(path, std::ios::binary);
    if (file.is_open()) {
        file.read(reinterpret_cast<char*>(loadedFile.data()), fileSize);
    } else {
        vislib::sys::Log::DefaultLog.WriteError("ImageLoader: Could not open the file \"%s\" from disk", path.c_str());
    }

    vislib::graphics::BitmapImage image;

    if (vislib::graphics::BitmapCodecCollection::DefaultCollection().LoadBitmapImage(
            image, loadedFile.data(), fileSize)) {
        image.Convert(vislib::graphics::BitmapImage::TemplateByteRGB);
        this->imageMutex.lock();
        this->newImageData.insert(std::pair(path.string(), image));
        this->imageMutex.unlock();
    } else {
        vislib::sys::Log::DefaultLog.WriteError("ImageLoader: failed decoding file \"%s\"", path.c_str());
        return false;
    }

    return true;
}

/*
 * ImageLoader::loadingLoop
 */
void ImageLoader::loadingLoop(void) {
    while (this->keepRunning) {
        if (!this->imageLoadingQueue.empty()) {
            this->queueMutex.lock();
            auto elem = this->imageLoadingQueue.front();
            this->imageLoadingQueue.pop();
            this->loadImage(elem);
            this->queueElements.erase(elem);
            this->queueMutex.unlock();
            this->newImageAvailable = true;
        }
    }
}
