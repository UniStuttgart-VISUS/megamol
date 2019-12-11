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
#include "image_calls/Image2DCall_2.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FilePathParam.h"

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
    , imageData(std::make_shared<image_calls::Image2DCall_2::ImageVector>())
    , datahash(0) {

    this->callRequestImage.SetCallback(
        image_calls::Image2DCall_2::ClassName(), image_calls::Image2DCall_2::FunctionName(0), &ImageLoader::GetData);
    this->MakeSlotAvailable(&this->callRequestImage);

    this->filenameSlot.SetParameter(new param::FilePathParam(""));
    this->MakeSlotAvailable(&this->filenameSlot);
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
    return true;
}

/*
 * ImageLoader::release
 */
void ImageLoader::release(void) {}

/*
 * ImageLoader::GetData
 */
bool ImageLoader::GetData(core::Call& call) {
    image_calls::Image2DCall_2* ic = dynamic_cast<image_calls::Image2DCall_2*>(&call);
    if (ic == nullptr) return false;

    if (this->filenameSlot.IsDirty()) {
        this->filenameSlot.ResetDirty();
        this->imageData->clear();
        auto tpath = this->filenameSlot.Param<param::FilePathParam>()->Value();
        std::filesystem::path path(tpath.PeekBuffer());

        // check path extension
        if (path.has_extension() && path.extension().string().compare(".txt") != 0) { // normal file
            if (!this->loadImage(path.string())) return false;
        } else { // list of files
            std::ifstream file(path);
            if (file.is_open()) {
                std::string line;
                while (std::getline(file, line)) {
                    std::filesystem::path curPath(line);
                    this->loadImage(curPath);
                }
            }
        }
        ++this->datahash;
    }

    ic->SetImagePtr(this->imageData);
    ic->SetDataHash(this->datahash);

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
        this->imageData->push_back(std::make_pair(image, path.string()));
    } else {
        vislib::sys::Log::DefaultLog.WriteError("ImageLoader: failed decoding file \"%s\"", path.c_str());
        return false;
    }

    return true;
}
