#include "ImageSeriesLoader.h"
#include "imageseries/ImageSeries2DCall.h"

#include "mmcore/misc/PngBitmapCodec.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/graphics/BitmapCodecCollection.h"

#include <filesystem>
#include <regex>

using Log = megamol::core::utility::log::Log;

namespace megamol::ImageSeries {

ImageSeriesLoader::ImageSeriesLoader()
        : getDataCallee("getData", "Returns data from the image series for the requested timestamp.")
        , pathParam("Path", "Directory from which image files should be loaded.")
        , patternParam("Filename pattern", "Regular expression to filter file names by.")
        , imageCache([](const AsyncImageData2D& imageData) { return imageData.getByteSize(); }) {

    getDataCallee.SetCallback(ImageSeries2DCall::ClassName(),
        ImageSeries2DCall::FunctionName(ImageSeries2DCall::CallGetData), &ImageSeriesLoader::getDataCallback);
    getDataCallee.SetCallback(ImageSeries2DCall::ClassName(),
        ImageSeries2DCall::FunctionName(ImageSeries2DCall::CallGetMetaData), &ImageSeriesLoader::getMetaDataCallback);
    MakeSlotAvailable(&getDataCallee);

    pathParam << new core::param::FilePathParam("", core::param::FilePathParam::Flag_Directory);
    pathParam.SetUpdateCallback(&ImageSeriesLoader::pathChangedCallback);
    MakeSlotAvailable(&pathParam);

    patternParam << new core::param::StringParam("\\.png");
    patternParam.SetUpdateCallback(&ImageSeriesLoader::patternChangedCallback);
    MakeSlotAvailable(&patternParam);

    // Copy bitmap codec collection (for thread-safety)
    bitmapCodecCollection = std::make_shared<vislib::graphics::BitmapCodecCollection>(
        vislib::graphics::BitmapCodecCollection::BuildDefaultCollection());

    // Support loading PNG files
    bitmapCodecCollection->AddCodec(new sg::graphics::PngBitmapCodec);

    // Set default image cache size to 512 MB
    imageCache.setMaximumSize(512 * 1024 * 1024);
}

ImageSeriesLoader::~ImageSeriesLoader() {
    Release();
}

bool ImageSeriesLoader::create() {
    return true;
}

void ImageSeriesLoader::release() {
    imageCache.clear();
}

bool ImageSeriesLoader::getDataCallback(core::Call& caller) {
    if (auto call = dynamic_cast<ImageSeries2DCall*>(&caller)) {
        // Copy base metadata fields
        auto output = metadata;

        // TODO Obtain suitable frame
        output.imageIndex = timestampToFrameIndex(call->GetInput().time);
        output.resultTime = frameIndexToTimestamp(output.imageIndex);

        if (output.imageIndex < imageFilesFiltered.size()) {
            // TODO load image asynchronously
            const auto& path = imageFilesFiltered[output.imageIndex];
            output.filename = path.string();
            output.imageData = imageCache.findOrCreate(output.imageIndex,
                [&](std::uint32_t) { return std::make_shared<AsyncImageData2D>(loadImageFile(path)); });
        }

        // TODO validate that width and height match series metadata

        // Write metadata and image data
        call->SetOutput(std::move(output));
        return true;
    } else {
        return false;
    }
}

bool ImageSeriesLoader::getMetaDataCallback(core::Call& caller) {
    if (auto call = dynamic_cast<ImageSeries2DCall*>(&caller)) {
        // Write only metadata without image data.
        call->SetOutput(metadata);
        return true;
    } else {
        return false;
    }
}

bool ImageSeriesLoader::pathChangedCallback(core::param::ParamSlot& param) {
    // TODO mutex?
    refreshDirectory();
    return true;
}

bool ImageSeriesLoader::patternChangedCallback(core::param::ParamSlot& param) {
    // TODO mutex?
    filterImageFiles();
    return true;
}

bool ImageSeriesLoader::formatChangedCallback(core::param::ParamSlot& param) {
    // TODO mutex?
    updateMetadata();
    return true;
}

void ImageSeriesLoader::refreshDirectory() {
    // TODO split loading logic into separate class
    // TODO perform directory iteration asynchronously
    auto path = pathParam.Param<core::param::FilePathParam>()->Value();

    imageFilesUnfiltered.clear();

    for (auto& entry : std::filesystem::recursive_directory_iterator(path)) {
        imageFilesUnfiltered.push_back(entry.path());
    }

    std::sort(imageFilesUnfiltered.begin(), imageFilesUnfiltered.end());

    filterImageFiles();
}

void ImageSeriesLoader::filterImageFiles() {
    imageFilesFiltered.clear();

    std::regex pattern(patternParam.Param<core::param::StringParam>()->Value(), std::regex_constants::icase);
    for (auto& entry : imageFilesUnfiltered) {
        if (std::regex_search(entry.string(), pattern)) {
            imageFilesFiltered.push_back(entry);
        }
    }

    updateMetadata();
}

void ImageSeriesLoader::updateMetadata() {
    metadata = {};

    // No image files -> no data
    if (imageFilesFiltered.empty()) {
        return;
    }

    metadata.imageCount = imageFilesFiltered.size();
    metadata.minimumTime = 0; // TODO allow customizing bounds per-dataset
    metadata.maximumTime = 1; // TODO allow customizing bounds per-dataset
    metadata.framerate = metadata.imageCount / (metadata.maximumTime - metadata.minimumTime);

    // Find first valid image in series to populate metadata
    for (auto& path : imageFilesFiltered) {
        if (auto image = loadImageFile(path)) {
            metadata.width = image->Width();
            metadata.height = image->Height();
            break;
        }
    }
}

std::size_t ImageSeriesLoader::timestampToFrameIndex(double timestamp) const {
    double normalized = (timestamp - metadata.minimumTime) / (metadata.maximumTime - metadata.minimumTime);
    // Ensure that we never end up with a negative index
    std::int32_t limit = std::max<std::int32_t>(metadata.imageCount, 1) - 1;
    return std::max<std::int32_t>(0, std::min<std::int32_t>(normalized * limit, limit));
}

double ImageSeriesLoader::frameIndexToTimestamp(std::size_t index) const {
    // Ensure that we never divide by 0
    double limit = std::max<std::int32_t>(metadata.imageCount, 2) - 1;
    return metadata.minimumTime + (index / limit) * (metadata.maximumTime - metadata.minimumTime);
}

std::shared_ptr<vislib::graphics::BitmapImage> ImageSeriesLoader::loadImageFile(
    const std::filesystem::path& path) const {

    const char* filename = path.c_str();
    try {
        auto img = std::make_shared<vislib::graphics::BitmapImage>();
        if (!bitmapCodecCollection->LoadBitmapImage(*img, filename)) {
            throw vislib::Exception("No suitable codec found", __FILE__, __LINE__);
        }
        return img;
    } catch (vislib::Exception& ex) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load image '%s': %s (%s, %d)\n", filename, ex.GetMsgA(),
            ex.GetFile(), ex.GetLine());
    } catch (...) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load image '%s': unexpected exception\n", filename);
    }

    return nullptr;
}


} // namespace megamol::ImageSeries
