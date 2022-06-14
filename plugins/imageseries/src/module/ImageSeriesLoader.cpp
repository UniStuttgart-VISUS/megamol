#include "ImageSeriesLoader.h"
#include "imageseries/ImageSeries2DCall.h"

#include "../filter/AsyncFilterRunner.h"
#include "../filter/ImageLoadFilter.h"

#include "mmcore/misc/PngBitmapCodec.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/graphics/BitmapCodecCollection.h"

#include <charconv>
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

    // Set default image cache size to 512 MB
    imageCache.setMaximumSize(512 * 1024 * 1024);
}

ImageSeriesLoader::~ImageSeriesLoader() {
    Release();
}

bool ImageSeriesLoader::create() {
    filterRunner = std::make_unique<filter::AsyncFilterRunner>();
    return true;
}

void ImageSeriesLoader::release() {
    imageCache.clear();
    filterRunner = nullptr;
}

bool ImageSeriesLoader::getDataCallback(core::Call& caller) {
    if (auto call = dynamic_cast<ImageSeries2DCall*>(&caller)) {
        // Copy base metadata fields
        auto output = outputPrototype;

        // TODO Obtain suitable frame
        output.imageIndex = timestampToFrameIndex(call->GetInput().time);
        output.resultTime = frameIndexToTimestamp(output.imageIndex);

        if (output.imageIndex < imageFilesFiltered.size()) {
            const auto& path = imageFilesFiltered[output.imageIndex];
            output.filename = path.string();

            ImageMetadata meta = metadata;
            meta.index = output.imageIndex;
            meta.valid = true;
            output.imageData = imageCache.findOrCreate(output.imageIndex, [&](std::uint32_t) {
                return filterRunner->run<ImageSeries::filter::ImageLoadFilter>(getBitmapCodecs(), path, meta);
            });
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
        call->SetOutput(outputPrototype);
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

inline bool lexicalNumericComparatorImpl(const char* a, const char* aEnd, const char* b, const char* bEnd) {
    if (a >= aEnd || b >= bEnd) {
        return a >= aEnd;
    } else if (std::isdigit(a[0]) != std::isdigit(b[0])) {
        return std::isdigit(a[0]);
    } else if (!std::isdigit(a[0])) {
        return a[0] == b[0] ? lexicalNumericComparatorImpl(a + 1, aEnd, b + 1, bEnd) : a[0] < b[0];
    }

    int numA = 0, numB = 0;
    auto subA = std::from_chars(a, aEnd, numA);
    auto subB = std::from_chars(b, bEnd, numB);
    if (numA != numB) {
        return numA < numB;
    }

    return lexicalNumericComparatorImpl(subA.ptr, aEnd, subB.ptr, bEnd);
}

inline bool lexicalNumericComparator(const std::string& a, const std::string& b) {
    return lexicalNumericComparatorImpl(a.data(), a.data() + a.size(), b.data(), b.data() + b.size());
}

void ImageSeriesLoader::refreshDirectory() {
    // TODO split loading logic into separate class
    // TODO perform directory iteration asynchronously
    auto path = pathParam.Param<core::param::FilePathParam>()->Value();

    imageFilesUnfiltered.clear();

    for (auto& entry : std::filesystem::recursive_directory_iterator(path)) {
        imageFilesUnfiltered.push_back(entry.path());
    }

    std::sort(imageFilesUnfiltered.begin(), imageFilesUnfiltered.end(), &lexicalNumericComparator);
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
    imageCache.clear();
    outputPrototype = {};

    // No image files -> no data
    if (imageFilesFiltered.empty()) {
        return;
    }

    outputPrototype.imageCount = imageFilesFiltered.size();
    outputPrototype.minimumTime = 0; // TODO allow customizing bounds per-dataset
    outputPrototype.maximumTime = 1; // TODO allow customizing bounds per-dataset
    outputPrototype.framerate =
        outputPrototype.imageCount / (outputPrototype.maximumTime - outputPrototype.minimumTime);

    // Find first valid image in series to populate metadata
    for (auto& path : imageFilesFiltered) {
        if (auto image = loadImageFile(path)) {
            outputPrototype.width = image->Width();
            outputPrototype.height = image->Height();
            outputPrototype.bytesPerPixel = image->BytesPerPixel();

            metadata.width = image->Width();
            metadata.height = image->Height();
            metadata.channels = image->GetChannelCount();
            metadata.bytesPerChannel = image->BytesPerPixel() / std::max<unsigned int>(1, image->GetChannelCount());
            metadata.filename = pathParam.Param<core::param::FilePathParam>()->Value();
            break;
        }
    }
}

std::size_t ImageSeriesLoader::timestampToFrameIndex(double timestamp) const {
    double normalized =
        (timestamp - outputPrototype.minimumTime) / (outputPrototype.maximumTime - outputPrototype.minimumTime);
    // Ensure that we never end up with a negative index
    std::int32_t limit = std::max<std::int32_t>(outputPrototype.imageCount, 1);
    return std::max<std::int32_t>(0, std::min<std::int32_t>(normalized * limit, limit - 1));
}

double ImageSeriesLoader::frameIndexToTimestamp(std::size_t index) const {
    // Ensure that we never divide by 0
    double limit = std::max<std::int32_t>(outputPrototype.imageCount, 2) - 1;
    return outputPrototype.minimumTime + (index / limit) * (outputPrototype.maximumTime - outputPrototype.minimumTime);
}

std::shared_ptr<vislib::graphics::BitmapImage> ImageSeriesLoader::loadImageFile(
    const std::filesystem::path& path) const {

    const char* filename = path.c_str();
    try {
        auto img = std::make_shared<vislib::graphics::BitmapImage>();
        if (!getBitmapCodecs()->LoadBitmapImage(*img, filename)) {
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

std::shared_ptr<vislib::graphics::BitmapCodecCollection> ImageSeriesLoader::getBitmapCodecs() const {
    // Copy bitmap codec collection (for thread-safety)
    auto bitmapCodecCollection = std::make_shared<vislib::graphics::BitmapCodecCollection>(
        vislib::graphics::BitmapCodecCollection::BuildDefaultCollection());
    bitmapCodecCollection->AddCodec(new sg::graphics::PngBitmapCodec);
    return bitmapCodecCollection;
}


} // namespace megamol::ImageSeries
