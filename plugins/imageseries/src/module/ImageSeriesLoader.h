#ifndef IMAGESERIES_SRC_MODULE_IMAGESERIESLOADER_HPP_
#define IMAGESERIES_SRC_MODULE_IMAGESERIESLOADER_HPP_

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/utility/graphics/BitmapCodecCollection.h"

#include "imageseries/AsyncImageData2D.h"
#include "imageseries/ImageSeries2DCall.h"

#include "../util/LRUCache.h"

#include <filesystem>

namespace megamol::ImageSeries {

namespace filter {
class AsyncFilterRunner;
}

/**
 * Data source for multiple 2D images, loaded from a directory in the local file system.
 */
class ImageSeriesLoader : public core::Module {

public:
    /**
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "ImageSeriesLoader";
    }

    /**
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Data source module for an image series comprising multiple image files.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    ImageSeriesLoader();

    ~ImageSeriesLoader() override;

protected:
    /**
     * Initializes this loader instance.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() override;

    /**
     * Releases all resources used by this loader instance.
     */
    void release() override;

    /**
     * Implementation of the getData call.
     */
    bool getDataCallback(core::Call& caller);

    /**
     * Implementation of the getMetaData call.
     */
    bool getMetaDataCallback(core::Call& caller);

    /**
     * Callback for changes to the 'path' parameter.
     */
    bool pathChangedCallback(core::param::ParamSlot& param);

    /**
     * Callback for changes to the 'pattern' parameter.
     */
    bool patternChangedCallback(core::param::ParamSlot& param);

    /**
     * Callback for changes to the 'image format' parameter.
     */
    bool formatChangedCallback(core::param::ParamSlot& param);

private:
    void refreshDirectory();
    void filterImageFiles();
    void updateMetadata();

    std::size_t timestampToFrameIndex(double timestamp) const;
    double frameIndexToTimestamp(std::size_t index) const;

    std::shared_ptr<vislib::graphics::BitmapImage> loadImageFile(const std::filesystem::path& path) const;

    std::unique_ptr<filter::AsyncFilterRunner> filterRunner;

    core::CalleeSlot getDataCallee;

    /// Path to the directory to read image files from
    core::param::ParamSlot pathParam;

    /// Regex pattern to filter image series by
    core::param::ParamSlot patternParam;

    std::vector<std::filesystem::path> imageFilesUnfiltered;
    std::vector<std::filesystem::path> imageFilesFiltered;

    util::LRUCache<std::uint32_t, AsyncImageData2D> imageCache;

    ImageSeries2DCall::Output metadata;

    std::shared_ptr<vislib::graphics::BitmapCodecCollection> getBitmapCodecs() const;
};

} // namespace megamol::ImageSeries

#endif
