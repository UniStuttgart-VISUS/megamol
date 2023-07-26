/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "../filter/AsyncFilterRunner.h"
#include "../util/LRUCache.h"

#include "imageseries/AsyncImageData2D.h"
#include "imageseries/ImageSeries2DCall.h"

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

#include "vislib/graphics/BitmapCodecCollection.h"

#include <memory>
#include <string>
#include <vector>

namespace megamol::ImageSeries {

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

    std::shared_ptr<vislib::graphics::BitmapImage> loadImageFile(const std::string& path) const;

    std::unique_ptr<filter::AsyncFilterRunner<>> filterRunner;

    core::CalleeSlot getDataCallee;

    /// Path to the directory to read image files from
    core::param::ParamSlot pathParam;

    /// Regex pattern to filter image series by
    core::param::ParamSlot patternParam;

    std::vector<std::string> imageFilesUnfiltered;
    std::vector<std::string> imageFilesFiltered;

    util::LRUCache<std::uint32_t, AsyncImageData2D<>> imageCache;

    ImageMetadata metadata;
    ImageSeries2DCall::Output outputPrototype;

    std::shared_ptr<vislib::graphics::BitmapCodecCollection> getBitmapCodecs() const;
};

} // namespace megamol::ImageSeries
