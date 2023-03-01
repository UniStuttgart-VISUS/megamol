#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

#include "imageseries/AsyncImageData2D.h"
#include "imageseries/ImageSeries2DCall.h"

#include "../filter/AsyncFilterRunner.h"
#include "../util/LRUCache.h"

namespace megamol::ImageSeries {

/**
 * Preprocesses 2D image series of fluid flow experiments or simulations.
 */
class ImageSeriesTimestampFilter : public core::Module {

public:
    /**
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "ImageSeriesTimestampFilter";
    }

    /**
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Generates a single image from an image series, encoding the frame index for the first non-zero value "
               "at each pixel";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    ImageSeriesTimestampFilter();

    ~ImageSeriesTimestampFilter() override;

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
     * Callback for changes to any of the filtering parameters.
     */
    bool filterParametersChangedCallback(core::param::ParamSlot& param);

private:
    static ImageSeries::ImageSeries2DCall::Output requestFrame(core::CallerSlot& source, double timestamp);
    static ImageSeries::ImageSeries2DCall::Output requestMetadata(core::CallerSlot& source);

    core::CalleeSlot getDataCallee;

    core::CallerSlot getInputCaller;

    core::param::ParamSlot denoiseIterations;
    core::param::ParamSlot denoiseNeighborThreshold;

    std::unique_ptr<filter::AsyncFilterRunner<>> filterRunner;

    util::LRUCache<typename AsyncImageData2D<>::Hash, AsyncImageData2D<>> imageCache;
};

} // namespace megamol::ImageSeries
