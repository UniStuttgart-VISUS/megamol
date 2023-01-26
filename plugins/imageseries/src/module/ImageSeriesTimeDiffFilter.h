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
class ImageSeriesTimeDiffFilter : public core::Module {

public:
    /**
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "ImageSeriesTimeDiffFilter";
    }

    /**
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Computes the weighted differences between adjacent frames in an image series.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    ImageSeriesTimeDiffFilter();

    ~ImageSeriesTimeDiffFilter() override;

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
    core::CallerSlot getReferenceCaller;

    core::param::ParamSlot frameCountParam;

    util::LRUCache<AsyncImageData2D::Hash, AsyncImageData2D> imageCache;

    std::unique_ptr<filter::AsyncFilterRunner> filterRunner;
};

} // namespace megamol::ImageSeries
