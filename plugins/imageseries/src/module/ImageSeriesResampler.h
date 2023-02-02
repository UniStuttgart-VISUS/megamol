#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

#include "imageseries/AffineTransform2DCall.h"
#include "imageseries/AsyncImageData2D.h"
#include "imageseries/ImageSeries2DCall.h"

#include "../filter/AsyncFilterRunner.h"
#include "../registration/AsyncImageRegistrator.h"
#include "../util/LRUCache.h"

namespace megamol::ImageSeries {

/**
 * Data source for multiple 2D images, loaded from a directory in the local file system.
 */
class ImageSeriesResampler : public core::Module {

public:
    /**
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "ImageSeriesResampler";
    }

    /**
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Spatially and temporally aligns an input image series to a reference image series.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    ImageSeriesResampler();

    ~ImageSeriesResampler() override;

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
     * Implementation of the getTransform call.
     */
    bool getTransformCallback(core::Call& caller);

    /**
     * Callback for changes to any of the timestamp parameters.
     */
    bool timestampChangedCallback(core::param::ParamSlot& param);

    /**
     * Callback for changes to any of the image registration parameters.
     */
    bool registrationCallback(core::param::ParamSlot& param);

private:
    double toAlignedTimestamp(double timestamp) const;
    double fromAlignedTimestamp(double timestamp) const;

    ImageSeries2DCall::Output transformMetadata(ImageSeries2DCall::Output metadata) const;

    static double transformTimestamp(double timestamp, double min1, double max1, double min2, double max2);

    std::shared_ptr<const AsyncImageData2D<>> fetchImage(core::CallerSlot& caller, double timestamp) const;

    void updateTransformationMatrix();

    core::CalleeSlot getDataCallee;
    core::CalleeSlot getTransformCallee;

    core::CallerSlot getInputCaller;
    core::CallerSlot getReferenceCaller;
    core::CallerSlot getTransformCaller;

    /// Reference points for temporally aligning the dataset
    core::param::ParamSlot keyTimeInput1Param;
    core::param::ParamSlot keyTimeReference1Param;
    core::param::ParamSlot keyTimeInput2Param;
    core::param::ParamSlot keyTimeReference2Param;

    core::param::ParamSlot imageRegistrationParam;
    core::param::ParamSlot imageRegistrationAutoParam;

    // TODO store transformation matrix as separate parameter
    std::unique_ptr<registration::AsyncImageRegistrator> registrator;
    glm::mat3x2 cachedTransformMatrix;
    bool suppressed = false;

    util::LRUCache<typename AsyncImageData2D<>::Hash, AsyncImageData2D<>> imageCache;

    std::unique_ptr<filter::AsyncFilterRunner<>> filterRunner;
};

} // namespace megamol::ImageSeries
