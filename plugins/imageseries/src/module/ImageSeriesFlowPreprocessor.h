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
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

#include <memory>

namespace megamol::ImageSeries {

/**
 * Preprocesses 2D image series of fluid flow experiments or simulations.
 */
class ImageSeriesFlowPreprocessor : public core::Module {

public:
    /**
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "ImageSeriesFlowPreprocessor";
    }

    /**
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Preprocesses 2D image series of fluid flow experiments or simulations.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    ImageSeriesFlowPreprocessor();

    ~ImageSeriesFlowPreprocessor() override;

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
    core::CalleeSlot getDataCallee;

    core::CallerSlot getInputCaller;
    core::CallerSlot getMaskCaller;

    core::param::ParamSlot maskFrameParam;
    core::param::ParamSlot deinterlaceParam;
    core::param::ParamSlot segmentationEnabledParam;
    core::param::ParamSlot segmentationThresholdParam;
    core::param::ParamSlot segmentationNegationParam;

    util::LRUCache<typename AsyncImageData2D<>::Hash, AsyncImageData2D<>> imageCache;

    std::unique_ptr<filter::AsyncFilterRunner<>> filterRunner;
};

} // namespace megamol::ImageSeries
