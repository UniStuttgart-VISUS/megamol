/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "../filter/AsyncFilterRunner.h"
#include "../filter/FlowTimeLabelFilter.h"
#include "../util/LRUCache.h"

#include "imageseries/AsyncImageData2D.h"
#include "imageseries/ImageSeries2DCall.h"
#include "imageseries/graph/GraphData2D.h"

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

#include <memory>

namespace megamol::ImageSeries {

/**
 * Labels connected components across a timestamp-mapped single image generated from a series.
 */
class ImageSeriesFlowLabeler : public core::Module {

public:
    /**
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "ImageSeriesFlowLabeler";
    }

    /**
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Labels connected components across a timestamp-mapped single image generated from a series.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    ImageSeriesFlowLabeler();

    ~ImageSeriesFlowLabeler() override;

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
    core::CalleeSlot getGraphCallee;

    core::CallerSlot getTimeMapCaller;

    core::param::ParamSlot inflowAreaParam;
    core::param::ParamSlot inflowMarginParam;
    core::param::ParamSlot velocityMethodParam;
    core::param::ParamSlot outputImageParam;

    core::param::ParamSlot isolatedParam;
    core::param::ParamSlot falseSourcesParam;
    core::param::ParamSlot falseSinksParam;

    core::param::ParamSlot combineTrivialParam;
    core::param::ParamSlot removeTrivialParam;
    core::param::ParamSlot resolveDiamondsParam;
    core::param::ParamSlot minAreaParam;
    core::param::ParamSlot keepBreakthroughNodesParam;
    core::param::ParamSlot keepVelocityJumpsParam;
    core::param::ParamSlot keepVelocityJumpsFactorParam;

    core::param::ParamSlot outputGraphsParam;
    core::param::ParamSlot outputLabelImagesParam;
    core::param::ParamSlot outputTimeImagesParam;
    core::param::ParamSlot outputPathParam;

    util::LRUCache<typename AsyncImageData2D<filter::FlowTimeLabelFilter::Output>::Hash,
        AsyncImageData2D<filter::FlowTimeLabelFilter::Output>>
        imageCache;

    std::unique_ptr<filter::AsyncFilterRunner<AsyncImageData2D<filter::FlowTimeLabelFilter::Output>>> filterRunner;
};

} // namespace megamol::ImageSeries
