#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

#include "imageseries/AsyncImageData2D.h"
#include "imageseries/ImageSeries2DCall.h"
#include "imageseries/graph/GraphData2D.h"

#include "../filter/AsyncFilterRunner.h"
#include "../filter/FlowTimeLabelFilter.h"
#include "../util/LRUCache.h"

#include <memory>
#include <tuple>

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

    core::param::ParamSlot outputImageParam;
    core::param::ParamSlot inflowAreaParam;
    core::param::ParamSlot inflowMarginParam;
    core::param::ParamSlot minObstacleSizeParam;
    core::param::ParamSlot minAreaParam;

    core::param::ParamSlot isolatedParam;
    core::param::ParamSlot falseSourcesParam;
    core::param::ParamSlot falseSinksParam;
    core::param::ParamSlot unimportantSinksParam;
    core::param::ParamSlot resolveDiamondsParam;
    core::param::ParamSlot combineTrivialParam;
    core::param::ParamSlot combineTinyParam;

    util::LRUCache<typename AsyncImageData2D<filter::FlowTimeLabelFilter::Output>::Hash,
        AsyncImageData2D<filter::FlowTimeLabelFilter::Output>>
        imageCache;

    std::unique_ptr<filter::AsyncFilterRunner<AsyncImageData2D<filter::FlowTimeLabelFilter::Output>>>
        filterRunner;
};

} // namespace megamol::ImageSeries
