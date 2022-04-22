#ifndef IMAGESERIES_SRC_MODULE_IMAGESERIESGRAPHGENERATOR_HPP_
#define IMAGESERIES_SRC_MODULE_IMAGESERIESGRAPHGENERATOR_HPP_

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

#include "imageseries/graph/GraphData2D.h"
#include "imageseries/graph/GraphData2DCall.h"

#include "imageseries/ImageSeries2DCall.h"

#include "../blob/BlobGraphBuilder.h"

namespace megamol::ImageSeries {

/**
 * Labels connected components in monochrome images within an image series.
 */
class ImageSeriesGraphGenerator : public core::Module {

public:
    /**
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "ImageSeriesGraphGenerator";
    }

    /**
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Generates a transport network from a labeled image series, spanning all frames in the series.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    ImageSeriesGraphGenerator();

    ~ImageSeriesGraphGenerator() override;

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
     * Callback for changes to any of the filtering parameters.
     */
    bool filterParametersChangedCallback(core::param::ParamSlot& param);

private:
    ImageSeries2DCall::Output requestFrame(core::CallerSlot& caller, double time);
    ImageSeries2DCall::Output requestFrameByIndex(core::CallerSlot& caller, std::size_t index);

    core::CalleeSlot getDataCallee;

    core::CallerSlot getInputCaller;
    core::CallerSlot getLabelsCaller;

    core::param::ParamSlot flowFrontParam;

    AsyncImageData2D::Hash labelHash = 0;
    AsyncImageData2D::Hash valueHash = 0;

    std::shared_ptr<blob::BlobGraphBuilder> graphBuilder;
    std::shared_ptr<const graph::AsyncGraphData2D> asyncGraph;
};

} // namespace megamol::ImageSeries

#endif
