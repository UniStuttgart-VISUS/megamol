#include "ImageSeriesGraphGenerator.h"
#include "imageseries/ImageSeries2DCall.h"
#include "imageseries/graph/GraphData2DCall.h"

#include "mmcore/misc/PngBitmapCodec.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/StringParam.h"

#include "../util/ImageUtils.h"

using Log = megamol::core::utility::log::Log;
using Presentation = megamol::core::param::AbstractParamPresentation::Presentation;

namespace megamol::ImageSeries {

ImageSeriesGraphGenerator::ImageSeriesGraphGenerator()
        : getDataCallee("getData", "Returns the resulting graph.")
        , getInputCaller("requestInputImageSeries", "Requests gray value data from an image series.")
        , getLabelsCaller("requestLabelImageSeries", "Requests label data from an image series.") {

    getInputCaller.SetCompatibleCall<typename ImageSeries2DCall::CallDescription>();
    MakeSlotAvailable(&getInputCaller);

    getLabelsCaller.SetCompatibleCall<typename ImageSeries2DCall::CallDescription>();
    MakeSlotAvailable(&getLabelsCaller);

    getDataCallee.SetCallback(GraphData2DCall::ClassName(), GraphData2DCall::FunctionName(GraphData2DCall::CallGetData),
        &ImageSeriesGraphGenerator::getDataCallback);
    MakeSlotAvailable(&getDataCallee);
}

ImageSeriesGraphGenerator::~ImageSeriesGraphGenerator() {
    Release();
}

bool ImageSeriesGraphGenerator::create() {
    return true;
}

void ImageSeriesGraphGenerator::release() {
    graphBuilder = nullptr;
    asyncGraph = nullptr;
}

bool ImageSeriesGraphGenerator::getDataCallback(core::Call& caller) {
    if (auto* call = dynamic_cast<GraphData2DCall*>(&caller)) {
        // Query first images
        auto initInput = requestFrame(getInputCaller, 0.0);
        auto initLabel = requestFrame(getLabelsCaller, 0.0);

        // Compare hashes to detect changes
        if (valueHash != initInput.getHash() || labelHash != initLabel.getHash()) {
            valueHash = initInput.getHash();
            labelHash = initLabel.getHash();
            asyncGraph = nullptr;
        }

        // Init graph builder if necessary
        if (asyncGraph == nullptr && initInput.imageData && initLabel.imageData) {
            graphBuilder = std::make_shared<blob::BlobGraphBuilder>();

            // TODO don't push frames all at once to avoid overloading work queue
            for (std::size_t i = 0; i < initInput.imageCount; ++i) {
                graphBuilder->addFrame(requestFrameByIndex(getLabelsCaller, i).imageData,
                    requestFrameByIndex(getInputCaller, i).imageData);
            }

            asyncGraph = graphBuilder->finalize();
        }

        GraphData2DCall::Output output;
        output.graph = asyncGraph;
        call->SetOutput(output);
        return true;
    }
    return false;
}

bool ImageSeriesGraphGenerator::filterParametersChangedCallback(core::param::ParamSlot& param) {
    graphBuilder = nullptr;
    asyncGraph = nullptr;
    return true;
}

ImageSeries2DCall::Output ImageSeriesGraphGenerator::requestFrame(core::CallerSlot& caller, double time) {
    if (auto* call = caller.CallAs<ImageSeries2DCall>()) {
        // Convert frame index into timestamp
        ImageSeries2DCall::Input input;
        input.time = time;
        call->SetInput(input);

        // Get image data
        if ((*call)(ImageSeries2DCall::CallGetData)) {
            return call->GetOutput();
        }
    }

    return ImageSeries2DCall::Output();
}

ImageSeries2DCall::Output ImageSeriesGraphGenerator::requestFrameByIndex(core::CallerSlot& caller, std::size_t index) {
    if (auto* call = caller.CallAs<ImageSeries2DCall>()) {
        // Retrieve metadata for timing info
        if ((*call)(ImageSeries2DCall::CallGetMetaData)) {
            // Convert frame index into timestamp
            return requestFrame(caller, call->GetOutput().minimumTime + (index + 0.5f) / call->GetOutput().framerate);
        }
    }

    return ImageSeries2DCall::Output();
}


} // namespace megamol::ImageSeries
