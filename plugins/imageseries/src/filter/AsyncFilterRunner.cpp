#include "AsyncFilterRunner.h"

namespace megamol::ImageSeries::filter {

AsyncFilterRunner::AsyncFilterRunner() {}

AsyncFilterRunner::~AsyncFilterRunner() {}

AsyncFilterRunner::AsyncImageData AsyncFilterRunner::runFunction(
    std::function<ImageData()> filter, ImageMetadata metadata) {
    return std::make_shared<const AsyncImageData2D>(filter, metadata);
}

} // namespace megamol::ImageSeries::filter
