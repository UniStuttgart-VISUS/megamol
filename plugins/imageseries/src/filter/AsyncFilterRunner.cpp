#include "AsyncFilterRunner.h"

namespace megamol::ImageSeries::filter {

AsyncFilterRunner::AsyncFilterRunner() {}

AsyncFilterRunner::~AsyncFilterRunner() {}

AsyncFilterRunner::AsyncImageData AsyncFilterRunner::runFunction(
    std::function<ImageData()> filter, std::size_t byteSize) {
    return std::make_shared<const AsyncImageData2D>(filter, byteSize);
}

} // namespace megamol::ImageSeries::filter