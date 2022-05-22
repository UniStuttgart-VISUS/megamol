#include "AsyncFilterRunner.h"

namespace megamol::ImageSeries::filter {

AsyncFilterRunner::AsyncFilterRunner() {}

AsyncFilterRunner::~AsyncFilterRunner() {}

AsyncFilterRunner::AsyncImageData AsyncFilterRunner::runFunction(
    std::function<ImageData()> filter, std::size_t byteSize, AsyncImageData2D::Hash hash) {
    return std::make_shared<const AsyncImageData2D>(filter, byteSize, hash);
}

} // namespace megamol::ImageSeries::filter
