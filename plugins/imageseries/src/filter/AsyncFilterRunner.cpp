#include "AsyncFilterRunner.h"

namespace megamol::ImageSeries::filter {

AsyncFilterRunner::AsyncFilterRunner() {}

AsyncFilterRunner::~AsyncFilterRunner() {}

AsyncFilterRunner::AsyncImageData AsyncFilterRunner::runFunction(std::function<ImageData()> filter) {
    // TODO run asynchronously via threads
    return std::make_shared<const AsyncImageData2D>(filter());
}

} // namespace megamol::ImageSeries::filter