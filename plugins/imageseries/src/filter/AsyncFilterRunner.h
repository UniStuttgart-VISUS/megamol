#pragma once

#include "imageseries/AsyncImageData2D.h"

#include <functional>
#include <memory>

namespace megamol::ImageSeries::filter {

/**
 * Encapsulates image series processing functionality with support for asynchronous execution.
 *
 * Inputs and outputs are provided via async image objects.
 * The filter itself may also perform its work on a separate thread.
 */
template<typename AsyncImageData = AsyncImageData2D<>>
class AsyncFilterRunner {
public:
    AsyncFilterRunner();
    ~AsyncFilterRunner();

    template<typename Filter, typename... Args>
    std::shared_ptr<const AsyncImageData> run(Args&&... args) {
        std::shared_ptr<Filter> filter = std::make_shared<Filter>(std::forward<Args>(args)...);
        return runFunction([filter]() { return (*filter)(); }, filter->getMetadata());
    }

    std::shared_ptr<const AsyncImageData> runFunction(
        std::function<std::shared_ptr<const typename AsyncImageData::BitmapImage>()> filter, ImageMetadata metadata);
};

template<typename AsyncImageData>
inline AsyncFilterRunner<AsyncImageData>::AsyncFilterRunner() {}

template<typename AsyncImageData>
inline AsyncFilterRunner<AsyncImageData>::~AsyncFilterRunner() {}

template<typename AsyncImageData>
inline std::shared_ptr<const AsyncImageData> AsyncFilterRunner<AsyncImageData>::runFunction(
    std::function<std::shared_ptr<const typename AsyncImageData::BitmapImage>()> filter, ImageMetadata metadata) {
    return std::make_shared<const AsyncImageData>(filter, metadata);
}

} // namespace megamol::ImageSeries::filter
