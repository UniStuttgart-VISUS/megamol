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
class AsyncFilterRunner {
public:
    using AsyncImageData = std::shared_ptr<const AsyncImageData2D>;
    using ImageData = std::shared_ptr<const AsyncImageData2D::BitmapImage>;

    AsyncFilterRunner();
    ~AsyncFilterRunner();

    template<typename Filter, typename... Args>
    AsyncImageData run(Args&&... args) {
        std::shared_ptr<Filter> filter = std::make_shared<Filter>(std::forward<Args>(args)...);
        return runFunction([filter]() { return (*filter)(); }, filter->getMetadata());
    }

    AsyncImageData runFunction(std::function<ImageData()> filter, ImageMetadata metadata);
};

} // namespace megamol::ImageSeries::filter
