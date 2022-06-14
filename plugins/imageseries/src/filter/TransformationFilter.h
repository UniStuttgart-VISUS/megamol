#ifndef SRC_IMAGESERIES_FILTER_TRANSFORMATIONFILTER_HPP_
#define SRC_IMAGESERIES_FILTER_TRANSFORMATIONFILTER_HPP_

#include "imageseries/AsyncImageData2D.h"

#include "glm/mat3x2.hpp"

#include <memory>

namespace megamol::ImageSeries::filter {

class TransformationFilter {
public:
    using AsyncImagePtr = std::shared_ptr<const AsyncImageData2D>;
    using ImagePtr = std::shared_ptr<const AsyncImageData2D::BitmapImage>;

    struct Input {
        AsyncImagePtr image;
        glm::mat3x2 transform;
    };

    TransformationFilter(Input input);
    TransformationFilter(AsyncImagePtr image, glm::mat3x2 transform);
    ImagePtr operator()();

    ImageMetadata getMetadata() const;

private:
    Input input;
};

} // namespace megamol::ImageSeries::filter


#endif
