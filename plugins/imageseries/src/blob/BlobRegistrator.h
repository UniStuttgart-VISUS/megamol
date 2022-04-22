#ifndef SRC_IMAGESERIES_BLOB_BLOBREGISTRATOR_HPP_
#define SRC_IMAGESERIES_BLOB_BLOBREGISTRATOR_HPP_

#include "imageseries/AsyncImageData2D.h"

#include "../filter/BlobLabelFilter.h"

#include <memory>

namespace megamol::ImageSeries::blob {

class BlobRegistrator {
public:
    using AsyncImagePtr = std::shared_ptr<const AsyncImageData2D>;
    using ImagePtr = std::shared_ptr<const AsyncImageData2D::BitmapImage>;
    using Label = filter::BlobLabelFilter::Label;

    struct Input {
        AsyncImagePtr image;
        AsyncImagePtr predecessor;
        bool flowFrontMode = false;
    };

    struct Output {
        struct Link {
            Label source;
            Label dest;
        };

        std::vector<Link> links;
    };

    Output apply(Input input);
};

} // namespace megamol::ImageSeries::blob


#endif
