#include "imageseries/AsyncImageData2D.h"

#include "vislib/graphics/BitmapImage.h"

namespace megamol::ImageSeries {

AsyncImageData2D::AsyncImageData2D(std::shared_ptr<const BitmapImage> imageData)
        : available(true)
        , byteSize(imageData != nullptr ? imageData->Width() * imageData->Height() : 0)
        , imageData(imageData) {}

AsyncImageData2D::AsyncImageData2D(std::size_t byteSize) : byteSize(byteSize) {}

bool AsyncImageData2D::isWaiting() const {
    return !available;
}

bool AsyncImageData2D::isFinished() const {
    return available;
}

bool AsyncImageData2D::isValid() const {
    return available && imageData;
}

bool AsyncImageData2D::isFailed() const {
    return available && !imageData;
}

std::size_t AsyncImageData2D::getByteSize() const {
    return byteSize;
}

void AsyncImageData2D::setImageData(std::shared_ptr<const BitmapImage> imageData) {
    if (!available) {
        this->imageData = imageData;
        available = true;
    } else {
        // TODO log a warning when trying to overwrite existing async image data
    }
}

std::shared_ptr<const vislib::graphics::BitmapImage> AsyncImageData2D::getImageData() const {
    return available ? imageData : nullptr;
}

} // namespace megamol::ImageSeries