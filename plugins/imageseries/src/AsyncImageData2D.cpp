#include "imageseries/AsyncImageData2D.h"

#include "vislib/graphics/BitmapImage.h"

namespace megamol::ImageSeries {

AsyncImageData2D::AsyncImageData2D(std::shared_ptr<const BitmapImage> imageData)
        : available(true)
        , byteSize(imageData != nullptr ? imageData->Width() * imageData->Height() : 0)
        , imageData(imageData)
        , hash(computeHash()) {}

AsyncImageData2D::AsyncImageData2D(std::size_t byteSize) : byteSize(byteSize), hash(computeHash()) {}

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

AsyncImageData2D::Hash AsyncImageData2D::getHash() const {
    return imageData ? hash : 0;
}

AsyncImageData2D::Hash AsyncImageData2D::computeHash() {
    // TODO use an actual hash function instead of a counter!
    static std::atomic<Hash> currentHash = ATOMIC_VAR_INIT(1);
    return currentHash++;
}

void AsyncImageData2D::setImageData(std::shared_ptr<const BitmapImage> imageData) {
    if (!available) {
        this->imageData = imageData;
        available = true;
        conditionVariable.notify_all();
    } else {
        // TODO log a warning when trying to overwrite existing async image data
    }
}

std::shared_ptr<const vislib::graphics::BitmapImage> AsyncImageData2D::tryGetImageData() const {
    return available ? imageData : nullptr;
}

std::shared_ptr<const vislib::graphics::BitmapImage> AsyncImageData2D::getImageData() const {
    if (!available) {
        // TODO once C++20 is available, use std::atomic::wait() instead
        std::unique_lock<std::mutex> lock(mutex);
        conditionVariable.wait(lock, [this]() { return isFinished(); });
    }
    return imageData;
}

} // namespace megamol::ImageSeries