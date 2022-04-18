#include "imageseries/AsyncImageData2D.h"

#include "vislib/graphics/BitmapImage.h"

namespace megamol::ImageSeries {

AsyncImageData2D::AsyncImageData2D(ImageProvider imageProvider, std::size_t byteSize)
        : byteSize(byteSize)
        , hash(computeHash()) {
    job = getThreadPool().submit([this, imageProvider]() { imageData = imageProvider(); });
}

AsyncImageData2D::AsyncImageData2D(std::shared_ptr<const BitmapImage> imageData)
        : byteSize(imageData != nullptr ? imageData->Width() * imageData->Height() * imageData->BytesPerPixel() : 0)
        , imageData(imageData)
        , hash(imageData != nullptr ? computeHash() : 0) {}

AsyncImageData2D::~AsyncImageData2D() {
    // Try to cancel job
    if (!job.cancel()) {
        // If not possible, wait for its completion
        job.await();
    }
}

bool AsyncImageData2D::isWaiting() const {
    return job.isPending();
}

bool AsyncImageData2D::isFinished() const {
    return !job.isPending();
}

bool AsyncImageData2D::isValid() const {
    return isFinished() && imageData;
}

bool AsyncImageData2D::isFailed() const {
    return isFinished() && !imageData;
}

std::size_t AsyncImageData2D::getByteSize() const {
    return byteSize;
}

AsyncImageData2D::Hash AsyncImageData2D::getHash() const {
    return hash;
}

AsyncImageData2D::Hash AsyncImageData2D::computeHash() {
    // TODO use an actual hash function instead of a counter!
    static std::atomic<Hash> currentHash = ATOMIC_VAR_INIT(1);
    return currentHash++;
}

std::shared_ptr<const vislib::graphics::BitmapImage> AsyncImageData2D::tryGetImageData() const {
    return isFinished() ? imageData : nullptr;
}

std::shared_ptr<const vislib::graphics::BitmapImage> AsyncImageData2D::getImageData() const {
    job.execute();
    return imageData;
}

util::WorkerThreadPool& AsyncImageData2D::getThreadPool() {
    return util::WorkerThreadPool::getSharedInstance();
}

} // namespace megamol::ImageSeries
