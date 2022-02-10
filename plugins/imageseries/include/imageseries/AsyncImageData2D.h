#ifndef INCLUDE_IMAGESERIES_ASYNCIMAGEDATA2D_H_
#define INCLUDE_IMAGESERIES_ASYNCIMAGEDATA2D_H_

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <vector>

namespace vislib::graphics {
class BitmapImage;
}

namespace megamol::ImageSeries {

class AsyncImageData2D {
public:
    using BitmapImage = vislib::graphics::BitmapImage;

    AsyncImageData2D(std::shared_ptr<const BitmapImage> imageData);
    AsyncImageData2D(std::size_t byteSize);

    bool isWaiting() const;
    bool isFinished() const;

    bool isValid() const;
    bool isFailed() const;

    std::size_t getByteSize() const;

    void setImageData(std::shared_ptr<const BitmapImage> imageData);
    std::shared_ptr<const BitmapImage> tryGetImageData() const;
    std::shared_ptr<const BitmapImage> getImageData() const;

private:
    std::atomic_bool available = ATOMIC_VAR_INIT(false);
    std::size_t byteSize = 0;
    std::shared_ptr<const BitmapImage> imageData;

    // TODO once C++20 is available, use std::atomic::wait() instead
    mutable std::mutex mutex;
    mutable std::condition_variable conditionVariable;
};


} // namespace megamol::ImageSeries

#endif
