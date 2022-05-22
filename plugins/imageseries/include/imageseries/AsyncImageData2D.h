#ifndef INCLUDE_IMAGESERIES_ASYNCIMAGEDATA2D_H_
#define INCLUDE_IMAGESERIES_ASYNCIMAGEDATA2D_H_

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "util/ImageUtils.h"
#include "util/WorkerThreadPool.h"

namespace vislib::graphics {
class BitmapImage;
}

namespace megamol::ImageSeries {

namespace util {
class WorkerThreadPool;
class Job;
} // namespace util

class AsyncImageData2D {
public:
    using BitmapImage = vislib::graphics::BitmapImage;
    using Hash = megamol::ImageSeries::util::Hash;
    using ImageProvider = std::function<std::shared_ptr<const BitmapImage>()>;

    AsyncImageData2D(ImageProvider imageProvider, std::size_t byteSize, Hash hash);
    AsyncImageData2D(std::shared_ptr<const BitmapImage> imageData = nullptr);
    ~AsyncImageData2D();

    bool isWaiting() const;
    bool isFinished() const;

    bool isValid() const;
    bool isFailed() const;

    std::size_t getByteSize() const;

    Hash getHash() const;

    std::shared_ptr<const BitmapImage> tryGetImageData() const;
    std::shared_ptr<const BitmapImage> getImageData() const;

private:
    static util::WorkerThreadPool& getThreadPool();

    Hash computeHash(std::shared_ptr<const BitmapImage> imageData);

    std::size_t byteSize = 0;
    std::shared_ptr<const BitmapImage> imageData;
    Hash hash = 0;

    mutable util::Job job;
};

} // namespace megamol::ImageSeries

namespace std {
template<>
struct hash<std::shared_ptr<const megamol::ImageSeries::AsyncImageData2D>> {
    std::size_t operator()(const std::shared_ptr<const megamol::ImageSeries::AsyncImageData2D>& data) const {
        return data ? data->getHash() : 0;
    }
};

} // namespace std


#endif
