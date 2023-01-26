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
#include "util/PerfTimer.h"
#include "util/WorkerThreadPool.h"

namespace vislib::graphics {
class BitmapImage;
}

namespace megamol::ImageSeries {

namespace util {
class WorkerThreadPool;
class Job;
} // namespace util

struct ImageMetadata {
    enum class Mode {
        Color,
        Grayscale,
        Labels,
    };

    bool valid = false;
    Mode mode = Mode::Color;
    std::uint32_t index = 0;
    std::uint32_t imageCount = 1;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t channels = 1;
    std::uint32_t bytesPerChannel = 1;
    util::Hash hash = 0;
    std::string filename;

    std::size_t getByteSize() const {
        return width * height * channels * bytesPerChannel;
    }
};

class AsyncImageData2D {
public:
    using BitmapImage = vislib::graphics::BitmapImage;
    using Hash = megamol::ImageSeries::util::Hash;
    using ImageProvider = std::function<std::shared_ptr<const BitmapImage>()>;

    AsyncImageData2D() = default;
    AsyncImageData2D(ImageProvider imageProvider, ImageMetadata metadata);
    ~AsyncImageData2D();

    bool isWaiting() const;
    bool isFinished() const;

    bool isValid() const;
    bool isFailed() const;

    const ImageMetadata& getMetadata() const;
    std::size_t getByteSize() const;
    Hash getHash() const;

    std::shared_ptr<const BitmapImage> tryGetImageData() const;
    std::shared_ptr<const BitmapImage> getImageData() const;

private:
    static util::WorkerThreadPool& getThreadPool();

    Hash computeHash(std::shared_ptr<const BitmapImage> imageData);

    ImageMetadata metadata;
    std::shared_ptr<const BitmapImage> imageData;

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
