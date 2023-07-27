/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "util/ImageUtils.h"
#include "util/PerfTimer.h"
#include "util/WorkerThreadPool.h"

#include <functional>
#include <memory>
#include <string>

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

template<class BitmapImageT = vislib::graphics::BitmapImage>
class AsyncImageData2D {
public:
    using BitmapImage = BitmapImageT;
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


template<class BitmapImageT>
inline AsyncImageData2D<BitmapImageT>::AsyncImageData2D(ImageProvider imageProvider, ImageMetadata metadata)
        : metadata(metadata) {
    job = getThreadPool().submit([this, imageProvider]() { imageData = imageProvider(); });
}

template<class BitmapImageT>
inline AsyncImageData2D<BitmapImageT>::~AsyncImageData2D() {
    // Try to cancel job
    if (!job.cancel()) {
        // If not possible, wait for its completion
        job.await();
    }
}

template<class BitmapImageT>
inline bool AsyncImageData2D<BitmapImageT>::isWaiting() const {
    return job.isPending();
}

template<class BitmapImageT>
inline bool AsyncImageData2D<BitmapImageT>::isFinished() const {
    return !job.isPending();
}

template<class BitmapImageT>
inline bool AsyncImageData2D<BitmapImageT>::isValid() const {
    return isFinished() && imageData;
}

template<class BitmapImageT>
inline bool AsyncImageData2D<BitmapImageT>::isFailed() const {
    return isFinished() && !imageData;
}

template<class BitmapImageT>
inline const ImageMetadata& AsyncImageData2D<BitmapImageT>::getMetadata() const {
    return metadata;
}

template<class BitmapImageT>
inline std::size_t AsyncImageData2D<BitmapImageT>::getByteSize() const {
    return metadata.getByteSize();
}

template<class BitmapImageT>
inline typename AsyncImageData2D<BitmapImageT>::Hash AsyncImageData2D<BitmapImageT>::getHash() const {
    return metadata.hash;
}

template<class BitmapImageT>
inline typename AsyncImageData2D<BitmapImageT>::Hash AsyncImageData2D<BitmapImageT>::computeHash(
    std::shared_ptr<const BitmapImageT> imageData) {
    return std::hash<BitmapImageT>()();
}

template<class BitmapImageT>
inline std::shared_ptr<const BitmapImageT> AsyncImageData2D<BitmapImageT>::tryGetImageData() const {
    return isFinished() ? imageData : nullptr;
}

template<class BitmapImageT>
inline std::shared_ptr<const BitmapImageT> AsyncImageData2D<BitmapImageT>::getImageData() const {
    job.execute();
    return imageData;
}

template<class BitmapImageT>
inline util::WorkerThreadPool& AsyncImageData2D<BitmapImageT>::getThreadPool() {
    return util::WorkerThreadPool::getSharedInstance();
}


} // namespace megamol::ImageSeries

namespace std {
template<>
struct hash<std::shared_ptr<const megamol::ImageSeries::AsyncImageData2D<>>> {
    std::size_t operator()(const std::shared_ptr<const megamol::ImageSeries::AsyncImageData2D<>>& data) const {
        return data ? data->getHash() : 0;
    }
};

template<>
struct hash<std::shared_ptr<const vislib::graphics::BitmapImage>> {
    std::size_t operator()(std::shared_ptr<const vislib::graphics::BitmapImage> imageData) const {
        return imageData ? megamol::ImageSeries::util::hashBytes(imageData->PeekData(),
                               imageData->BytesPerPixel() * imageData->Width() * imageData->Height())
                         : 0;
    }
};

} // namespace std
