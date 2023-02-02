#pragma once

#include "ImageRegistrator.h"

#include <atomic>
#include <mutex>
#include <thread>

namespace megamol::ImageSeries::registration {

class AsyncImageRegistrator {
public:
    using AsyncImagePtr = std::shared_ptr<const AsyncImageData2D<>>;

    AsyncImageRegistrator();
    ~AsyncImageRegistrator();

    void setInputImage(AsyncImagePtr image);
    AsyncImagePtr getInputImage() const;

    void setReferenceImage(AsyncImagePtr image);
    AsyncImagePtr getReferenceImage() const;

    const glm::mat3x2& getTransform() const;

    float getMeanSquareError() const;

    int getStepsSinceLastImprovement() const;

    void setActive(bool active);
    bool isActive() const;

private:
    std::shared_ptr<ImageRegistrator> registrator;

    AsyncImagePtr inputImage;
    AsyncImagePtr referenceImage;
    glm::mat3x2 transform;
    float meanSquareError = 0.f;
    int stepsSinceLastImprovement = 0;
    bool resetPending = false;

    std::atomic_bool active = ATOMIC_VAR_INIT(false);
    std::unique_ptr<std::thread> thread;
    mutable std::mutex mutex;
};

} // namespace megamol::ImageSeries::registration
