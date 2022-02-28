#include "AsyncImageRegistrator.h"

namespace megamol::ImageSeries::registration {

AsyncImageRegistrator::AsyncImageRegistrator()
        : registrator(std::make_shared<ImageRegistrator>())
        , transform(1, 0, 0, 1, 0, 0) {}

AsyncImageRegistrator::~AsyncImageRegistrator() {
    setActive(false);
}

void AsyncImageRegistrator::setInputImage(AsyncImagePtr image) {
    std::unique_lock<std::mutex> lock(mutex);
    this->inputImage = image;
}

AsyncImageRegistrator::AsyncImagePtr AsyncImageRegistrator::getInputImage() const {
    std::unique_lock<std::mutex> lock(mutex);
    return inputImage;
}

void AsyncImageRegistrator::setReferenceImage(AsyncImagePtr image) {
    std::unique_lock<std::mutex> lock(mutex);
    this->referenceImage = image;
}

AsyncImageRegistrator::AsyncImagePtr AsyncImageRegistrator::getReferenceImage() const {
    std::unique_lock<std::mutex> lock(mutex);
    return this->referenceImage;
}

const glm::mat3x2& AsyncImageRegistrator::getTransform() const {
    std::unique_lock<std::mutex> lock(mutex);
    return this->transform;
}

void AsyncImageRegistrator::setActive(bool active) {
    if (active && !thread) {
        this->active = true;
        thread = std::make_unique<std::thread>([this]() {
            while (this->active) {
                {
                    // Update parameters
                    std::unique_lock<std::mutex> lock(mutex);
                    this->registrator->setInputImage(inputImage);
                    this->registrator->setReferenceImage(referenceImage);

                    // Update result
                    this->transform = this->registrator->getTransform();
                }
                registrator->step();
                // TODO auto-stop once conditions are met
            }

            // Push final update
            std::unique_lock<std::mutex> lock(mutex);
            this->transform = this->registrator->getTransform();
        });
    } else if (!active && thread) {
        // Wait for thread to finish
        this->active = false;
        thread->join();
        thread = nullptr;
    }
}

bool AsyncImageRegistrator::isActive() const {
    return thread != nullptr;
}


} // namespace megamol::ImageSeries::registration
