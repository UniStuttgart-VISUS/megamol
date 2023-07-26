/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#include "AsyncImageRegistrator.h"

#include <chrono>
#include <memory>
#include <mutex>
#include <thread>

using namespace std::chrono_literals;

namespace megamol::ImageSeries::registration {

AsyncImageRegistrator::AsyncImageRegistrator()
        : registrator(std::make_shared<ImageRegistrator>())
        , transform(1, 0, 0, 1, 0, 0) {
    registrator->setConvergenceRateLinear(0.00000001f);
    registrator->setConvergenceRateAffine(0.05f);
}

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

float AsyncImageRegistrator::getMeanSquareError() const {
    std::unique_lock<std::mutex> lock(mutex);
    return this->meanSquareError;
}

int AsyncImageRegistrator::getStepsSinceLastImprovement() const {
    std::unique_lock<std::mutex> lock(mutex);
    return this->stepsSinceLastImprovement;
}

void AsyncImageRegistrator::setActive(bool active) {
    if (active && !thread) {
        this->active = true;

        this->registrator->reset();
        this->stepsSinceLastImprovement = 0;

        thread = std::make_unique<std::thread>([this]() {
            auto nextUpdate = std::chrono::high_resolution_clock::now();
            while (this->active) {
                // Update parameters/result if enough time has passed
                auto now = std::chrono::high_resolution_clock::now();
                if (now > nextUpdate) {
                    std::unique_lock<std::mutex> lock(mutex);
                    if (this->resetPending) {}
                    this->registrator->setInputImage(inputImage);
                    this->registrator->setReferenceImage(referenceImage);

                    this->transform = this->registrator->getTransform();
                    this->meanSquareError = this->registrator->getMeanSquareError();
                    this->stepsSinceLastImprovement = this->registrator->getStepsSinceLastImprovement();
                    nextUpdate = now + 200ms;
                }
                registrator->step();
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
