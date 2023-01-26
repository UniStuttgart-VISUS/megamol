#include "ImageRegistrator.h"

#include "../filter/AsyncFilterRunner.h"
#include "../filter/Convolution2DFilter.h"
#include "../filter/DerivativeFilter.h"
#include "imageseries/util/ImageUtils.h"

namespace megamol::ImageSeries::registration {

ImageRegistrator::ImageRegistrator() : transform(1, 0, 0, 1, 0, 0) {}

void ImageRegistrator::setInputImage(AsyncImagePtr image) {
    if (this->inputImage != image) {
        this->inputImage = image;

        // Blue input image
        filter::Convolution2DFilter::Input blurInput;
        blurInput.image = this->inputImage;
        blurInput.kernelX = filter::Convolution2DFilter::makeGaussianKernel(8, 20);
        blurInput.kernelY = blurInput.kernelX;

        auto blurredImage =
            std::make_shared<AsyncImageData2D>(filter::Convolution2DFilter(blurInput), image->getMetadata());
        this->inputDerivative = filter::DerivativeFilter(blurredImage)();
        this->biasedAverageMeanSquareError = -1.f;
        this->stepsSinceLastImprovement = 0;
    }
}

ImageRegistrator::AsyncImagePtr ImageRegistrator::getInputImage() const {
    return inputImage;
}

void ImageRegistrator::setReferenceImage(AsyncImagePtr image) {
    this->referenceImage = image;
}

ImageRegistrator::AsyncImagePtr ImageRegistrator::getReferenceImage() const {
    return referenceImage;
}

void ImageRegistrator::setTransform(glm::mat3x2 transform) {
    this->transform = transform;
}

const glm::mat3x2& ImageRegistrator::getTransform() const {
    return transform;
}

void ImageRegistrator::setConvergenceRateLinear(float rate) {
    this->convergenceRateLinear = rate;
}

float ImageRegistrator::getConvergenceRateLinear() const {
    return convergenceRateLinear;
}

void ImageRegistrator::setConvergenceRateAffine(float rate) {
    this->convergenceRateAffine = rate;
}

float ImageRegistrator::getConvergenceRateAffine() const {
    return convergenceRateAffine;
}

float ImageRegistrator::getMeanSquareError() const {
    return meanSquareError;
}

void ImageRegistrator::reset() {
    this->biasedAverageMeanSquareError = -1.f;
    this->stepsSinceLastImprovement = 0;
    this->transform = glm::mat3x2(1, 0, 0, 1, 0, 0);
}

int ImageRegistrator::getStepsSinceLastImprovement() const {
    return stepsSinceLastImprovement;
}

void ImageRegistrator::step() {
    auto referenceData = referenceImage ? referenceImage->getImageData() : nullptr;
    auto inputData = inputImage ? inputImage->getImageData() : nullptr;

    if (referenceData == nullptr || inputData == nullptr || inputDerivative == nullptr) {
        return;
    }

    util::ImageSampler<uint8_t, 1> reference(*referenceData);
    util::ImageSampler<uint8_t, 1> input(*inputData);
    util::ImageSampler<uint8_t, 2> derivative(*inputDerivative);

    glm::mat3x2 sum(0.f);
    std::size_t sampleCount = 0;
    double squareErrorSum = 0;

    for (std::size_t y = 0; y < input.getHeight(); ++y) {
        for (std::size_t x = 0; x < input.getWidth(); ++x) {
            auto point = glm::vec2(transform * glm::vec3(x, y, 1.f));

            if (point.x < 0 || point.y < 0 || point.x > reference.getWidth() - 1 ||
                point.y > reference.getHeight() - 1) {
                continue;
            }

            auto referenceSample = reference.get(x, y) / 255.f;
            auto inputSample = input.lerp(point) / 255.f;
            auto derivativeSample = (derivative.lerp(point) - glm::vec2(127.f, 127.f)) / 255.f;

            auto difference = referenceSample - inputSample;
            squareErrorSum += difference.x * difference.x;

            auto offset = difference.x * 2.f * derivativeSample;
            sum += glm::mat3x2(x * offset.x, x * offset.y, y * offset.x, y * offset.y, offset.x, offset.y);
            sampleCount++;
        }
    }

    auto limitVectorLength = [](glm::vec2 vec, float maxLength) {
        float length = glm::length(vec);
        return length > maxLength ? vec / length * maxLength : vec;
    };

    if (sampleCount != 0) {
        meanSquareError = squareErrorSum / sampleCount;
        if (biasedAverageMeanSquareError < 0.f) {
            biasedAverageMeanSquareError = meanSquareError;
        } else if (meanSquareError < biasedAverageMeanSquareError) {
            biasedAverageMeanSquareError = (biasedAverageMeanSquareError + meanSquareError) * 0.5f;
            stepsSinceLastImprovement = 0;
        } else {
            stepsSinceLastImprovement++;
        }
        transform += glm::mat3x2(sum[0] * (convergenceRateLinear * biasedAverageMeanSquareError),
            sum[1] * (convergenceRateLinear * biasedAverageMeanSquareError),
            limitVectorLength(sum[2] * (convergenceRateAffine * biasedAverageMeanSquareError), 2.f));
    }
}


} // namespace megamol::ImageSeries::registration
