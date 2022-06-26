#ifndef SRC_IMAGESERIES_REGISTRATION_IMAGEREGISTRATOR_HPP_
#define SRC_IMAGESERIES_REGISTRATION_IMAGEREGISTRATOR_HPP_

#include "imageseries/AsyncImageData2D.h"

#include "glm/mat3x2.hpp"

#include <memory>

namespace megamol::ImageSeries::registration {

class ImageRegistrator {
public:
    using AsyncImagePtr = std::shared_ptr<const AsyncImageData2D>;
    using ImagePtr = std::shared_ptr<const AsyncImageData2D::BitmapImage>;

    ImageRegistrator();

    ImageRegistrator(const ImageRegistrator& registrator) = default;
    ImageRegistrator& operator=(const ImageRegistrator& registrator) = default;

    void setInputImage(AsyncImagePtr image);
    AsyncImagePtr getInputImage() const;

    void setReferenceImage(AsyncImagePtr image);
    AsyncImagePtr getReferenceImage() const;

    void setTransform(glm::mat3x2 transform);
    const glm::mat3x2& getTransform() const;

    void setConvergenceRateLinear(float rate);
    float getConvergenceRateLinear() const;

    void setConvergenceRateAffine(float rate);
    float getConvergenceRateAffine() const;

    float getMeanSquareError() const;

    void reset();
    int getStepsSinceLastImprovement() const;

    void step();

private:
    AsyncImagePtr inputImage;
    ImagePtr inputDerivative;
    AsyncImagePtr referenceImage;
    glm::mat3x2 transform;
    float convergenceRateLinear = 0.0001f;
    float convergenceRateAffine = 100.f;
    float meanSquareError = 0.f;
    float biasedAverageMeanSquareError = -1.f;
    int stepsSinceLastImprovement = 0;
};

} // namespace megamol::ImageSeries::registration


#endif
