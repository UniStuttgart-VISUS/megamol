#include "Convolution2DFilter.h"

#include "vislib/graphics/BitmapImage.h"

#include <cmath>
#include <cstring>
#include <vector>

namespace megamol::ImageSeries::filter {

Convolution2DFilter::Convolution2DFilter(Input input) : input(std::move(input)) {}

Convolution2DFilter::ImagePtr Convolution2DFilter::operator()() {
    using Image = AsyncImageData2D::BitmapImage;

    // Wait for image data to be ready
    auto image = input.image ? input.image->getImageData() : nullptr;

    // Empty or too small -> return nothing
    if (!image || image->Width() < 1 || image->Height() < 1) {
        return nullptr;
    }

    // TODO: add compatibility with non-byte images
    if (image->GetChannelCount() != 1 || image->GetChannelType() != Image::ChannelType::CHANNELTYPE_BYTE) {
        return nullptr;
    }

    // Create output image
    auto result = std::make_shared<Image>(image->Width(), image->Height(), 1, Image::ChannelType::CHANNELTYPE_BYTE);

    const auto* dataIn = image->PeekDataAs<std::uint8_t>();
    auto* dataOut = result->PeekDataAs<std::uint8_t>();
    std::size_t width = result->Width();
    std::size_t height = result->Height();

    // Create intermediate storage
    std::vector<float> intermediate(width * height);

    auto clamp = [](int low, int value, int up) { return std::min(std::max(value, low), up); };
    auto getIndex = [=](std::size_t x, std::size_t y) { return x + y * width; };

    int kernelWidth = input.kernelX.size();
    int kernelXOff = kernelWidth / 2;

    // Convolve along X-axis
    for (std::size_t y = 0; y < height; y++) {
        for (std::size_t x = 0; x < width; x++) {
            float value = 0;
            std::size_t yi = clamp(0, y, height - 1);
            for (int i = 0; i < kernelWidth; ++i) {
                value += input.kernelX[i] * dataIn[getIndex(clamp(0, int(x) + kernelXOff - i, width - 1), yi)];
            }
            intermediate[getIndex(x, y)] = value;
        }
    }

    int kernelHeight = input.kernelY.size();
    int kernelYOff = kernelHeight / 2;

    // Convolve along Y-axis
    for (std::size_t y = 0; y < height; y++) {
        for (std::size_t x = 0; x < width; x++) {
            float value = 0;
            std::size_t xi = clamp(0, x, width - 1);
            for (int i = 0; i < kernelHeight; ++i) {
                value += input.kernelY[i] * intermediate[getIndex(xi, clamp(0, int(y) + kernelYOff - i, height - 1))];
            }
            dataOut[getIndex(x, y)] = clamp(0, value, 255);
        }
    }

    return std::const_pointer_cast<const Image>(result);
}

ImageMetadata Convolution2DFilter::getMetadata() const {
    if (input.image) {
        ImageMetadata metadata = input.image->getMetadata();
        metadata.hash = util::computeHash(input.image, input.kernelX, input.kernelY);
        return metadata;
    } else {
        return {};
    }
}

std::vector<float> Convolution2DFilter::makeGaussianKernel(float sigma, std::size_t radius) {
    static const double invSqrtTau = std::sqrt(2.0 * 3.14159265358979323846);

    std::vector<float> kernel(radius * 2 + 1);
    double sum = 0;
    for (std::size_t i = 0; i <= radius; ++i) {
        double sample = (invSqrtTau / sigma) * exp(-(double(i) * i) / (2 * sigma * sigma));
        kernel[radius + i] = sample;
        kernel[radius - i] = sample;
        sum += (i == 0 ? 1 : 2) * sample;
    }

    for (float& entry : kernel) {
        entry /= sum;
    }

    return kernel;
}

} // namespace megamol::ImageSeries::filter
