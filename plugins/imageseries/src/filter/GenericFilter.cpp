#include "GenericFilter.h"

#include "vislib/graphics/BitmapImage.h"

#include <limits>

namespace megamol::ImageSeries::filter {

namespace detail {

template<typename T, typename P, int channels>
struct GenericFilterImplementation {
    const T* in1 = nullptr;
    const T* in2 = nullptr;
    T* out = nullptr;
    std::size_t width = 0;
    std::size_t height = 0;
    std::size_t size = 0;

    static constexpr T Tmin = std::numeric_limits<T>::min();
    static constexpr T Tmax = std::numeric_limits<T>::max();

    GenericFilterImplementation(const AsyncImageData2D::BitmapImage& input1,
        const AsyncImageData2D::BitmapImage& input2, AsyncImageData2D::BitmapImage& output) {
        this->in1 = input1.PeekDataAs<T>();
        this->in2 = input2.PeekDataAs<T>();
        this->out = output.PeekDataAs<T>();
        this->width = input1.Width();
        this->height = input1.Height();
        this->size = width * height;
    }

    void run(GenericFilter::Operation operation) {
        switch (operation) {
        case GenericFilter::Operation::Difference:
            return difference();
        default:
            return;
        }
    };

    void difference() {
        for (std::size_t i = 0; i < size; ++i) {
            for (std::size_t c = 0; c < channels; ++c) {
                P value1 = in1[i * channels + c];
                P value2 = in2[i * channels + c];
                out[i * channels + c] = static_cast<T>(std::max<P>(std::min<P>(value1 - value2, Tmax), Tmin));
            }
        }
    };
};

template<typename T, typename P>
struct GenericFilterRunner {
    static void run(const AsyncImageData2D::BitmapImage& input1, const AsyncImageData2D::BitmapImage& input2,
        AsyncImageData2D::BitmapImage& output, GenericFilter::Operation operation) {
        switch (input1.GetChannelCount()) {
        case 1:
            return GenericFilterImplementation<T, P, 1>(input1, input2, output).run(operation);
        case 2:
            return GenericFilterImplementation<T, P, 2>(input1, input2, output).run(operation);
        case 3:
            return GenericFilterImplementation<T, P, 3>(input1, input2, output).run(operation);
        case 4:
            return GenericFilterImplementation<T, P, 4>(input1, input2, output).run(operation);
        default:
            break;
        }
    }
};

} // namespace detail

GenericFilter::GenericFilter(Input input) : input(std::move(input)) {}

GenericFilter::GenericFilter(AsyncImagePtr image1, AsyncImagePtr image2, Operation op) {
    input.image1 = image1;
    input.image2 = image2;
}

GenericFilter::ImagePtr GenericFilter::operator()() {
    using Image = AsyncImageData2D::BitmapImage;

    // Wait for image data to be ready
    auto img1 = input.image1 ? input.image1->getImageData() : nullptr;
    auto img2 = input.image2 ? input.image2->getImageData() : nullptr;

    // Empty -> return nothing
    if (!img1 || !img2) {
        return nullptr;
    }

    // Size must match
    if (img1->Width() != img2->Width() || img1->Height() != img2->Height()) {
        return nullptr;
    }

    // Channels must match
    if (img1->GetChannelCount() != img2->GetChannelCount() || img1->GetChannelType() != img2->GetChannelType()) {
        return nullptr;
    }

    // Create output image
    auto result =
        std::make_shared<Image>(img1->Width(), img1->Height(), img1->GetChannelCount(), img1->GetChannelType());

    switch (img1->GetChannelType()) {
    case Image::ChannelType::CHANNELTYPE_BYTE:
        detail::GenericFilterRunner<std::uint8_t, std::int32_t>::run(*img1, *img2, *result, input.operation);
        break;
    case Image::ChannelType::CHANNELTYPE_WORD:
        detail::GenericFilterRunner<std::int16_t, std::int32_t>::run(*img1, *img2, *result, input.operation);
        break;
    case Image::ChannelType::CHANNELTYPE_FLOAT:
        detail::GenericFilterRunner<float, double>::run(*img1, *img2, *result, input.operation);
        break;
    default:
        break;
    }

    return std::const_pointer_cast<const Image>(result);
}

ImageMetadata GenericFilter::getMetadata() const {
    if (input.image1) {
        ImageMetadata metadata = input.image1->getMetadata();
        metadata.hash = util::computeHash(input.image1, input.image2, (int)input.operation);
        return metadata;
    } else {
        return {};
    }
}

} // namespace megamol::ImageSeries::filter
