#ifndef INCLUDE_IMAGESERIES_IMAGESERIES2D_H_
#define INCLUDE_IMAGESERIES_IMAGESERIES2D_H_

#include <memory>

#include "imageseries/AsyncImageData2D.h"
#include "mmstd/data/AbstractGetDataCall.h"
#include "mmcore/factories/CallAutoDescription.h"

namespace megamol::ImageSeries {

class ImageSeries2DCall : public megamol::core::AbstractGetDataCall {

public:
    using CallDescription = typename megamol::core::factories::CallAutoDescription<ImageSeries2DCall>;

    /** Function index for obtaining data for an image in the series. **/
    static const unsigned int CallGetData = 0;

    /** Function index for obtaining metadata for the whole series. **/
    static const unsigned int CallGetMetaData = 1;

    /**
     * Answer the name of this call.
     *
     * @return The name of this call.
     */
    static const char* ClassName(void) {
        return "ImageSeries2DCall";
    }

    /**
     * Answer a human readable description of this call.
     *
     * @return A human readable description of this call.
     */
    static const char* Description(void) {
        return "Call to transport 2D image series data";
    }

    /**
     * Answer the number of functions used for this call.
     *
     * @return The number of functions used for this call.
     */
    static unsigned int FunctionCount(void) {
        return 2;
    }

    /**
     * Answer the name of the function used for this call.
     *
     * @param idx The index of the function to return the name of.
     *
     * @return The name of the requested function, or a nullpointer for out-of-bounds requests.
     */
    static const char* FunctionName(unsigned int idx) {
        switch (idx) {
        case CallGetData:
            return "GetData";
        case CallGetMetaData:
            return "GetMetaData";
        }
        return nullptr;
    }

    ImageSeries2DCall() = default;
    ~ImageSeries2DCall() override = default;

    struct Input {
        // Timestamp for which an image should be requested from the series
        double time = 0;
    };

    struct Output {
        // Total number of discrete images across the series. 0 for empty series.
        std::uint32_t imageCount = 0;

        // imageCount / (maximumTime - minimumTime). 0 for empty series.
        double framerate = 0;

        // Minimum inclusive timestamp that may be requested. 0 for empty series.
        double minimumTime = 0;

        // Maximum inclusive timestamp that may be requested. 0 for empty series.
        double maximumTime = 0;

        // Horizontal pixel resolution of all images in the series. 0 for empty series.
        std::uint32_t width = 0;

        // Vertical pixel resolution of all images in the series. 0 for empty series.
        std::uint32_t height = 0;

        // Number of bytes per pixel. 0 for empty series or if unknown.
        std::uint8_t bytesPerPixel = 0;

        // Effective time of the image that was returned (may be clamped or quantized).
        double resultTime = 0;

        // GetData: Index of the current image within the series.
        std::uint32_t imageIndex = 0;

        // GetData: Name of the current image file.
        std::string filename;

        // GetData: Holds the last requested image's pixels, once available.
        std::shared_ptr<const AsyncImageData2D<>> imageData;

        // GetData: Convenience function to get the hash without needing a null check for the image data.
        AsyncImageData2D<>::Hash getHash() const {
            return imageData ? imageData->getHash() : 0;
        }
    };

    const Input& GetInput() const {
        return input;
    }

    void SetInput(Input input) {
        this->input = std::move(input);
    }

    const Output& GetOutput() const {
        return output;
    }

    void SetOutput(Output output) {
        this->output = std::move(output);
    }

private:
    Input input;
    Output output;
};

} // namespace megamol::ImageSeries

#endif
