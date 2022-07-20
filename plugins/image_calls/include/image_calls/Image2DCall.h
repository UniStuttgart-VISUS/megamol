#pragma once

#include <memory>

#include "mmcore/CallAutoDescription.h"
#include "mmstd/data/AbstractGetDataCall.h"

namespace megamol {
namespace image_calls {

class Image2DCall : public megamol::core::AbstractGetDataCall {
public:
    /**
     * Answer the name of this call.
     *
     * @return The name of this call.
     */
    static const char* ClassName(void) {
        return "Image2DCall";
    }

    /**
     * Answer a human readable description of this call.
     *
     * @return A human readable description of this call.
     */
    static const char* Description(void) {
        return "Call to transport 2D image data";
    }

    /**
     * Answer the number of functions used for this call.
     *
     * @return The number of functions used for this call.
     */
    static unsigned int FunctionCount(void) {
        return 1;
    }

    /**
     * Answer the name of the function used for this call.
     *
     * @param idx The index of the function to return it's name.
     *
     * @return The name of the requested function.
     */
    static const char* FunctionName(unsigned int idx) {
        switch (idx) {
        case 0:
            return "GetData";
        }
        return nullptr;
    }

    enum Encoding : uint8_t { PNG, BMP, JPEG, SNAPPY, RAW };

    enum Format : uint8_t { RGB, RGBA };

    Image2DCall();

    virtual ~Image2DCall() = default;

    void* GetData() const {
        return this->data_;
    }

    Encoding GetEncoding() const {
        return this->enc_;
    }

    Format GetFormat() const {
        return this->format_;
    }

    size_t GetWidth() const {
        return this->width_;
    }

    size_t GetHeight() const {
        return this->height_;
    }

    size_t GetFilesize() const {
        return this->filesize_;
    }

    void SetData(Encoding const enc, Format const format, size_t width, size_t height, size_t filesize, void* data) {
        this->enc_ = enc;
        this->format_ = format;
        this->width_ = width;
        this->height_ = height;
        this->filesize_ = filesize;
        this->data_ = data;
    }

private:
    size_t width_, height_, filesize_;

    Encoding enc_;

    Format format_;

    void* data_;

}; // end class Image2DCall

typedef megamol::core::CallAutoDescription<Image2DCall> Image2DCallDescription;

} // end namespace image_calls
} // end namespace megamol
