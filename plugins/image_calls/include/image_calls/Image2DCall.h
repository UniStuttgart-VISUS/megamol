#pragma once

#include <memory>
#include <string>
#include <map>

#include "mmcore/AbstractGetDataCall.h"
#include "mmcore/CallAutoDescription.h"
#include "mmcore/utility/graphics/BitmapCodecCollection.h"

namespace megamol {
namespace image_calls {

class Image2DCall : public megamol::core::AbstractGetDataCall {
public:
    typedef std::map<std::string, vislib::graphics::BitmapImage> ImageMap;

    /** Index of the GetData function */
    static const uint32_t CallForGetData;

    /** Index of the GetMetaData function */
    static const uint32_t CallForGetMetaData;

    /** Index of the SetWishlist function */
    static const uint32_t CallForSetWishlist;

    /** Index of the WaitForData function */
    static const uint32_t CallForWaitForData;

    /** Index of the DeleteData function */
    static const uint32_t CallForDeleteData;

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
        case 1:
            return "GetMetaData";
        case 2:
            return "SetWishlist";
        case 3:
            return "WaitForData";
        case 4:
            return "DeleteData";
        }
        return nullptr;
    }

    /**
     * Answer the count of stored images
     *
     * @return The number of stored images
     */
    size_t GetImageCount(void) const;

    /**
     * Sets the data pointer
     *
     * @param ptr Pointer to the vector storing the images
     */
    void SetImagePtr(const std::shared_ptr<ImageMap> ptr);

    /**
     * Returns the currently stored image vector
     *
     * @return Pointer to the vector storing the images
     */
    const std::shared_ptr<ImageMap> GetImagePtr(void) const;

    /**
     * Set the pointer to the field containing all paths to the available files.
     *
     * @param ptr Pointer to the data vector
     */
    void SetAvailablePathsPtr(const std::shared_ptr<std::vector<std::string>> ptr);

    /**
     * Returns the pointer to the vector containing all available image file paths.
     *
     * @return All available image file paths.
     */
    const std::shared_ptr<std::vector<std::string>> GetAvailablePathsPtr(void) const;

    /**
     * Sets the pointer to the wishlist, containing all indices of the desired images
     * When this is set to nullptr, all images are set as wished for.
     * 
     * @param ptr Pointer to the vector containing all desired image indices
     */
    void SetWishlistPtr(const std::shared_ptr<std::vector<uint64_t>> ptr);

    /**
     * Returns the pointer to the wishlist
     *
     * @return Pointer to the wishlist
     */
    const std::shared_ptr<std::vector<uint64_t>> GetWishlistPtr(void) const;

    /** Ctor. */
    Image2DCall();

    /** Dtor. */
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
    /** Pointer to the stored data */
    std::shared_ptr<ImageMap> imagePtr;

    /** Pointer to the list storing the paths to all available (but not necessarily loaded) images */
    std::shared_ptr<std::vector<std::string>> availablePathsPtr;

    /** Pointer to the list storing the indices of all wished figures */
    std::shared_ptr<std::vector<uint64_t>> wishlistPtr;
};

typedef megamol::core::CallAutoDescription<Image2DCall> Image2DCallDescription;

} // namespace image_calls
} // namespace megamol
