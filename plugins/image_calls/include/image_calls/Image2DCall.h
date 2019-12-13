#pragma once

#include <memory>
#include <string>
#include <map>

#include "mmcore/AbstractGetDataCall.h"
#include "mmcore/CallAutoDescription.h"
#include "vislib/graphics/BitmapCodecCollection.h"

#include "image_calls/image_calls.h"

namespace megamol {
namespace image_calls {

class image_calls_API Image2DCall : public megamol::core::AbstractGetDataCall {
public:
    typedef std::map<std::string, vislib::graphics::BitmapImage> ImageMap;

    /**
     * Answer the name of this call.
     *
     * @return The name of this call.
     */
    static const char* ClassName(void) { return "Image2DCall"; }

    /**
     * Answer a human readable description of this call.
     *
     * @return A human readable description of this call.
     */
    static const char* Description(void) { return "New Call to transport 2D image data"; }

    /**
     * Answer the number of functions used for this call.
     *
     * @return The number of functions used for this call.
     */
    static unsigned int FunctionCount(void) { return 1; }

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

    void SetAvailablePathsPtr(const std::shared_ptr<std::vector<std::string>> ptr);

    const std::shared_ptr<std::vector<std::string>> GetAvailablePathsPtr(void) const;

    void SetWishlistPtr(const std::shared_ptr<std::vector<uint64_t>> ptr);

    const std::shared_ptr<std::vector<uint64_t>> GetWishlistPtr(void) const;

    /** Ctor. */
    Image2DCall();

    /** Dtor. */
    virtual ~Image2DCall() = default;

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
