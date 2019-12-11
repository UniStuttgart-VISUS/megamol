#pragma once

#include <memory>
#include <string>

#include "mmcore/AbstractGetDataCall.h"
#include "mmcore/CallAutoDescription.h"
#include "vislib/graphics/BitmapCodecCollection.h"

#include "image_calls/image_calls.h"

namespace megamol {
namespace image_calls {

class image_calls_API Image2DCall_2 : public megamol::core::AbstractGetDataCall {
public:
    typedef std::vector<std::pair<vislib::graphics::BitmapImage, std::string>> ImageVector;

    /**
     * Answer the name of this call.
     *
     * @return The name of this call.
     */
    static const char* ClassName(void) { return "Image2DCall_2"; }

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
    void SetImagePtr(const std::shared_ptr<ImageVector> ptr);

    /**
     * Returns the currently stored image vector
     *
     * @return Pointer to the vector storing the images
     */
    const std::shared_ptr<ImageVector> GetImageVector(void) const;

    /** Ctor. */
    Image2DCall_2();

    /** Dtor. */
    virtual ~Image2DCall_2() = default;

private:
    /** Pointer to the stored data */
    std::shared_ptr<ImageVector> imagePtr;
};

typedef megamol::core::CallAutoDescription<Image2DCall_2> Image2DCall_2Description;

} // namespace image_calls
} // namespace megamol
