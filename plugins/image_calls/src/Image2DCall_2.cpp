#include "stdafx.h"
#include "image_calls/Image2DCall_2.h"

using namespace megamol;
using namespace megamol::image_calls;

/*
 * Image2DCall_2::Image2DCall_2
 */
Image2DCall_2::Image2DCall_2(void) : core::AbstractGetDataCall(), imagePtr(nullptr) {}

/*
 * Image2DCall_2::GetImageCount
 */
size_t Image2DCall_2::GetImageCount(void) const {
    if (this->imagePtr != nullptr) {
        return this->imagePtr->size();
    }
    return 0;
}

/*
 * Image2DCall_2::SetImagePtr
 */
void Image2DCall_2::SetImagePtr(const std::shared_ptr<Image2DCall_2::ImageVector> ptr) {
    this->imagePtr = ptr;
}

/*
 * Image2DCall_2::GetImageVector
 */
const std::shared_ptr<Image2DCall_2::ImageVector> Image2DCall_2::GetImageVector(void) const {
    return this->imagePtr;
}
