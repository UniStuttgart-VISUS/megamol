#include "image_calls/Image2DCall.h"

using namespace megamol;
using namespace megamol::image_calls;

/*
 * Image2DCall::CallForGetData
 */
const uint32_t Image2DCall::CallForGetData = 0;

/*
 * Image2DCall::CallForGetMetaData
 */
const uint32_t Image2DCall::CallForGetMetaData = 1;

/*
 * Image2DCall::CallForSetWishlist
 */
const uint32_t Image2DCall::CallForSetWishlist = 2;

/*
 * Image2DCall::CallForWaitForData
 */
const uint32_t Image2DCall::CallForWaitForData = 3;

/*
 * Image2DCall::CallForDeleteData
 */
const uint32_t Image2DCall::CallForDeleteData = 4;

/*
 * Image2DCall::Image2DCall
 */
Image2DCall::Image2DCall(void)
        : core::AbstractGetDataCall()
        , availablePathsPtr(nullptr)
        , imagePtr(nullptr)
        , wishlistPtr(nullptr) {}

/*
 * Image2DCall::GetImageCount
 */
size_t Image2DCall::GetImageCount(void) const {
    if (this->imagePtr != nullptr) {
        return this->imagePtr->size();
    }
    return 0;
}

size_t Image2DCall::GetAvailablePathsCount() const {
    if (this->availablePathsPtr != nullptr) {
        return this->availablePathsPtr->size();
    }
}


/*
 * Image2DCall::SetImagePtr
 */
void Image2DCall::SetImagePtr(const std::shared_ptr<Image2DCall::ImageMap> ptr) {
    this->imagePtr = ptr;
}

/*
 * Image2DCall::GetImagePtr
 */
const std::shared_ptr<Image2DCall::ImageMap> Image2DCall::GetImagePtr(void) const {
    return this->imagePtr;
}

/*
 * Image2DCall::SetAvailablePathsPtr
 */
void Image2DCall::SetAvailablePathsPtr(const std::shared_ptr<std::vector<std::string>> ptr) {
    this->availablePathsPtr = ptr;
}

/*
 * Image2DCall::GetAvailablePathsPtr
 */
const std::shared_ptr<std::vector<std::string>> Image2DCall::GetAvailablePathsPtr(void) const {
    return this->availablePathsPtr;
}

/*
 * Image2DCall::SetWishlistPtr
 */
void Image2DCall::SetWishlistPtr(const std::shared_ptr<std::vector<uint64_t>> ptr) {
    this->wishlistPtr = ptr;
}

/*
 * Image2DCall::GetWishlistPtr
 */
const std::shared_ptr<std::vector<uint64_t>> Image2DCall::GetWishlistPtr(void) const {
    return this->wishlistPtr;
}
