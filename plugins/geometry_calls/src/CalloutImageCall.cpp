#include "geometry_calls/CalloutImageCall.h"
#include "stdafx.h"

namespace megamol::geocalls {

/*
 * CalloutImageCall::CallForGetID
 */
const unsigned int CalloutImageCall::CallForGetID = 0;

/*
 * CalloutImageCall::CallForSetID
 */
const unsigned int CalloutImageCall::CallForSetID = 1;

/*
 * CalloutImageCall::CallForGetImage
 */
const unsigned int CalloutImageCall::CallForGetImage = 2;

/*
 * CalloutImageCall::CallForSetImage
 */
const unsigned int CalloutImageCall::CallForSetImage = 3;

/*
 * CalloutImageCall::CallForGetPointer
 */
const unsigned int CalloutImageCall::CallForGetPointAt = 4;

/*
 * CalloutImageCall::CallForSetPointer
 */
const unsigned int CalloutImageCall::CallForSetPointAt = 5;

/*
 * CalloutImageCall::CallForGetWidth
 */
const unsigned int CalloutImageCall::CallForGetWidth = 6;

/*
 * CalloutImageCall::CallForSetWidth
 */
const unsigned int CalloutImageCall::CallForSetWidth = 7;

/*
 * CalloutImageCall::CallForGetHeight
 */
const unsigned int CalloutImageCall::CallForGetHeight = 8;

/*
 * CalloutImageCall::CallForSetHeight
 */
const unsigned int CalloutImageCall::CallForSetHeight = 9;

int qr_id_value = -1;
int width_value = 0;
int height_value = 0;

CalloutImageCall::CalloutImageCall(void)
        : qr_id(&qr_id_value)
        , qr_image(NULL)
        , qr_pointer(NULL)
        , width(&width_value)
        , height(&height_value) {}


CalloutImageCall::~CalloutImageCall(void) {
    qr_image = NULL;
    qr_pointer = NULL;
}
} // namespace megamol::geocalls
