#include "geometry_calls/QRCodeDataCall.h"

namespace megamol::geocalls {

/*
 * QRCodeDataCall::CallForGetText
 */
const unsigned int QRCodeDataCall::CallForGetText = 0;

/*
 * QRCodeDataCall::CallForSetText
 */
const unsigned int QRCodeDataCall::CallForSetText = 1;

/*
 * QRCodeDataCall::CallForGetPointer
 */
const unsigned int QRCodeDataCall::CallForGetPointAt = 2;

/*
 * QRCodeDataCall::CallForSetPointer
 */
const unsigned int QRCodeDataCall::CallForSetPointAt = 3;

/*
 * QRCodeDataCall::CallForGetBoundingBox
 */
const unsigned int QRCodeDataCall::CallForGetBoundingBox = 4;

/*
 * QRCodeDataCall::CallForSetBoundingBox
 */
const unsigned int QRCodeDataCall::CallForSetBoundingBox = 5;

QRCodeDataCall::QRCodeDataCall(void) : qr_text(NULL), qr_pointer(NULL), bbox(NULL) {}


QRCodeDataCall::~QRCodeDataCall(void) {
    qr_text = NULL;
    qr_pointer = NULL;
    bbox = NULL;
}
} // namespace megamol::geocalls
