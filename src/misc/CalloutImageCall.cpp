#include "stdafx.h"
#include "misc/CalloutImageCall.h"

using namespace megamol::core;


/*
 * CalloutImageCall::CallForGetID
 */
const unsigned int misc::CalloutImageCall::CallForGetID = 0;

/*
 * CalloutImageCall::CallForSetID
 */
const unsigned int misc::CalloutImageCall::CallForSetID = 1;

/*
 * CalloutImageCall::CallForGetImage
 */
const unsigned int misc::CalloutImageCall::CallForGetImage = 2;

/*
 * CalloutImageCall::CallForSetImage
 */
const unsigned int misc::CalloutImageCall::CallForSetImage = 3;

/*
 * CalloutImageCall::CallForGetPointer
 */
const unsigned int misc::CalloutImageCall::CallForGetPointAt = 4;

/*
 * CalloutImageCall::CallForSetPointer
 */
const unsigned int misc::CalloutImageCall::CallForSetPointAt = 5;

/*
 * CalloutImageCall::CallForGetWidth
 */
const unsigned int misc::CalloutImageCall::CallForGetWidth = 6;

/*
 * CalloutImageCall::CallForSetWidth
 */
const unsigned int misc::CalloutImageCall::CallForSetWidth = 7;

/*
 * CalloutImageCall::CallForGetHeight
 */
const unsigned int misc::CalloutImageCall::CallForGetHeight = 8;

/*
 * CalloutImageCall::CallForSetHeight
 */
const unsigned int misc::CalloutImageCall::CallForSetHeight = 9;

int qr_id_value = -1;
int width_value = 0;
int height_value = 0;

megamol::core::misc::CalloutImageCall::CalloutImageCall(void) : qr_id(&qr_id_value), qr_image(NULL), qr_pointer(NULL), width(&width_value), height(&height_value)
{

}


misc::CalloutImageCall::~CalloutImageCall(void) 
{
	qr_image = NULL;
	qr_pointer = NULL;
}
