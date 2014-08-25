#include "stdafx.h"
#include "QRCodeDataCall.h"

using namespace megamol::core;


/*
 * QRCodeDataCall::CallForGetText
 */
const unsigned int misc::QRCodeDataCall::CallForGetText = 0;

/*
 * QRCodeDataCall::CallForSetText
 */
const unsigned int misc::QRCodeDataCall::CallForSetText = 1;

/*
 * QRCodeDataCall::CallForGetPointer
 */
const unsigned int misc::QRCodeDataCall::CallForGetPointAt = 2;

/*
 * QRCodeDataCall::CallForSetPointer
 */
const unsigned int misc::QRCodeDataCall::CallForSetPointAt = 3;

/*
 * QRCodeDataCall::CallForGetBoundingBox
 */
const unsigned int misc::QRCodeDataCall::CallForGetBoundingBox = 4;

/*
 * QRCodeDataCall::CallForSetBoundingBox
 */
const unsigned int misc::QRCodeDataCall::CallForSetBoundingBox = 5;

misc::QRCodeDataCall::QRCodeDataCall(void) : qr_text(NULL), qr_pointer(NULL), bbox(NULL)
{

}


misc::QRCodeDataCall::~QRCodeDataCall(void) 
{
	qr_text = NULL;
	qr_pointer = NULL;
	bbox = NULL;
}
