/*
 * ShallowSimpleMessageHeader.cpp
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/ShallowSimpleMessageHeader.h"


/*
 * vislib::net::ShallowSimpleMessageHeader::ShallowSimpleMessageHeader
 */
vislib::net::ShallowSimpleMessageHeader::ShallowSimpleMessageHeader(
		SimpleMessageHeaderData *data) : Super() {
	VLSTACKTRACE("ShallowSimpleMessageHeader::ShallowSimpleMessageHeader", 
		__FILE__, __LINE__);
	ASSERT(data != NULL);
	this->data = data;
}


/*
 * vislib::net::ShallowSimpleMessageHeader::~ShallowSimpleMessageHeader
 */
vislib::net::ShallowSimpleMessageHeader::~ShallowSimpleMessageHeader(void) {
    VLSTACKTRACE("ShallowSimpleMessageHeader::~ShallowSimpleMessageHeader",
		__FILE__, __LINE__);
}


/*
 * vislib::net::ShallowSimpleMessageHeader::PeekData
 */
vislib::net::SimpleMessageHeaderData *
vislib::net::ShallowSimpleMessageHeader::PeekData(void) {
	VLSTACKTRACE("ShallowSimpleMessageHeader::PeekData", __FILE__, __LINE__);
	return this->data;
}


/*
 * vislib::net::ShallowSimpleMessageHeader::PeekData
 */
const vislib::net::SimpleMessageHeaderData *
vislib::net::ShallowSimpleMessageHeader::PeekData(void) const {
	VLSTACKTRACE("ShallowSimpleMessageHeader::PeekData", __FILE__, __LINE__);
	return this->data;
}


/*
 * vislib::net::ShallowSimpleMessageHeader::ShallowSimpleMessageHeader
 */
vislib::net::ShallowSimpleMessageHeader::ShallowSimpleMessageHeader(void) 
		: data(NULL) {
	VLSTACKTRACE("ShallowSimpleMessageHeader::ShallowSimpleMessageHeader", 
		__FILE__, __LINE__);
}
