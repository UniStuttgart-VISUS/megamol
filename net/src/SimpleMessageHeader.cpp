/*
 * SimpleMessageHeader.cpp
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/SimpleMessageHeader.h"


/*
 * vislib::net::SimpleMessageHeader::SimpleMessageHeader
 */
vislib::net::SimpleMessageHeader::SimpleMessageHeader(void) {
	VLSTACKTRACE("SimpleMessageHeader::SimpleMessageHeader", __FILE__, 
		__LINE__);
	this->data->MessageID = 0;
	this->data->BodySize = 0;
}


/*
 * vislib::net::SimpleMessageHeader::SimpleMessageHeader
 */
vislib::net::SimpleMessageHeader::SimpleMessageHeader(
		const SimpleMessageHeader& rhs) : Super() {
	VLSTACKTRACE("SimpleMessageHeader::SimpleMessageHeader", __FILE__, 
		__LINE__);
	*this = rhs;
}


/*
 * vislib::net::SimpleMessageHeader::SimpleMessageHeader
 */
vislib::net::SimpleMessageHeader::SimpleMessageHeader(
		const SimpleMessageHeaderData& data) : Super() {
	VLSTACKTRACE("SimpleMessageHeader::SimpleMessageHeader", __FILE__, 
		__LINE__);
	*this = data;
}


/*
 * vislib::net::SimpleMessageHeader::SimpleMessageHeader
 */
vislib::net::SimpleMessageHeader::SimpleMessageHeader(
		const SimpleMessageHeaderData *data) : Super() {
	VLSTACKTRACE("SimpleMessageHeader::SimpleMessageHeader", __FILE__, 
		__LINE__);
	*this = data;
}


/*
 * vislib::net::SimpleMessageHeader::~SimpleMessageHeader
 */
vislib::net::SimpleMessageHeader::~SimpleMessageHeader(void) {
	VLSTACKTRACE("SimpleMessageHeader::~SimpleMessageHeader", __FILE__, 
		__LINE__);
}
