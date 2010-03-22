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
	this->PeekData()->MessageID = 0;
	this->PeekData()->BodySize = 0;
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
		const AbstractSimpleMessageHeader& rhs) : Super() {
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


/*
 * vislib::net::SimpleMessageHeader::PeekData
 */
vislib::net::SimpleMessageHeaderData *
vislib::net::SimpleMessageHeader::PeekData(void) {
	VLSTACKTRACE("SimpleMessageHeader::PeekData", __FILE__, __LINE__);
	return &(this->data);
}


/*
 * vislib::net::SimpleMessageHeader::PeekData
 */
const vislib::net::SimpleMessageHeaderData *
vislib::net::SimpleMessageHeader::PeekData(void) const {
	VLSTACKTRACE("SimpleMessageHeader::PeekData", __FILE__, __LINE__);
	return &(this->data);
}
