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
	THE_STACK_TRACE;
	this->PeekData()->MessageID = 0;
	this->PeekData()->BodySize = 0;
}


/*
 * vislib::net::SimpleMessageHeader::SimpleMessageHeader
 */
vislib::net::SimpleMessageHeader::SimpleMessageHeader(
		const SimpleMessageHeader& rhs) : Super() {
	THE_STACK_TRACE;
	*this = rhs;
}


/*
 * vislib::net::SimpleMessageHeader::SimpleMessageHeader
 */
vislib::net::SimpleMessageHeader::SimpleMessageHeader(
		const AbstractSimpleMessageHeader& rhs) : Super() {
	THE_STACK_TRACE;
	*this = rhs;
}


/*
 * vislib::net::SimpleMessageHeader::SimpleMessageHeader
 */
vislib::net::SimpleMessageHeader::SimpleMessageHeader(
		const SimpleMessageHeaderData& data) : Super() {
	THE_STACK_TRACE;
	*this = data;
}


/*
 * vislib::net::SimpleMessageHeader::SimpleMessageHeader
 */
vislib::net::SimpleMessageHeader::SimpleMessageHeader(
		const SimpleMessageHeaderData *data) : Super() {
	THE_STACK_TRACE;
	*this = data;
}


/*
 * vislib::net::SimpleMessageHeader::~SimpleMessageHeader
 */
vislib::net::SimpleMessageHeader::~SimpleMessageHeader(void) {
	THE_STACK_TRACE;
}


/*
 * vislib::net::SimpleMessageHeader::PeekData
 */
vislib::net::SimpleMessageHeaderData *
vislib::net::SimpleMessageHeader::PeekData(void) {
	THE_STACK_TRACE;
	return &(this->data);
}


/*
 * vislib::net::SimpleMessageHeader::PeekData
 */
const vislib::net::SimpleMessageHeaderData *
vislib::net::SimpleMessageHeader::PeekData(void) const {
	THE_STACK_TRACE;
	return &(this->data);
}
