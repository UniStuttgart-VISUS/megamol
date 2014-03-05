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
	THE_STACK_TRACE;
	ASSERT(data != NULL);
	this->data = data;
}


/*
 * vislib::net::ShallowSimpleMessageHeader::~ShallowSimpleMessageHeader
 */
vislib::net::ShallowSimpleMessageHeader::~ShallowSimpleMessageHeader(void) {
    THE_STACK_TRACE;
}


/*
 * vislib::net::ShallowSimpleMessageHeader::PeekData
 */
vislib::net::SimpleMessageHeaderData *
vislib::net::ShallowSimpleMessageHeader::PeekData(void) {
	THE_STACK_TRACE;
	return this->data;
}


/*
 * vislib::net::ShallowSimpleMessageHeader::PeekData
 */
const vislib::net::SimpleMessageHeaderData *
vislib::net::ShallowSimpleMessageHeader::PeekData(void) const {
	THE_STACK_TRACE;
	return this->data;
}


/*
 * vislib::net::ShallowSimpleMessageHeader::ShallowSimpleMessageHeader
 */
vislib::net::ShallowSimpleMessageHeader::ShallowSimpleMessageHeader(void) 
		: data(NULL) {
	THE_STACK_TRACE;
}
