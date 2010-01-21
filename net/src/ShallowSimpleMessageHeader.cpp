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
