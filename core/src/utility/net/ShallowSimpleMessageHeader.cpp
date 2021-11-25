/*
 * ShallowSimpleMessageHeader.cpp
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "mmcore/utility/net/ShallowSimpleMessageHeader.h"


/*
 * vislib::net::ShallowSimpleMessageHeader::ShallowSimpleMessageHeader
 */
vislib::net::ShallowSimpleMessageHeader::ShallowSimpleMessageHeader(SimpleMessageHeaderData* data) : Super() {
    ASSERT(data != NULL);
    this->data = data;
}


/*
 * vislib::net::ShallowSimpleMessageHeader::~ShallowSimpleMessageHeader
 */
vislib::net::ShallowSimpleMessageHeader::~ShallowSimpleMessageHeader(void) {}


/*
 * vislib::net::ShallowSimpleMessageHeader::PeekData
 */
vislib::net::SimpleMessageHeaderData* vislib::net::ShallowSimpleMessageHeader::PeekData(void) {
    return this->data;
}


/*
 * vislib::net::ShallowSimpleMessageHeader::PeekData
 */
const vislib::net::SimpleMessageHeaderData* vislib::net::ShallowSimpleMessageHeader::PeekData(void) const {
    return this->data;
}


/*
 * vislib::net::ShallowSimpleMessageHeader::ShallowSimpleMessageHeader
 */
vislib::net::ShallowSimpleMessageHeader::ShallowSimpleMessageHeader(void) : data(NULL) {}
