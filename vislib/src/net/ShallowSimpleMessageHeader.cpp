/*
 * ShallowSimpleMessageHeader.cpp
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/net/ShallowSimpleMessageHeader.h"


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
vislib::net::ShallowSimpleMessageHeader::~ShallowSimpleMessageHeader() {}


/*
 * vislib::net::ShallowSimpleMessageHeader::PeekData
 */
vislib::net::SimpleMessageHeaderData* vislib::net::ShallowSimpleMessageHeader::PeekData() {
    return this->data;
}


/*
 * vislib::net::ShallowSimpleMessageHeader::PeekData
 */
const vislib::net::SimpleMessageHeaderData* vislib::net::ShallowSimpleMessageHeader::PeekData() const {
    return this->data;
}


/*
 * vislib::net::ShallowSimpleMessageHeader::ShallowSimpleMessageHeader
 */
vislib::net::ShallowSimpleMessageHeader::ShallowSimpleMessageHeader() : data(NULL) {}
