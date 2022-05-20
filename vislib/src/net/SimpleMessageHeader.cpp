/*
 * SimpleMessageHeader.cpp
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "mmcore/utility/net/SimpleMessageHeader.h"


/*
 * vislib::net::SimpleMessageHeader::SimpleMessageHeader
 */
vislib::net::SimpleMessageHeader::SimpleMessageHeader(void) {
    this->PeekData()->MessageID = 0;
    this->PeekData()->BodySize = 0;
}


/*
 * vislib::net::SimpleMessageHeader::SimpleMessageHeader
 */
vislib::net::SimpleMessageHeader::SimpleMessageHeader(const SimpleMessageHeader& rhs) : Super() {
    *this = rhs;
}


/*
 * vislib::net::SimpleMessageHeader::SimpleMessageHeader
 */
vislib::net::SimpleMessageHeader::SimpleMessageHeader(const AbstractSimpleMessageHeader& rhs) : Super() {
    *this = rhs;
}


/*
 * vislib::net::SimpleMessageHeader::SimpleMessageHeader
 */
vislib::net::SimpleMessageHeader::SimpleMessageHeader(const SimpleMessageHeaderData& data) : Super() {
    *this = data;
}


/*
 * vislib::net::SimpleMessageHeader::SimpleMessageHeader
 */
vislib::net::SimpleMessageHeader::SimpleMessageHeader(const SimpleMessageHeaderData* data) : Super() {
    *this = data;
}


/*
 * vislib::net::SimpleMessageHeader::~SimpleMessageHeader
 */
vislib::net::SimpleMessageHeader::~SimpleMessageHeader(void) {}


/*
 * vislib::net::SimpleMessageHeader::PeekData
 */
vislib::net::SimpleMessageHeaderData* vislib::net::SimpleMessageHeader::PeekData(void) {
    return &(this->data);
}


/*
 * vislib::net::SimpleMessageHeader::PeekData
 */
const vislib::net::SimpleMessageHeaderData* vislib::net::SimpleMessageHeader::PeekData(void) const {
    return &(this->data);
}
