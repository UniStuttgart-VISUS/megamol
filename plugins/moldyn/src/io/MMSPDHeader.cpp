/*
 * MMSPDHeader.cpp
 *
 * Copyright (C) 2011 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "io/MMSPDHeader.h"

using namespace megamol::moldyn::io;


/****************************************************************************/

/*
 * MMSPDHeader::Field::Field
 */
MMSPDHeader::Field::Field(void) : name(), type(TYPE_DOUBLE) {
    // intentionally empty
}


/*
 * MMSPDHeader::Field::~Field
 */
MMSPDHeader::Field::~Field(void) {
    // intentionally empty
}


/*
 * MMSPDHeader::Field::operator==
 */
bool MMSPDHeader::Field::operator==(const MMSPDHeader::Field& rhs) const {
    return this->name.Equals(rhs.name) && (this->type == rhs.type);
}


/****************************************************************************/

/*
 * MMSPDHeader::ConstField::ConstField
 */
MMSPDHeader::ConstField::ConstField(void) : Field() {
    this->data.valDouble = 0.0;
}


/*
 * MMSPDHeader::ConstField::~ConstField
 */
MMSPDHeader::ConstField::~ConstField(void) {
    // intentionally empty
}


/*
 * MMSPDHeader::ConstField::operator==
 */
bool MMSPDHeader::ConstField::operator==(const MMSPDHeader::ConstField& rhs) const {
    return Field::operator==(rhs) &&
           (((this->GetType() == TYPE_BYTE) && (this->data.valByte == rhs.data.valByte)) ||
               ((this->GetType() == TYPE_FLOAT) && (this->data.valFloat == rhs.data.valFloat)) ||
               ((this->GetType() == TYPE_DOUBLE) && (this->data.valDouble == rhs.data.valDouble)));
}


/****************************************************************************/

/*
 * MMSPDHeader::TypeDefinition::TypeDefinition
 */
MMSPDHeader::TypeDefinition::TypeDefinition(void) {
    // intentionally empty
}


/*
 * MMSPDHeader::TypeDefinition::~TypeDefinition
 */
MMSPDHeader::TypeDefinition::~TypeDefinition(void) {
    // intentionally empty
}


/*
 * MMSPDHeader::TypeDefinition::operator==
 */
bool MMSPDHeader::TypeDefinition::operator==(const MMSPDHeader::TypeDefinition& rhs) const {
    // intentionally empty
    return true;
}


/****************************************************************************/

/*
 * MMSPDHeader::MMSPDHeader
 */
MMSPDHeader::MMSPDHeader(void)
        : hasIDs(false)
        , bbox(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)
        , timeCount(1)
        , types(1, TypeDefinition(), 1)
        , particleCount(0) {
    // intentionally empty
}


/*
 * MMSPDHeader::~MMSPDHeader
 */
MMSPDHeader::~MMSPDHeader(void) {
    // intentionally empty
}
