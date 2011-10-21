/*
 * MMSPDHeader.cpp
 *
 * Copyright (C) 2011 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "MMSPDHeader.h"

using namespace megamol::core;


/****************************************************************************/

/*
 * moldyn::MMSPDHeader::Field::Field
 */
moldyn::MMSPDHeader::Field::Field(void) : name(), type(TYPE_DOUBLE) {
    // intentionally empty
}


/*
 * moldyn::MMSPDHeader::Field::~Field
 */
moldyn::MMSPDHeader::Field::~Field(void) {
    // intentionally empty
}


/*
 * moldyn::MMSPDHeader::Field::operator==
 */
bool moldyn::MMSPDHeader::Field::operator==(
        const moldyn::MMSPDHeader::Field& rhs) const {
    return this->name.Equals(rhs.name)
        && (this->type == rhs.type);
}


/****************************************************************************/

/*
 * moldyn::MMSPDHeader::ConstField::ConstField
 */
moldyn::MMSPDHeader::ConstField::ConstField(void) : Field() {
    this->data.valDouble = 0.0;
}


/*
 * moldyn::MMSPDHeader::ConstField::~ConstField
 */
moldyn::MMSPDHeader::ConstField::~ConstField(void) {
    // intentionally empty
}


/*
 * moldyn::MMSPDHeader::ConstField::operator==
 */
bool moldyn::MMSPDHeader::ConstField::operator==(
        const moldyn::MMSPDHeader::ConstField& rhs) const {
    return Field::operator==(rhs)
        && (((this->GetType() == TYPE_BYTE) && (this->data.valByte == rhs.data.valByte))
        || ((this->GetType() == TYPE_FLOAT) && (this->data.valFloat == rhs.data.valFloat))
        || ((this->GetType() == TYPE_DOUBLE) && (this->data.valDouble == rhs.data.valDouble))
        );
}


/****************************************************************************/

/*
 * moldyn::MMSPDHeader::TypeDefinition::TypeDefinition
 */
moldyn::MMSPDHeader::TypeDefinition::TypeDefinition(void) {
    // intentionally empty
}


/*
 * moldyn::MMSPDHeader::TypeDefinition::~TypeDefinition
 */
moldyn::MMSPDHeader::TypeDefinition::~TypeDefinition(void) {
    // intentionally empty
}


/*
 * moldyn::MMSPDHeader::TypeDefinition::operator==
 */
bool moldyn::MMSPDHeader::TypeDefinition::operator==(
        const moldyn::MMSPDHeader::TypeDefinition& rhs) const {
    // intentionally empty
    return true;
}


/****************************************************************************/

/*
 * moldyn::MMSPDHeader::MMSPDHeader
 */
moldyn::MMSPDHeader::MMSPDHeader(void) : hasIDs(false),
        bbox(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0), timeCount(1),
        types(1, TypeDefinition(), 1), particleCount(0) {
    // intentionally empty
}


/*
 * moldyn::MMSPDHeader::~MMSPDHeader
 */
moldyn::MMSPDHeader::~MMSPDHeader(void) {
    // intentionally empty
}
