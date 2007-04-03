/*
 * AbstractCursorEvent.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */


#include "vislib/AbstractCursorEvent.h"
#include "vislib/memutils.h"


/*
 * vislib::graphics::AbstractCursorEvent::AbstractCursorEvent
 */
vislib::graphics::AbstractCursorEvent::AbstractCursorEvent(void) 
        : button(0), countModTests(0), modifiers(NULL), modifierValues(NULL), testButton(false) {
}


/*
 * vislib::graphics::AbstractCursorEvent::~AbstractCursorEvent
 */
vislib::graphics::AbstractCursorEvent::~AbstractCursorEvent(void) {
    delete[] this->modifiers;
    delete[] this->modifierValues;
}


/*
 * vislib::graphics::AbstractCursorEvent::SetModifierTestCount
 */
void vislib::graphics::AbstractCursorEvent::SetModifierTestCount(unsigned int modTestCount) {
    delete[] this->modifiers;
    delete[] this->modifierValues;
    this->countModTests = modTestCount;
    this->modifiers = new InputModifiers::Modifier[this->countModTests];
    this->modifierValues = new bool[this->countModTests];
    ::memset(this->modifiers, 0, sizeof(InputModifiers::Modifier) * this->countModTests);
    ::memset(this->modifierValues, 0, sizeof(bool) * this->countModTests);
}


/*
 * vislib::graphics::AbstractCursorEvent::SetModifierTest
 */
void vislib::graphics::AbstractCursorEvent::SetModifierTest(
        unsigned int i, InputModifiers::Modifier modifier, bool value) {
    if (i >= this->countModTests) {
        throw IllegalParamException("i", __FILE__, __LINE__);
    }
    this->modifiers[i] = modifier;
    this->modifierValues[i] = value;
}


/*
 * vislib::graphics::AbstractCursorEvent::SetTestButton
 */
void vislib::graphics::AbstractCursorEvent::SetTestButton(unsigned int button) {
    this->testButton = true;
    this->button = button;
}
