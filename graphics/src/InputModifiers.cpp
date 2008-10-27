/*
 * InputModifiers.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */


#include "vislib/types.h"
#include "vislib/InputModifiers.h"


/*
 * vislib::graphics::InputModifiers::MOD_SHIFT
 */
const vislib::graphics::InputModifiers::Modifier vislib::graphics::InputModifiers::MODIFIER_SHIFT = 0;


/*
 * vislib::graphics::InputModifiers::MOD_CTRL
 */
const vislib::graphics::InputModifiers::Modifier vislib::graphics::InputModifiers::MODIFIER_CTRL = 1;


/*
 * vislib::graphics::InputModifiers::MOD_ALT
 */
const vislib::graphics::InputModifiers::Modifier vislib::graphics::InputModifiers::MODIFIER_ALT = 2;


/*
 * vislib::graphics::InputModifiers::InputModifiers
 */
vislib::graphics::InputModifiers::InputModifiers(unsigned int modCount) 
        : modCount(0), modState(NULL), observers() {
    this->SetModifierCount(modCount);
}


/*
 * vislib::graphics::InputModifiers::~InputModifiers
 */
vislib::graphics::InputModifiers::~InputModifiers(void) {
    this->modCount = 0;
    delete[] this->modState;
    this->modState = NULL;
    this->observers.Clear();
}


/*
 * vislib::graphics::InputModifiers::SetModifierCount
 */
void vislib::graphics::InputModifiers::SetModifierCount(unsigned int modCnt) {
    delete[] this->modState;
    this->modCount = modCnt;
    this->modState = new bool[this->modCount];
    ::memset(this->modState, 0, sizeof(bool) * this->modCount);
}


/*
 * vislib::graphics::InputModifiers::SetModifierState
 */
void vislib::graphics::InputModifiers::SetModifierState(Modifier modifier, bool down) {
    if (modifier >= this->modCount) {
        throw IllegalParamException("modifier", __FILE__, __LINE__);
    }

    // no update message required
    if (this->modState[modifier] == down) return;

    if (down) this->modState[modifier] = true;

    vislib::SingleLinkedList<Observer*>::Iterator iter = this->observers.GetIterator();
    while (iter.HasNext()) {
        Observer *o = iter.Next();
        o->ModifierChanged(*this, modifier, down);
    }

    if (!down) this->modState[modifier] = false;
}


/*
 * vislib::graphics::InputModifiers::RegisterObserver
 */
void vislib::graphics::InputModifiers::RegisterObserver(Observer *observer) {
    if (!this->observers.Contains(observer)) {
        this->observers.Add(observer);
    }
}


/*
 * vislib::graphics::InputModifiers::UnregisterObserver
 */
void vislib::graphics::InputModifiers::UnregisterObserver(Observer *observer) {
    this->observers.RemoveAll(observer);
}
