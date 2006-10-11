/*
 * AbstractCursor.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */


#include "vislib/AbstractCursor.h"
#include "vislib/memutils.h"
#include <memory.h>


/*
 * vislib::graphics::AbstractCursor::AbstractCursor
 */
vislib::graphics::AbstractCursor::AbstractCursor(void) 
        : btnCnt(0), btnStates(NULL), modCnt(0), modStates(NULL) {
    this->events.Clear();
}


/*
 * vislib::graphics::AbstractCursor::AbstractCursor
 */
vislib::graphics::AbstractCursor::AbstractCursor(const AbstractCursor& rhs) 
        : btnCnt(0), btnStates(NULL), modCnt(0), modStates(NULL) {
    this->events.Clear();
    *this = rhs;
}


/*
 * vislib::graphics::AbstractCursor::~AbstractCursor
 */
vislib::graphics::AbstractCursor::~AbstractCursor(void) {
    delete[] this->btnStates;
    delete[] this->modStates;
    
    this->events.Clear(); // Does not delete the event object listed
}


/*
 * vislib::graphics::AbstractCursor::operator=
 */
vislib::graphics::AbstractCursor& vislib::graphics::AbstractCursor::operator=(
        const vislib::graphics::AbstractCursor &rhs) {
    delete[] this->btnStates;
    delete[] this->modStates;
    this->btnCnt = rhs.btnCnt;
    this->modCnt = rhs.modCnt;
    this->btnStates = new bool[this->btnCnt];
    this->modStates = new bool[this->modCnt];
    ::memcpy(this->btnStates, rhs.btnStates, sizeof(bool) * this->btnCnt);
    ::memcpy(this->modStates, rhs.modStates, sizeof(bool) * this->modCnt);
    this->events = rhs.events;
    return *this;
}


/*
 * vislib::graphics::AbstractCursor::SetButtonCount
 */
void vislib::graphics::AbstractCursor::SetButtonCount(unsigned int btnCnt) {
    delete[] this->btnStates;
    this->btnCnt = btnCnt;
    this->btnStates = new bool[this->btnCnt];
    ::memset(this->btnStates, 0, sizeof(bool) * this->btnCnt);
}


/*
 * vislib::graphics::AbstractCursor::SetModifierCount
 */
void vislib::graphics::AbstractCursor::SetModifierCount(unsigned int modCnt) {
    delete[] this->modStates;
    this->modCnt = modCnt;
    this->modStates = new bool[this->modCnt];
    ::memset(this->modStates, 0, sizeof(bool) * this->modCnt);
}


/*
 * vislib::graphics::AbstractCursor::SetButtonState
 */
void vislib::graphics::AbstractCursor::SetButtonState(unsigned int btn, bool down) {
    if (btn >= this->btnCnt) {
        throw IllegalParamException("btn", __FILE__, __LINE__);
    }

    if (down) {
        this->btnStates[btn] = true;
        this->TestTriggerAllEvents(true, true, 
            AbstractCursorEvent::REASON_BUTTON_DOWN, btn);
    } else {
        this->TestTriggerAllEvents(true, true, 
            AbstractCursorEvent::REASON_BUTTON_UP, btn);
        this->btnStates[btn] = false;
    }
}


/*
 * vislib::graphics::AbstractCursor::SetModifierState
 */
void vislib::graphics::AbstractCursor::SetModifierState(unsigned int modifier, bool down) {
    if (modifier >= this->modCnt) {
        throw IllegalParamException("btn", __FILE__, __LINE__);
    }

    if (down) {
        this->modStates[modifier] = true;
        this->TestTriggerAllEvents(true, true, 
            AbstractCursorEvent::REASON_MOD_DOWN, modifier);
    } else {
        this->TestTriggerAllEvents(true, true, 
            AbstractCursorEvent::REASON_MOD_UP, modifier);
        this->modStates[modifier] = false;
    }
}


/*
 * vislib::graphics::AbstractCursor::UnregisterCursorEvent
 */
void vislib::graphics::AbstractCursor::UnregisterCursorEvent(AbstractCursorEvent *cursorEvent) {
    this->events.Remove(cursorEvent);
    cursorEvent->Trigger(this, AbstractCursorEvent::REASON_REMOVED, 0);
}


/*
 * vislib::graphics::AbstractCursor::TriggerMoved
 */
void vislib::graphics::AbstractCursor::TriggerMoved(void) {
    this->TestTriggerAllEvents(true, true, AbstractCursorEvent::REASON_MOVE, 0);
}


/*
 * vislib::graphics::AbstractCursor::RegisterCursorEvent
 */
void vislib::graphics::AbstractCursor::RegisterCursorEvent(AbstractCursorEvent *cursorEvent) {
    this->events.Append(cursorEvent);
    cursorEvent->Trigger(this, AbstractCursorEvent::REASON_ADDED, 0);
}


/*
 * vislib::graphics::AbstractCursor::TestEvent
 */
bool vislib::graphics::AbstractCursor::TestEvent(
        AbstractCursorEvent *cursorEvent, bool testBtn, bool testMod) const {

    if (testBtn && cursorEvent->DoesButtonTest()) {
        unsigned int btn = cursorEvent->GetTestButton();
        if ((btn >= this->btnCnt) || (!this->btnStates[btn])) {
            return false;
        }
    }

    if (testMod && cursorEvent->DoesModifierTest()) {
        unsigned int modTestCount = cursorEvent->GetModifierTestCount();
        for (unsigned int i = 0; i < modTestCount; i++) {
            unsigned int testMod = cursorEvent->GetTestModifier(i);
            if (testMod >= this->modCnt) {
                return false;
            }
            if (this->modStates[testMod] != cursorEvent->GetTestModifierValue(i)) {
                return false;
            }
        }
    }

    return true;
}


/*
 * vislib::graphics::AbstractCursor::TestTriggerAllEvents
 */
void vislib::graphics::AbstractCursor::TestTriggerAllEvents(bool testBtn, bool testMod, 
        AbstractCursorEvent::TriggerReason reason, unsigned int param) {

    vislib::SingleLinkedList<AbstractCursorEvent *>::Iterator it = this->events.GetIterator();

    while (it.HasNext()) {
        AbstractCursorEvent *e = it.Next();

        if (this->TestEvent(e, testBtn, testMod)) {
            e->Trigger(this, reason, param);
        }
    }
}
