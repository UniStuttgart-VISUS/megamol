/*
 * AbstractCursor.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */


#include "vislib/AbstractCursor.h"
#include "vislib/memutils.h"
#include "vislib/Trace.h"

#include <memory.h>


/*
 * vislib::graphics::AbstractCursor::AbstractCursor
 */
vislib::graphics::AbstractCursor::AbstractCursor(void) 
        : btnCnt(0), btnStates(NULL), mods(NULL) {
    this->events.Clear();
}


/*
 * vislib::graphics::AbstractCursor::AbstractCursor
 */
vislib::graphics::AbstractCursor::AbstractCursor(const AbstractCursor& rhs) 
        : btnCnt(0), btnStates(NULL), mods(rhs.mods) {
    this->events.Clear();
    *this = rhs;
}


/*
 * vislib::graphics::AbstractCursor::~AbstractCursor
 */
vislib::graphics::AbstractCursor::~AbstractCursor(void) {
    delete[] this->btnStates;
    
    this->events.Clear(); // Does not delete the event object listed
}


/*
 * vislib::graphics::AbstractCursor::operator=
 */
vislib::graphics::AbstractCursor& vislib::graphics::AbstractCursor::operator=(
        const vislib::graphics::AbstractCursor &rhs) {
    delete[] this->btnStates;
    this->btnCnt = rhs.btnCnt;
    this->btnStates = new bool[this->btnCnt];
    ::memcpy(this->btnStates, rhs.btnStates, sizeof(bool) * this->btnCnt);
    this->mods = rhs.mods;
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
 * vislib::graphics::AbstractCursor::SetButtonState
 */
void vislib::graphics::AbstractCursor::SetButtonState(unsigned int btn, 
                                                      bool down) {
    // mueller: Some of our Linux-gluts report button 4 for scrolling using the
    //          mouse wheel. This causes the application to crash, because no 
    //          one checks for this error here. Therefore, I think it is better
    //          to fail silently.
    //if (btn >= this->btnCnt) {
    //    throw IllegalParamException("btn", __FILE__, __LINE__);
    //}
    if (btn >= this->btnCnt) {
        VLTRACE(Trace::LEVEL_VL_WARN, "%d is an illegal value for \"btn\".\n");
        return;
    }

    // no update message required
    if (this->btnStates[btn] == down) return;

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
 * vislib::graphics::AbstractCursor::ModifierChanged
 */
void vislib::graphics::AbstractCursor::ModifierChanged(
        const InputModifiers& sender, InputModifiers::Modifier mod, bool pressed) {
    // Trigger Events
    this->TestTriggerAllEvents(true, true, 
        pressed ? AbstractCursorEvent::REASON_MOD_DOWN : AbstractCursorEvent::REASON_MOD_UP,
        static_cast<unsigned int>(mod));
}

//
///*
// * vislib::graphics::AbstractCursor::SetModifierState
// */
//void vislib::graphics::AbstractCursor::SetModifierState(unsigned int modifier, bool down) {
//    if (modifier >= this->modCnt) {
//        throw IllegalParamException("btn", __FILE__, __LINE__);
//    }
//
//    if (down) {
//        this->modStates[modifier] = true;
//        this->TestTriggerAllEvents(true, true, 
//            AbstractCursorEvent::REASON_MOD_DOWN, modifier);
//    } else {
//        this->TestTriggerAllEvents(true, true, 
//            AbstractCursorEvent::REASON_MOD_UP, modifier);
//        this->modStates[modifier] = false;
//    }
//}


/*
 * vislib::graphics::AbstractCursor::UnregisterCursorEvent
 */
void vislib::graphics::AbstractCursor::UnregisterCursorEvent(AbstractCursorEvent *cursorEvent) {
    this->events.RemoveAll(cursorEvent);
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
    if (!this->events.Contains(cursorEvent)) {
        this->events.Append(cursorEvent);
        cursorEvent->Trigger(this, AbstractCursorEvent::REASON_ADDED, 0);
    }
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

    if (testMod && cursorEvent->DoesModifierTest() && (this->mods != NULL)) {
        unsigned int modTestCount = cursorEvent->GetModifierTestCount();
        for (unsigned int i = 0; i < modTestCount; i++) {
            InputModifiers::Modifier testMod = cursorEvent->GetTestModifier(i);
            if (testMod >= this->mods->GetModifierCount()) {
                return false;
            }
            if (this->mods->GetModifierState(testMod) != cursorEvent->GetTestModifierValue(i)) {
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
