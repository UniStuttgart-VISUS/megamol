/*
 * AbstractCursor.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTCURSOR_H_INCLUDED
#define VISLIB_ABSTRACTCURSOR_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/IllegalParamException.h"
#include "vislib/SingleLinkedList.h"
#include "vislib/AbstractCursorEvent.h"


namespace vislib {
namespace graphics {


    /**
     * Abstract base class for cursors
     */
    class AbstractCursor {
    public:

        /** Dtor. */
        virtual ~AbstractCursor(void);

        /**
         * Sets the number of buttons of this cursor and resets all button 
         * states. No events will be triggered!
         *
         * @param btnCnt The new number of buttons.
         */
        void SetButtonCount(unsigned int btnCnt);

        /**
         * Answer the number of buttons of this cursor.
         *
         * @return The number of buttons.
         */
        inline unsigned int GetButtonCount(void) {
            return this->btnCnt;
        }

        /**
         * Sets the number of modifiers for this cursor and resets all modifier
         * states. No events will be triggered!
         *
         * @param modCnt The new number of modifiers.
         */
        void SetModifierCount(unsigned int modCnt);

        /**
         * Answer the number of modifiers for this cursor.
         *
         * @return The number of modifiers.
         */
        inline unsigned int GetModifierCount(void) {
            return this->modCnt;
        }

        /**
         * Sets the State of a button and triggers all registered cursor 
         * events, if all of their tests succeed, with REASON_BUTTON_DOWN or
         * with REASON_MOD_UP depending on the parameter down. 
         *
         * If your up to call SetButtonState and SetModifierState in one all at
         * once, you should first do all of your SetModifierState calls and 
         * then do all of your SetButtonState calls, to ensure that the 
         * modifier tests of the events are correctly evaluated.
         *
         * @param btn The number of the button whichs state will be set.
         * @param down The new value for the buttons state.
         *
         * @throws IllegalParamException if btn is larger or equal to the 
         *         number of buttons of this cursor.
         */
        virtual void SetButtonState(unsigned int btn, bool down);

        /**
         * Answer the State of a button.
         *
         * @param btn The number of the button whichs state will be returned.
         *
         * @return The state of the button.
         *
         * @throws IllegalParamException if btn is larger or equal to the 
         *         number of buttons of this cursor.
         */
        inline bool GetButtonState(unsigned int btn) {
            if (btn >= this->btnCnt) {
                throw IllegalParamException("btn", __FILE__, __LINE__);
            }
            return this->btnStates[btn];
        }

        /**
         * Sets the State of a modifier and triggeres all registered cursor
         * events, if all of their tests succeed, with REASON_MOD_DOWN or with
         * REASON_MOD_UP depending on the parameter down.
         *
         * If your up to call SetButtonState and SetModifierState in one all at
         * once, you should first do all of your SetModifierState calls and 
         * then do all of your SetButtonState calls, to ensure that the 
         * modifier tests of the events are correctly evaluated.
         *
         * @param modifier The number of the modifier whichs state will be 
         *                 set.
         * @param down The new value for the modifiers state.
         *
         * @throws IllegalParamException if modifier is larger or equal to the
         *         number of modifiers for this cursor.
         */
        virtual void SetModifierState(unsigned int modifier, bool down);

        /**
         * Answer the State of a modifier.
         *
         * @param modifier The number of the modifier whichs state will be 
         *                 returned.
         *
         * @return The state of the modifier.
         *
         * @throws IllegalParamException if modifier is larger or equal to the
         *         number of modifiers for this cursor.
         */
        inline bool GetModifierState(unsigned int modifier) {
            if (modifier >= this->modCnt) {
                throw IllegalParamException("modifier", __FILE__, __LINE__);
            }
            return this->modStates[modifier];
        }

        /**
         * Removes a cursor event from the observer queue of this cursor. If
         * this cursor event is not registered to this cursor, nothing is done.
         * The event is triggered with REASON_REMOVE, regardless of its tests.
         *
         * @param cursorEvent The cursor event to be added.
         */
        void UnregisterCursorEvent(AbstractCursorEvent *cursorEvent);


    protected:

        /** ctor */
        AbstractCursor(void);

        /** 
         * copy ctor 
         *
         * @rhs Reference to the source object
         */
        AbstractCursor(const AbstractCursor& rhs);

        /** 
         * Assignment operator
         *
         * @rhs Reference to the source object
         */
        AbstractCursor& operator=(const AbstractCursor &rhs);

        /**
         * Adds a cursor event to the observer queue of this cursor. The 
         * ownership of the event object is not changed. The object will not
         * be freed when this cursor is destroied and the caller must ensure
         * that the event object is valid as long as it is registered to this
         * cursor. The cursor event is triggered with REASON_ADD, regarless of
         * its tests. If the cursor event is already registered to this cursor, 
         * nothing is done.
         *
         * @param cursorEvent The cursor event to be added.
         */
        virtual void RegisterCursorEvent(AbstractCursorEvent *cursorEvent);

        /**
         * Child classes must call this methode if the cursor position changed
         * to trigger all registered cursor events with REASON_MOVE.
         */
        void TriggerMoved(void);
    private:

        /** number of buttons */
        unsigned int btnCnt;

        /** button states */
        bool *btnStates;

        /** number of modifiers */
        unsigned int modCnt;

        /** modifier states */
        bool *modStates;

        /** observer list of registered cursor events */
        SingleLinkedList<AbstractCursorEvent *> events;

        /**
         * Tests if an cursor event may be triggered with the current state of
         * the cursor.
         *
         * @param cursorEvent The cursor event to test.
         * @param testBtn Whether or not to do the button test.
         * @param testMod Whether or not to do the modifier tests.
         *
         * @return true if all cursor event tests succeeded, false otherwise.
         */
        bool TestEvent(AbstractCursorEvent *cursorEvent, bool testBtn, bool testMod) const;

        /**
         * Tests all events and triggers all events with succeeded tests with 
         * the given parameters.
         *
         * @param testBtn Whether or not to do the button test.
         * @param testMod Whether or not to do the modifier tests.
         * @param reason The reason for the trigger call.
         * @param param The reason parameter.
         */
        void TestTriggerAllEvents(bool testBtn, bool testMod, 
            AbstractCursorEvent::TriggerReason reason, unsigned int param);
    };
    
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTCURSOR_H_INCLUDED */
