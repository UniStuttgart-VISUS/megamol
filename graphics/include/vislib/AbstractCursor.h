/*
 * AbstractCursor.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTCURSOR_H_INCLUDED
#define VISLIB_ABSTRACTCURSOR_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/IllegalParamException.h"
#include "vislib/SingleLinkedList.h"
#include "vislib/AbstractCursorEvent.h"
#include "vislib/InputModifiers.h"


namespace vislib {
namespace graphics {

    /**
     * Abstract base class for cursors
     */
    class AbstractCursor : public InputModifiers::Observer {
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
         * Sets the State of a button and triggers all registered cursor 
         * events, if all of their tests succeed, with REASON_BUTTON_DOWN or
         * with REASON_BUTTON_UP depending on the parameter down. 
         *
         * If you are up to call SetButtonState and SetModifierState of the 
         * corresponding InputModifiers object all at once, you should first 
         * do all of your SetModifierState calls and then do all of your 
         * SetButtonState calls, to ensure that the modifier tests of the 
         * events are correctly evaluated.
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
         * Callback on modifier changed.
         * @see vislib::graphics::InputModifiers::Observer::ModifierChanged
         */
        virtual void ModifierChanged(const InputModifiers& sender, 
            InputModifiers::Modifier mod, bool pressed);

        /**
         * Removes a cursor event from the observer queue of this cursor. If
         * this cursor event is not registered to this cursor, nothing is done.
         * The event is triggered with REASON_REMOVE, regardless of its tests.
         *
         * @param cursorEvent The cursor event to be added.
         */
        void UnregisterCursorEvent(AbstractCursorEvent *cursorEvent);

        /**
         * Associates a InputModifiers object with this object. This must be 
         * used with care since differences in the states of the current and
         * the new InputModifiers object are not checked! The replaced object
         * will not be freed, since this object does not take the ownership of
         * the memory of the InputModifiers objects.
         *
         * @param mods The new InputModifiers object.
         */
        inline void SetInputModifiers(InputModifiers *mods) {
            this->mods = mods;
        }

        /**
         * Answer the associated InputModifiers object.
         *
         * @return The associated InputModifiers object.
         */
        inline InputModifiers * GetInputModifiers(void) const {
            return this->mods;
        }

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

        /** Pointer to InputModifiers object */
        InputModifiers *mods;

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
