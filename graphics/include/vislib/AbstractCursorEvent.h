/*
 * AbstractCursorEvent.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTCURSOREVENT_H_INCLUDED
#define VISLIB_ABSTRACTCURSOREVENT_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/IllegalParamException.h"
#include "vislib/InputModifiers.h"


namespace vislib {
namespace graphics {

    /* forward declaration */
    class AbstractCursor;

    /**
     * Abstract base class for cursor events
     */
    class AbstractCursorEvent {
    public:

        /** possible values for trigger reasons */
        enum TriggerReason {
            REASON_ADDED, /* Event has been added to an observer queue
                           * This event is triggered regardless of the set up
                           * tests.
                           * Parameter param is ignored.
                           */
            REASON_REMOVED, /* Event has been removed from an observer queue 
                             * This event is triggered regardless of the set up
                             * tests.
                             * Parameter param is ignored.
                             */
            REASON_MOVE, /* Cursor has moved 
                          * Parameter param is ignored.
                          */
            REASON_BUTTON_DOWN, /* Button down has occured 
                                 * Parameter param is the button.
                                 */
            REASON_BUTTON_UP, /* Button up has occured 
                               * Parameter param is the button.
                               */
            REASON_MOD_DOWN, /* Modifier down has occured 
                              * Parameter param is the Modifier.
                              */
            REASON_MOD_UP, /* Modifier up has occured 
                            * Parameter param is the Modifier.
                            */
            REASON_BOREDOM /* The process was bored and wanted to do something 
                            * Parameter param is set to an arbitrary value, but
                            * might be ignored.
                            */
        };

        /** ctor */
        AbstractCursorEvent(void);

        /** Dtor. */
        virtual ~AbstractCursorEvent(void);

        /**
         * Sets the count of modifier tests and sets all modifier tests to 
         * modifier 0 and value false.
         *
         * @param modTestCount Number of modifier tests for this event.
         */
        void SetModifierTestCount(unsigned int modTestCount);

        /**
         * Sets the modifier test i to modifier 'modifier' and value 'value'.
         *
         * @param i The number of modifer test.
         * @param modifier The modifier for the test i.
         * @param value The value for the test i.
         *
         * @throws IllegalParamException if i is larger or equal to number of
         *         modifier tests.
         */
        void SetModifierTest(unsigned int i, InputModifiers::Modifier modifier, bool value);

        /**
         * Sets the button for the button test and activates the button test.
         *
         * @param button The button for the test.
         */
        void SetTestButton(unsigned int button);

        /**
         * Deactivates the button test.
         */
        inline void DeactivateButtonTest(void) {
            this->testButton = false;
        }

        /**
         * Answer whether the button test is activated.
         *
         * @return true if the button test is activated, false otherwise.
         */
        inline bool DoesButtonTest(void) {
            return this->testButton;
        }

        /**
         * Answer whether modifier tests are activated.
         *
         * @return true if at least on modifier test is set, false otherwise.
         */
        inline bool DoesModifierTest(void) {
            return this->countModTests > 0;
        }

        /**
         * Returns the button value of the button test.
         *
         * @return the button of the button test.
         */
        inline unsigned int GetTestButton(void) {
            return this->button;
        }

        /**
         * Returns the number of the modifier tests.
         *
         * @return this number of the modifier tests.
         */
        inline unsigned int GetModifierTestCount(void) {
            return this->countModTests;
        }

        /**
         * Returns the modifier of the i-th modifier test.
         *
         * @param i The number of the modifier test.
         *
         * @return The modifier of the i-th modifier test.
         *
         * @throws IllegalParamException if i is larger or equal the number of
         *         modifier tests.
         */
        inline InputModifiers::Modifier GetTestModifier(unsigned int i) {
            if (i >= this->countModTests) {
                throw IllegalParamException("i", __FILE__, __LINE__);
            }
            return this->modifiers[i];
        }

        /**
         * Returns the modifier test value of the i-th modifier test.
         *
         * @param i The number of the modifier test.
         *
         * @return The modifier test value of the i-th modifier test.
         *
         * @throws IllegalParamException if i is larger or equal the number of
         *         modifier tests.
         */
        inline bool GetTestModifierValue(unsigned int i) {
            if (i >= this->countModTests) {
                throw IllegalParamException("i", __FILE__, __LINE__);
            }
            return this->modifierValues[i];
        }

        /**
         * Is called by an AbstractCursor which has this event in it's observer
         * list, if an event occured which is of interest to this 
         * AbstractCursorEvent because of the succeeded tests.
         *
         * @param caller The AbstractCursor object which calles this methode.
         * @param reason The reason why for this call.
         * @param param A reason depending parameter.
         */
        virtual void Trigger(AbstractCursor *caller, TriggerReason reason, unsigned int param) = 0;

    private:

        /** The button of the button test. */
        unsigned int button;

        /** The number of modifier tests. */
        unsigned int countModTests;

        /** The modifiers for the tests. */
        InputModifiers::Modifier *modifiers;

        /** The modifier test values for the tests. */
        bool *modifierValues;

        /** Whether the button test should be performed. */
        bool testButton;

    };
    
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTCURSOREVENT_H_INCLUDED */
