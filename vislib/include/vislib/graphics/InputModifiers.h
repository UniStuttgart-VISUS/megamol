/*
 * InputModifiers.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_INPUTMODIFIERS_H_INCLUDED
#define VISLIB_INPUTMODIFIERS_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/IllegalParamException.h"
#include "vislib/SingleLinkedList.h"


namespace vislib {
namespace graphics {


    /**
     * This class manages the input modifieres of the system. Normally these
     * are "Ctrl", "Alt", and "Shift".
     */
    class InputModifiers {
    public:

        /** type of a modifier (key) */
        typedef unsigned int Modifier;

        /**
         * Base interface class for all classes which want to receive uptdate-
         * callbacks from an InputModifiers object.
         */
        class Observer {
        public:

#ifndef _WIN32
            /** dtor */
            virtual ~Observer(void) { /* just to make stupid gcc 4 happy */ }
#endif /* !_WIN32 */

            /** 
             * called when the state of a modifier changed. The modifiers state
             * is always true, so the observers are informed on press event 
             * after the state has been set, and they are informed on release
             * events before the state has been unset.
             *
             * WARNING: setting the state of any modifier of the InputModifiers
             * object from within this callback may result in an endless
             * recursion.
             * 
             * @param sender The InputModifiers object calling.
             * @param mod The modifier which state changed.
             * @param pressed true if the reason for the call was that the
             *                modifier has been pressed, false if the modifier
             *                has been released.
             */
            virtual void ModifierChanged(const InputModifiers& sender, Modifier mod, bool pressed) = 0;
        };

        /** 
         * Ctor. 
         *
         * @param modCount The number of available modifiers. [0... modCount-1]
         */
        InputModifiers(unsigned int modCount = 3);

        /** Dtor. */
        ~InputModifiers(void);

        /**
         * Sets the number of modifiers and resets all modifier states. No 
         * events will be triggered at the observers!
         *
         * @param modCnt The new number of modifiers.
         */
        void SetModifierCount(unsigned int modCnt);

        /**
         * Answer the number of modifiers.
         *
         * @return The number of modifiers.
         */
        inline unsigned int GetModifierCount(void) {
            return this->modCount;
        }

        /**
         * Sets the State of a modifier and informs all registered observers.
         *
         * @param modifier The number of the modifier whichs state will be 
         *                 set.
         * @param down The new value for the modifiers state.
         *
         * @throws IllegalParamException if modifier is larger or equal to the
         *         number of modifiers for this cursor.
         */
        void SetModifierState(Modifier modifier, bool down);

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
        inline bool GetModifierState(Modifier modifier) {
            if (modifier >= this->modCount) {
                throw IllegalParamException("modifier", __FILE__, __LINE__);
            }
            return this->modState[modifier];
        }

        /**
         * Registers an observer, if it is not registered yet.
         *
         * @param observer The observer to be added to the observer list.
         */
        void RegisterObserver(Observer *observer);

        /**
         * Unregisters an observer.
         *
         * @param observer The observer which will be removed from the observer
         *                 list.
         */
        void UnregisterObserver(Observer *observer);

        /** symbolic constant for the shift modifier. (Value = 0) */
        static const Modifier MODIFIER_SHIFT;

        /** symbolic constant for the control modifier. (Value = 1) */
        static const Modifier MODIFIER_CTRL;
        
        /** symbolic constant for the alt modifier. (Value = 2) */
        static const Modifier MODIFIER_ALT;

    private:

        /** the number of modifiers. */
        unsigned int modCount;

        /** the state of the modifiers. */
        bool *modState;

        /** the list of registered observers. */
        SingleLinkedList<Observer*> observers;

    };
    
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_INPUTMODIFIERS_H_INCLUDED */

