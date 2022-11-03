/**
 * MegaMol
 * Copyright (c) 2018, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "mmcore/Call.h"
#include "mmcore/view/Input.h"
#include "mmcore/view/MouseFlags.h"

namespace megamol::core::view {

/**
 * Stateful input event.
 *
 * Note that this is a semi-shitty workaround for callbacks not having user defined signatures.
 * See also megamol::core::view::AbstractInputScope.
 */
struct InputEvent {
    enum class Tag : unsigned char { Empty, Key, Char, MouseButton, MouseMove, MouseScroll } tag;
    union {
        struct {
        } emptyData;

        struct {
            Key key;
            KeyAction action;
            Modifiers mods;
        } keyData;

        struct {
            unsigned int codePoint;
        } charData;

        struct {
            MouseButton button;
            MouseButtonAction action;
            Modifiers mods;
        } mouseButtonData;

        struct {
            double x;
            double y;
        } mouseMoveData;

        struct {
            double dx;
            double dy;
        } mouseScrollData;
    };

    InputEvent() : tag(Tag::Empty), emptyData() {}
};

/**
 * Base class of input calls
 */
class InputCall : public Call {
public:
    static const unsigned int FnOnKey = 0;
    static const unsigned int FnOnChar = 1;
    static const unsigned int FnOnMouseButton = 2;
    static const unsigned int FnOnMouseMove = 3;
    static const unsigned int FnOnMouseScroll = 4;

    /**
     * Answer the number of functions used for this call.
     *
     * @return The number of functions used for this call.
     */
    static unsigned int FunctionCount(void) {
        return 5;
    }

    /**
     * Answer the name of the function used for this call.
     *
     * @param idx The index of the function to return it's name.
     *
     * @return The name of the requested function.
     */
    static const char* FunctionName(unsigned int idx) {
        // clang-format off
#define CaseFunction(id) case Fn##id: return #id
        // clang-format on
        switch (idx) {
            CaseFunction(OnKey);
            CaseFunction(OnChar);
            CaseFunction(OnMouseButton);
            CaseFunction(OnMouseMove);
            CaseFunction(OnMouseScroll);
        default:
            return NULL;
        }
#undef CaseFunction
    }

    /** Ctor. */
    InputCall() = default;

    /** Dtor. */
    virtual ~InputCall(void) = default;


    /**
     * Answer the stored input event.
     *
     * @return The input event.
     */
    inline const InputEvent& GetInputEvent(void) const {
        return this->e;
    }

    /**
     * Stores an input event.
     *
     * @return The input event.
     */
    inline void SetInputEvent(const InputEvent& evt) {
        this->e = evt;
    }

    /**
     * Assignment operator
     *
     * @param rhs The right hand side operand
     *
     * @return A reference to this
     */
    InputCall& operator=(const InputCall& rhs) = default;

private:
    InputEvent e;
};


} // namespace megamol::core::view
