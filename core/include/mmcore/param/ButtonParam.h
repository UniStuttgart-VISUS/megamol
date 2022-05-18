/*
 * ButtonParam.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_BUTTONPARAM_H_INCLUDED
#define MEGAMOLCORE_BUTTONPARAM_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "AbstractParam.h"
#include "mmcore/view/Input.h"
#include "vislib/types.h"


namespace megamol {
namespace core {
namespace param {


/**
 * Special class for parameter objects representing a button. These
 * objects have no value at all, but trigger the update callback of the
 * slot whenever the button in the gui is pressed.
 */
class ButtonParam : public AbstractParam {
public:
    /**
     * Ctor.
     *
     * Be aware, that if you do not assign a key, the button will not be
     * available from viewers without a GUI.
     */
    ButtonParam();

    /**
     * Ctor.
     *
     * @param keycode The prefered key code for the button (if any). Be aware, that
     *                if you do not assign a key, the button will not be
     *                available from viewers without a GUI.
     */
    ButtonParam(const core::view::KeyCode& keycode);

    /**
     * Ctor.
     *
     * @param keycode The prefered key code for the button (if any). Be aware, that
     *                if you do not assign a key, the button will not be
     *                available from viewers without a GUI.
     */
    ButtonParam(const core::view::Key& key);

    /**
     * Ctor.
     *
     * @param key  The prefered key for the button (if any). Be aware, that
     *             if you do not assign a key, the button will not be
     *             available from viewers without a GUI.
     * @param mods The prefered modifier for the button (if any). Be aware, that
     *             if you do not assign a key, the button will not be
     *             available from viewers without a GUI.
     */
    ButtonParam(const core::view::Key& key, const core::view::Modifiers& mods);

    /**
     * Ctor.
     *
     * @param key The prefered key for the button (if any). Be aware, that
     *            if you do not assign a key, the button will not be
     *            available from viewers without a GUI.
     * @param mod The prefered modifier for the button (if any). Be aware, that
     *            if you do not assign a key, the button will not be
     *            available from viewers without a GUI.
     */
    ButtonParam(const core::view::Key& key, const core::view::Modifier& mod);

    /**
     * Dtor.
     */
    virtual ~ButtonParam(void);

    /**
     * Returns a machine-readable definition of the parameter.
     *
     * @param outDef A memory block to receive a machine-readable
     *               definition of the parameter.
     */
    std::string Definition() const override;

    /**
     * Tries to parse the given string as value for this parameter and
     * sets the new value if successful. This also triggers the update
     * mechanism of the slot this parameter is assigned to.
     *
     * @param v The new value for the parameter as string.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool ParseValue(std::string const& v) override;

    /**
     * Returns the value of the parameter as string.
     *
     * @return The value of the parameter as string.
     */
    std::string ValueString(void) const override;

    inline core::view::KeyCode GetKeyCode(void) const {
        return this->keycode;
    }

private:
    void initialize();

    /** The key of this button */
    core::view::KeyCode keycode;
};


} /* end namespace param */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_BUTTONPARAM_H_INCLUDED */
