/**
 * MegaMol
 * Copyright (c) 2008, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <functional>

#include "mmcore/param/AbstractParamPresentation.h"

namespace megamol::core::param {

/** forward declaration of owning class */
class AbstractParamSlot;

/**
 * Abstract base class for all parameter objects
 */
class AbstractParam {
public:
    friend class AbstractParamSlot;

    using ParamChangeCallback = std::function<void(AbstractParamSlot*)>;
    using PresentationChangeCallback = std::function<void(AbstractParamSlot*)>;

    /**
     * Dtor.
     */
    virtual ~AbstractParam();

    /**
     * Tries to parse the given string as value for this parameter and
     * sets the new value if successful. This also triggers the update
     * mechanism of the slot this parameter is assigned to.
     *
     * @param v The new value for the parameter as string.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool ParseValue(std::string const& v) = 0;

    /**
     * Returns the value of the parameter as string.
     *
     * @return The value of the parameter as string.
     */
    virtual std::string ValueString() const = 0;

    /**
     * Must be public for Button Press - Manuel Graeber
     * Sets the dirty flag of the owning parameter slot and might call the
     * update callback.
     */
    void setDirty();

    /**
     * Returns the value of the hash.
     *
     * @return The value of the hash.
     */
    inline uint64_t GetHash() const {
        return this->hash;
    }

    /**
     * Sets the value of the hash.
     *
     * @param hash The value of the hash.
     */
    inline void SetHash(const uint64_t& hash) {
        this->hash = hash;
    }

    void SetParamChangeCallback(ParamChangeCallback const& callback) {
        this->param_change_callback = callback;
    }

    void SetPresentationChangeCallback(PresentationChangeCallback const& callback) {
        this->presentation_change_callback = callback;
    }

    // TODO Temporary add wrappers around GuiPresentation() to avoid breaking changes for modules and merge hotfix
    //  until we know how this should be solved cleanly.
    inline void InitPresentation(AbstractParamPresentation::ParamType param_type) {
        gui_presentation.InitPresentation(param_type);
        indicatePresentationChange();
    }

    inline bool IsGUIVisible() const {
        return gui_presentation.IsGUIVisible();
    }

    inline void SetGUIVisible(bool visible) {
        if (gui_presentation.IsGUIVisible() != visible) {
            gui_presentation.SetGUIVisible(visible);
            indicatePresentationChange();
        }
    }

    inline bool IsGUIReadOnly() const {
        return gui_presentation.IsGUIReadOnly();
    }

    inline void SetGUIReadOnly(bool read_only) {
        if (gui_presentation.IsGUIReadOnly() != read_only) {
            gui_presentation.SetGUIReadOnly(read_only);
            indicatePresentationChange();
        }
    }

    bool IsGUIHighlight() const {
        return gui_presentation.IsHighlight();
    }

    void SetGUIHighlight(bool highlight) {
        if (gui_presentation.IsHighlight() != highlight) {
            gui_presentation.SetHighlight(highlight);
            indicatePresentationChange();
        }
    }

    inline AbstractParamPresentation::Presentation GetGUIPresentation() const {
        return gui_presentation.GetGUIPresentation();
    }

    void SetGUIPresentation(AbstractParamPresentation::Presentation presentS) {
        if (gui_presentation.GetGUIPresentation() != presentS) {
            gui_presentation.SetGUIPresentation(presentS);
            indicatePresentationChange();
        }
    }

protected:
    /**
     * Ctor.
     */
    AbstractParam();

    /**
     * Answers whether this parameter object is assigned to a public slot.
     *
     * @return 'true' if this parameter object is assigned to a public
     *         slot, 'false' otherwise.
     */
    bool isSlotPublic() const;

    void indicateParamChange() {
        param_change_callback(slot);
    }

    void indicatePresentationChange() {
        presentation_change_callback(slot);
    }

private:
    /** The holding slot */
    class AbstractParamSlot* slot = nullptr;

    /**
     * Hash indicating fundamental changes in parameter definition
     * (i.e. requires rebuilding the UI).
     */
    uint64_t hash;

    /**
     * The change callback is set by the MegaMol Graph/Frontend as a notification mechanism
     * to be made aware of module-driven or other parameters changes not made via the lua parameter setter function
     */
    ParamChangeCallback param_change_callback = [](auto*) {
        // needs default init for randomly created modules/params not to crash for default SetValue() calls
    };

    PresentationChangeCallback presentation_change_callback = [](auto*) {};

    AbstractParamPresentation gui_presentation;
};


} // namespace megamol::core::param
