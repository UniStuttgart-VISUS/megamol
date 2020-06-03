/*
 * AbstractParamPresentation.h
 *
 * Copyright (C) 2020 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_ABSTRACTPARAMPRESENTATION_H_INCLUDED
#define MEGAMOLCORE_ABSTRACTPARAMPRESENTATION_H_INCLUDED

#include "mmcore/api/MegaMolCore.std.h"


namespace megamol {
namespace core {
namespace param {


class MEGAMOLCORE_API AbstractParamPresentation {
public:

    enum Presentations : int {
        RawValue           = 0,           // Presentation representing the parameters value with default raw value widget
        PinValueToMouse    = 1 << 1       // Presentation pinning value of parameter to mouse position
    };

    /**
    * Answer visibility in GUI.
    *
    * @return GUI visibility
    */
    inline bool IsGUIVisible() const {
        return this->visible;
    }

    /**
    * Set visibility in GUI.
    *
    * @param visible True: visible in GUI, false: invisible
    */
    inline void SetGUIVisible(const bool visible) {
        this->visible = visible;
    }

    /**
    * Answer accessibility in GUI.
    *
    * @return GUI accessibility
    */
    inline bool IsGUIReadOnly() const {
        return this->read_only;
    }

    /**
    * Set accessibility in GUI.
    *
    * @param read_only True: read-only in GUI, false: writable
    */
    inline void SetGUIReadOnly(const bool read_only) {
        this->read_only = read_only;
    }     
    
    /**
    * Set presentation of parameter in GUI.
    *
    * @param presentation Presentation of parameter in GUI.
    */
    inline void SetGUIPresentation(AbstractParamPresentation::Presentations presentation) {
        this->presentation = presentation;
    }
    
    /**
    * Answer parameter presentation in GUI.
    *
    * @return GUI presentation.
    */
    inline AbstractParamPresentation::Presentations GetGUIPresentation() const {
        return this->presentation;
    }

protected:

    AbstractParamPresentation(void);

    virtual ~AbstractParamPresentation(void) = default;

private:

    /* Show or hide the parameter in the GUI.
       Paramter is implicitly hidden in GUI if other than raw value view is selected. */
    bool visible;

    /* Make parameter read-only in the GUI. */
    bool read_only;
    
    /* Presentation (= widget representation) of parameter in the GUI. */
    AbstractParamPresentation::Presentations presentation;
    
};

} /* end namespace param */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_ABSTRACTPARAMPRESENTATION_H_INCLUDED */
