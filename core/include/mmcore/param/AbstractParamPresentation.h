/*
 * AbstractParamPresentation.h
 *
 * Copyright (C) 2020 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_ABSTRACTPARAMPRESENTATION_H_INCLUDED
#define MEGAMOLCORE_ABSTRACTPARAMPRESENTATION_H_INCLUDED

#include "mmcore/api/MegaMolCore.std.h"

#include "vislib/sys/Log.h"


namespace megamol {
namespace core {
namespace param {


// Available GUI widget implementations for parameter presentation.
enum class Presentations : int { /// (limited to 32)
    NONE              = 0,
    Basic             = 1 << 1,      // Basic widget (must be supported for all parameter types) -> Default
    Color             = 1 << 2,      // Color editor widget
    FilePath          = 1 << 3,      // File path widget
    TransferFunction  = 1 << 4,      // Transfer function editor widget
    PinValueToMouse   = 1 << 5       // Pin parameter value to mouse position
};
inline Presentations operator|(Presentations a, Presentations b) {
    return static_cast<Presentations>(static_cast<int>(a) | static_cast<int>(b));
}
inline Presentations operator&(Presentations a, Presentations b) {
    return static_cast<Presentations>(static_cast<int>(a) & static_cast<int>(b));
}


class MEGAMOLCORE_API AbstractParamPresentation {
public:

    /**
    * Initalise presentation for parameter once.
    *
    * @param default_presentation   Default presentation to use for parameter.
    * @param compatible             Set compatible presentations.
    *
    * @return True on success, false otherwise.
    */
    bool InitPresentation(Presentations compatible, Presentations default_presentation, bool read_only = false, bool visible = true);

    /**
    * Answer visibility in GUI.
    *
    * @return GUI visibility
    */
    inline bool IsGUIVisible(void) const {
        return this->visible;
    }

    /**
    * Set visibility in GUI.
    *
    * @param visible True: visible in GUI, false: invisible
    */
    inline void SetGUIVisible(bool visible) {
        this->visible = visible;
    }

    /**
    * Answer accessibility in GUI.
    *
    * @return GUI accessibility
    */
    inline bool IsGUIReadOnly(void) const {
        return this->read_only;
    }

    /**
    * Set accessibility in GUI.
    *
    * @param read_only True: read-only in GUI, false: writable
    */
    inline void SetGUIReadOnly(bool read_only) {
        this->read_only = read_only;
    }     
    
    /**
    * Set presentation of parameter in GUI.
    *
    * @param presentation Presentation of parameter in GUI.
    *
    * @return True if given presentation is compatible, false otherwise.
    */
    bool SetGUIPresentation(Presentations presentation);
    
    /**
    * Answer parameter presentation in GUI.
    *
    * @return Parameter presentation.
    */
    inline Presentations GetGUIPresentation(void) const {
        return this->presentation;
    }

    /**
    * Answer whether given presentation is compatible with parameter.
    *
    * @return True if given presentation is compatible, false otherwise.
    */
    inline bool IsPresentationCompatible(Presentations presentation) const {
        return (Presentations::NONE != (presentation & this->compatible));
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
    Presentations presentation;
    
    /* Compatible presentations */
    Presentations compatible;

    /* Falg ensuring that initialisation can only be applied once. */
    bool initialised;
};

} /* end namespace param */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_ABSTRACTPARAMPRESENTATION_H_INCLUDED */
