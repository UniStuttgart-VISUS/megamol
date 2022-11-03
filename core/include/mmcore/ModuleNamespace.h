/**
 * MegaMol
 * Copyright (c) 2009, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "mmcore/AbstractNamedObjectContainer.h"

namespace megamol::core {

/**
 * Class represents a normal module namespace.
 */
class ModuleNamespace : public AbstractNamedObjectContainer {
public:
    /** Type alias for containers */
    typedef std::shared_ptr<ModuleNamespace> ptr_type;

    /** Type alias for containers */
    typedef std::shared_ptr<const ModuleNamespace> const_ptr_type;

    /**
     * Utility function to dynamically cast to a shared_ptr of this type
     *
     * @param p The shared pointer to cast from
     *
     * @return A shared pointer of this type
     */
    template<class T>
    inline static ptr_type dynamic_pointer_cast(std::shared_ptr<T> p) {
        return std::dynamic_pointer_cast<ModuleNamespace, T>(p);
    }

    /**
     * Utility function to dynamically cast to a shared_ptr of this type
     *
     * @param p The shared pointer to cast from
     *
     * @return A shared pointer of this type
     */
    template<class T>
    inline static const_ptr_type dynamic_pointer_cast(std::shared_ptr<const T> p) {
        return std::dynamic_pointer_cast<const ModuleNamespace, const T>(p);
    }

    /**
     * Ctor.
     *
     * @param name The name for the namespace
     */
    ModuleNamespace(const vislib::StringA& name);

    /**
     * Dtor.
     */
    virtual ~ModuleNamespace();

    /**
     * Clears the cleanup mark for this and all dependent objects.
     */
    virtual void ClearCleanupMark();

protected:
private:
};

} // namespace megamol::core
