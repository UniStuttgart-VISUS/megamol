/*
 * ModuleNamespace.h
 *
 * Copyright (C) 2009 - 2015 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_MODULENAMESPACE_H_INCLUDED
#define MEGAMOLCORE_MODULENAMESPACE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/AbstractNamedObjectContainer.h"


namespace megamol {
namespace core {

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
    virtual ~ModuleNamespace(void);

    /**
     * Clears the cleanup mark for this and all dependent objects.
     */
    virtual void ClearCleanupMark(void);

protected:
private:
};


} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_MODULENAMESPACE_H_INCLUDED */
