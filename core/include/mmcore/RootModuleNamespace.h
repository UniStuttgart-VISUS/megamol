/*
 * RootModuleNamespace.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_ROOTMODULENAMESPACE_H_INCLUDED
#define MEGAMOLCORE_ROOTMODULENAMESPACE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/ModuleNamespace.h"
#include "vislib/Array.h"
#include "vislib/RawStorage.h"
//#include "vislib/sys/FatReaderWriterLock.h"
#include "vislib/sys/ReaderWriterMutexWrapper.h"
#if defined(DEBUG) || defined(_DEBUG)
//#include "vislib/sys/CriticalSection.h"
#include "vislib/SingleLinkedList.h"
#endif


namespace megamol {
namespace core {

/**
 * Class represents the root namespace for the module namespace
 */
class RootModuleNamespace : public ModuleNamespace {
public:
    /** Type alias for containers */
    typedef std::shared_ptr<RootModuleNamespace> ptr_type;

    /** Type alias for containers */
    typedef std::shared_ptr<const RootModuleNamespace> const_ptr_type;

    /**
     * Utility function to dynamically cast to a shared_ptr of this type
     *
     * @param p The shared pointer to cast from
     *
     * @return A shared pointer of this type
     */
    template<class T>
    inline static ptr_type dynamic_pointer_cast(std::shared_ptr<T> p) {
        return std::dynamic_pointer_cast<RootModuleNamespace, T>(p);
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
        return std::dynamic_pointer_cast<const RootModuleNamespace, const T>(p);
    }

    /**
     * Ctor.
     */
    RootModuleNamespace(void);

    /**
     * Dtor.
     */
    virtual ~RootModuleNamespace(void);

    /**
     * Answer the full namespace path for 'path' seen relative to 'base'.
     *
     * @param base The base path
     * @param path The path
     *
     * @return The full namespace path
     */
    vislib::StringA FullNamespace(const vislib::StringA& base, const vislib::StringA& path) const;

    /**
     * Finds the specified module namespace
     *
     * @param path A array with the separated namespace names.
     * @param createMissing If true, all missing namespace object will be created.
     * @param quiet Flag to deactivate error logging
     *
     * @return The requested namespace or 'NULL' in case of an error.
     */
    ModuleNamespace::ptr_type FindNamespace(
        const vislib::Array<vislib::StringA>& path, bool createMissing, bool quiet = false);

    /**
     * Answer the reader-writer lock to lock the module graph
     *
     * @return The reader-writer lock to lock the module graph
     */
    virtual vislib::sys::AbstractReaderWriterLock& ModuleGraphLock(void);

    /**
     * Answer the reader-writer lock to lock the module graph
     *
     * @return The reader-writer lock to lock the module graph
     */
    virtual vislib::sys::AbstractReaderWriterLock& ModuleGraphLock(void) const;

protected:
private:
#ifdef _WIN32
#pragma warning(disable : 4251)
#endif /* _WIN32 */

    /** The graph access synchronization object */
    //mutable vislib::sys::FatReaderWriterLock lock;
    mutable vislib::sys::ReaderWriterMutexWrapper lock;

#ifdef _WIN32
#pragma warning(default : 4251)
#endif /* _WIN32 */
};


} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_ROOTMODULENAMESPACE_H_INCLUDED */
