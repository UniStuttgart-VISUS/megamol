/**
 * MegaMol
 * Copyright (c) 2009, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "mmcore/ModuleNamespace.h"
#include "vislib/Array.h"
#include "vislib/RawStorage.h"
//#include "vislib/sys/FatReaderWriterLock.h"
#include "vislib/sys/ReaderWriterMutexWrapper.h"
#if defined(DEBUG) || defined(_DEBUG)
//#include "vislib/sys/CriticalSection.h"
#include "vislib/SingleLinkedList.h"
#endif

namespace megamol::core {

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
    RootModuleNamespace();

    /**
     * Dtor.
     */
    virtual ~RootModuleNamespace();

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
    virtual vislib::sys::AbstractReaderWriterLock& ModuleGraphLock();

    /**
     * Answer the reader-writer lock to lock the module graph
     *
     * @return The reader-writer lock to lock the module graph
     */
    virtual vislib::sys::AbstractReaderWriterLock& ModuleGraphLock() const;

protected:
private:
    /** The graph access synchronization object */
    //mutable vislib::sys::FatReaderWriterLock lock;
    mutable vislib::sys::ReaderWriterMutexWrapper lock;
};

} // namespace megamol::core
