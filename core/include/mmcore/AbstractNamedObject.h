/**
 * MegaMol
 * Copyright (c) 2009, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <memory>

#include "mmcore/param/AbstractParam.h"
#include "vislib/SingleLinkedList.h"
#include "vislib/SmartPtr.h"
#include "vislib/String.h"
#include "vislib/sys/AbstractReaderWriterLock.h"
#include "vislib/sys/SyncObject.h"

namespace megamol::core {

// forward declarations of types
class AbstractNamedObjectContainer;

/**
 * Abstract base class for object placed in the module network namespaces
 */
class AbstractNamedObject : public std::enable_shared_from_this<AbstractNamedObject> {
public:
    friend class ::megamol::core::AbstractNamedObjectContainer;

    /** Shared ptr type alias */
    typedef ::std::shared_ptr<AbstractNamedObject> ptr_type;

    /** Shared ptr type alias */
    typedef ::std::shared_ptr<const AbstractNamedObject> const_ptr_type;

    /** Weak ptr type alias */
    typedef ::std::weak_ptr<AbstractNamedObject> weak_ptr_type;

    /**
     * Utility class
     */
    class GraphLocker : public vislib::sys::SyncObject {
    public:
        /**
         * Ctor.
         *
         * @param obj Any object from the module graph
         * @param writelock Flag if a write lock is required
         */
        GraphLocker(AbstractNamedObject::const_ptr_type obj, bool writelock);

        /**
         * Dtor.
         */
        virtual ~GraphLocker(void);

        /**
         * Acquire the lock.
         *
         * @throws SystemException If the lock could not be acquired.
         */
        virtual void Lock(void);

        /**
         * Release the lock.
         *
         * @throw SystemException If the lock could not be released.
         */
        virtual void Unlock(void);

    private:
        /** Flag if a write lock is required */
        bool writelock;

        /** The root object of the graph */
        AbstractNamedObject::const_ptr_type root;
    };

    /**
     * Dtor.
     */
    virtual ~AbstractNamedObject(void);

    /**
     * Answer the full name of the object
     *
     * @return The full name of the object
     */
    vislib::StringA FullName(void) const;

    /**
     * Answer the name of the object.
     *
     * @return The name of the object.
     */
    inline const vislib::StringA& Name(void) const {
        return this->name;
    }

    /**
     * Answer the owner identification pointer.
     *
     * @return The owner identification pointer.
     */
    inline const void* Owner(void) const {
        return this->owner;
    }

    /**
     * Answer the parent of the object.
     *
     * @return The parent of the object.
     */
    inline ptr_type Parent(void) {
        return this->parent.lock();
    }

    /**
     * Answer the parent of the object.
     *
     * @return The parent of the object.
     */
    inline const const_ptr_type Parent(void) const {
        return this->parent.lock();
    }

    /**
     * Answers the root of the module graph.
     *
     * @return The root of the module graph
     */
    inline ptr_type RootModule(void) {
        ptr_type rv = this->shared_from_this();
        ptr_type lrv = rv;
        while (rv) {
            lrv = rv;
            rv = rv->parent.lock();
        }
        return lrv;
    }

    /**
     * Answers the root of the module graph.
     *
     * @return The root of the module graph
     */
    inline const const_ptr_type RootModule(void) const {
        const_ptr_type rv = this->shared_from_this();
        const_ptr_type lrv = rv;
        while (rv) {
            lrv = rv;
            rv = rv->parent.lock();
        }
        return lrv;
    }

    /**
     * Sets the owner identification pointer. This can only be done once.
     *
     * @param owner The owner identification pointer.
     */
    void SetOwner(void* owner);

    /**
     * Gets the cleanup mark
     *
     * @return The cleanup mark
     */
    inline bool CleanupMark(void) const {
        return this->cleanupMark;
    }

    /**
     * Sets the cleanup mark to the given value
     *
     * @param value The new value for the cleanup mark
     */
    inline void SetCleanupMark(bool value) {
        this->cleanupMark = value;
    }

    /**
     * Sets the cleanup mark
     */
    virtual void SetAllCleanupMarks(void);

    /**
     * Clears the cleanup mark for this and all dependent objects.
     */
    virtual void ClearCleanupMark(void);

    /**
     * Performs the cleanup operation by removing and deleteing of all
     * marked objects.
     */
    virtual void PerformCleanup(void);

    /**
     * Disconnects calls from all slots which are marked for cleanup.
     */
    virtual void DisconnectCalls(void);

    /**
     * Answers whether the given parameter is relevant for this view.
     *
     * @param searched The already searched objects for cycle detection.
     * @param param The parameter to test.
     *
     * @return 'true' if 'param' is relevant, 'false' otherwise.
     */
    virtual bool IsParamRelevant(vislib::SingleLinkedList<const AbstractNamedObject*>& searched,
        const std::shared_ptr<param::AbstractParam>& param) const;

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
    /**
     * Helper method testing if the string name would be a valid object
     * name. Valid object names must not contain ':' characters.
     *
     * @param name The string to be tested.
     *
     * @return 'true' if name would be a valid object name,
     *         'false' otherwise.
     */
    static bool isNameValid(const vislib::StringA& name);

public: // for new MegaMolGraph, make this public:
    /**
     * Sets the name for the object.
     *
     * @param name The new name for the object.
     */
    void setName(const vislib::StringA& name);

    /**
     * Ctor.
     */
    AbstractNamedObject(void);

    /**
     * Sets the parent for the object.
     *
     * @param parent The new parent for the object.
     */
    void setParent(weak_ptr_type parent);

private:
#ifdef _WIN32
#pragma warning(disable : 4251)
#endif /* _WIN32 */
    /** the name of the object */
    vislib::StringA name;
#ifdef _WIN32
#pragma warning(default : 4251)
#endif /* _WIN32 */

    /** The parent of the object. Weak reference, do not delete. */
    weak_ptr_type parent;

    /** The identification of the owner of the object */
    const void* owner;

    /** The cleanup mark of this object */
    bool cleanupMark;
};


} // namespace megamol::core
