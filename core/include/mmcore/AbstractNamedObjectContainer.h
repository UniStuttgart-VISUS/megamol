/**
 * MegaMol
 * Copyright (c) 2009, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <list>
#include <memory>

#include "mmcore/AbstractNamedObject.h"
#include "vislib/macro_utils.h"

namespace megamol::core {

/**
 * Abstract base class for object placed in the module network namespaces
 */
class AbstractNamedObjectContainer : public AbstractNamedObject {
public:
    /** Type alias for containers */
    typedef std::shared_ptr<AbstractNamedObjectContainer> ptr_type;

    /** Type alias for containers */
    typedef std::shared_ptr<const AbstractNamedObjectContainer> const_ptr_type;

    /** Type of single linked list of children. */
    typedef std::list<AbstractNamedObject::ptr_type> child_list_type;

    /**
     * Utility function to dynamically cast to a shared_ptr of this type
     *
     * @param p The shared pointer to cast from
     *
     * @return A shared pointer of this type
     */
    template<class T>
    inline static ptr_type dynamic_pointer_cast(std::shared_ptr<T> p) {
        return std::dynamic_pointer_cast<AbstractNamedObjectContainer, T>(p);
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
        return std::dynamic_pointer_cast<const AbstractNamedObjectContainer, const T>(p);
    }

    /**
     * Dtor.
     */
    virtual ~AbstractNamedObjectContainer(void);

    /**
     * Adds a child to the list of children. The child object must not
     * have a parent. The memory ownership of the child object is not
     * changed.
     *
     * @param child The child to be added.
     */
    inline void AddChild(AbstractNamedObject::ptr_type child) {
        this->addChild(child);
    }

    /**
     * Removes a child from the list of children.
     *
     * @param child The child to be removed.
     */
    inline void RemoveChild(AbstractNamedObject::ptr_type child) {
        this->removeChild(child);
    }

    /**
     * Finds the child with the given name.
     *
     * @param name The name to search for.
     *
     * @return The found child or NULL if there is no child with this name.
     */
    inline AbstractNamedObject::ptr_type FindChild(const vislib::StringA& name) {
        return this->findChild(name);
    }

    /**
     * Finds a module in the module graph
     *
     * @param name The name to search for
     * @param forceRooted Start search at the root of the module graph
     *
     * @return The found object or NULL if no module matches
     */
    AbstractNamedObject::ptr_type FindNamedObject(const char* name, bool forceRooted = false);

    /**
     * Answer an interator to the first child
     *
     * @return An iterator to the first child
     */
    inline child_list_type::iterator ChildList_Begin() {
        return this->children.begin();
    }

    /**
     * Answer an interator to the first child
     *
     * @return An iterator to the first child
     */
    inline child_list_type::const_iterator ChildList_Begin() const {
        return this->children.begin();
    }

    /**
     * Answer an iterator behind the last child
     *
     * @return An iterator behind the last child
     */
    inline child_list_type::iterator ChildList_End() {
        return this->children.end();
    }

    /**
     * Answer an iterator behind the last child
     *
     * @return An iterator behind the last child
     */
    inline child_list_type::const_iterator ChildList_End() const {
        return this->children.end();
    }

    /**
     * Sets the cleanup mark and all marks of all children
     */
    virtual void SetAllCleanupMarks(void);

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
    bool IsParamRelevant(vislib::SingleLinkedList<const AbstractNamedObject*>& searched,
        const std::shared_ptr<param::AbstractParam>& param) const override;

protected:
    /**
     * Ctor.
     */
    AbstractNamedObjectContainer(void);

    /**
     * Adds a child to the list of children. The child object must not
     * have a parent. The memory ownership of the child object is not
     * changed.
     *
     * @param child The child to be added.
     */
    void addChild(AbstractNamedObject::ptr_type child);

    /**
     * Removes a child from the list of children.
     *
     * @param child The child to be removed.
     */
    void removeChild(AbstractNamedObject::ptr_type child);

    /**
     * Finds the child with the given name.
     *
     * @param name The name to search for.
     *
     * @return The found child or NULL if there is no child with this name.
     */
    AbstractNamedObject::ptr_type findChild(const vislib::StringA& name);

    /**
     * Ensures that all children correctly reference their parent
     */
    void fixParentBackreferences(void);

private:
    /** The children of the container */
    VISLIB_MSVC_SUPPRESS_WARNING(4251)
    child_list_type children;
};

} // namespace megamol::core
