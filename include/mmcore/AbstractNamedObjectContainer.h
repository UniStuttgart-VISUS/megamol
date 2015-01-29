/*
 * AbstractNamedObjectContainer.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_ABSTRACTNAMEDOBJECTCONTAINER_H_INCLUDED
#define MEGAMOLCORE_ABSTRACTNAMEDOBJECTCONTAINER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "api/MegaMolCore.std.h"
#include "AbstractNamedObject.h"
#include "vislib/ConstIterator.h"
#include "vislib/SingleLinkedList.h"


namespace megamol {
namespace core {


    /**
     * Abstract base class for object placed in the module network namespaces
     */
    class MEGAMOLCORE_API AbstractNamedObjectContainer: public AbstractNamedObject {
    public:

        /** Type of single linked list of children. */
        typedef vislib::SingleLinkedList<AbstractNamedObject*> ChildList;

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
        inline void AddChild(AbstractNamedObject *child) {
            this->addChild(child);
        }

        /**
         * Removes a child from the list of children.
         *
         * @param child The child to be removed.
         */
        inline void RemoveChild(AbstractNamedObject *child) {
            this->removeChild(child);
        }

        /**
         * Finds the child with the given name.
         *
         * @param name The name to search for.
         *
         * @return The found child or NULL if there is no child with this name.
         */
        inline AbstractNamedObject *FindChild(const vislib::StringA& name) {
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
        AbstractNamedObject *FindNamedObject(const char *name, bool forceRooted = false);

        /**
         * Answer an iterator of the children.
         *
         * @return An iterator of the children.
         */
        inline ChildList::Iterator GetChildIterator(void) {
            return this->getChildIterator();
        }

        /**
         * Answer a const iterator of the children.
         *
         * @return A const iterator of the children.
         */
        inline vislib::ConstIterator<ChildList::Iterator>
        GetConstChildIterator(void) const {
            return this->getConstChildIterator();
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
        virtual bool IsParamRelevant(
            vislib::SingleLinkedList<const AbstractNamedObject*>& searched,
            const vislib::SmartPtr<param::AbstractParam>& param) const;

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
        void addChild(AbstractNamedObject *child);

        /**
         * Removes a child from the list of children.
         *
         * @param child The child to be removed.
         */
        void removeChild(AbstractNamedObject *child);

        /**
         * Answer an iterator of the children.
         *
         * @return An iterator of the children.
         */
        ChildList::Iterator getChildIterator(void);

        /**
         * Answer a const iterator of the children.
         *
         * @return A const iterator of the children.
         */
        vislib::ConstIterator<ChildList::Iterator>
        getConstChildIterator(void) const;

        /**
         * Finds the child with the given name.
         *
         * @param name The name to search for.
         *
         * @return The found child or NULL if there is no child with this name.
         */
        AbstractNamedObject *findChild(const vislib::StringA& name);

    private:
#ifdef _WIN32
#pragma warning (disable: 4251)
#endif /* _WIN32 */
        /** The children of the container */
        ChildList children;
#ifdef _WIN32
#pragma warning (default: 4251)
#endif /* _WIN32 */

    };


} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_ABSTRACTNAMEDOBJECTCONTAINER_H_INCLUDED */
