/*
 * ObjectDescriptionManager.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_OBJECTDESCRIPTIONMANAGER_H_INCLUDED
#define MEGAMOLCORE_OBJECTDESCRIPTIONMANAGER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "vislib/ConstIterator.h"
#include "vislib/SingleLinkedList.h"
#include "vislib/String.h"
#include "vislib/IllegalParamException.h"


namespace megamol {
namespace core {

    /**
     * Singelton class template for managing object descriptions.
     * Template parameter T is a class derived from 'ObjectDescription'.
     */
    template<class T> class ObjectDescriptionManager {
    public:

        /** Type to iterate descriptions */
        typedef vislib::ConstIterator<typename vislib::SingleLinkedList<T*>::Iterator> DescriptionIterator;

        /** ctor */
        ObjectDescriptionManager(void);

        /** dtor */
        virtual ~ObjectDescriptionManager(void);

        /**
         * Searches for an object description object with the given name.
         *
         * @param classname The class name to search for.
         *
         * @return The found object description object or NULL if the name is
         *         not found.
         */
        virtual T *Find(const char *classname) const;

        /**
         * Gets an iterator over the registered descriptions.
         *
         * @return The iterator object
         */
        inline DescriptionIterator GetIterator(void) const {
            return this->descriptions.GetConstIterator();
        }

        /**
         * Registers an object description.
         *
         * @param objDesc A Pointer to the object description object to be
         *                registered. The memory ownership of this object is
         *                taken by the manager, thus the object must not be
         *                manipulated (freed) by the caller anymore.
         *
         * @throws vislib::IllegalParamException if there already is an object
         *         registered with the same name (case insensitive).
         */
        void Register(T *objDesc);

        /**
         * Unregisteres an object. It is safe to unregister an object never
         * registered.
         *
         * @param classname The name of the object to unregister.
         */
        void Unregister(const char *classname);

    private:

        /** The registered object descriptions */
        vislib::SingleLinkedList<T*> descriptions;

    };


    /*
     * ObjectDescriptionManager::ObjectDescriptionManager
     */
    template<class T>
    ObjectDescriptionManager<T>::ObjectDescriptionManager(void)
            : descriptions() {
        // intentionally empty
    }


    /*
     * ObjectDescriptionManager::~ObjectDescriptionManager
     */
    template<class T>
    ObjectDescriptionManager<T>::~ObjectDescriptionManager(void) {
        typename vislib::SingleLinkedList<T*>::Iterator iter
            = this->descriptions.GetIterator();
        while (iter.HasNext()) {
            delete iter.Next();
        }
        this->descriptions.Clear();
    }


    /*
     * ObjectDescriptionManager<T>::Register
     */
    template<class T>
    T* ObjectDescriptionManager<T>::Find(const char *classname) const {
        vislib::StringA nameA(classname);
        typename vislib::SingleLinkedList<T*>::Iterator iter
            = const_cast<ObjectDescriptionManager<T>*>(this)
            ->descriptions.GetIterator();
        while (iter.HasNext()) {
            T* obj = iter.Next();
            if (nameA.Equals(obj->ClassName(), false)) {
                return obj;
            }
        }
        return NULL;
    }


    /*
     * ObjectDescriptionManager<T>::Register
     */
    template<class T>
    void ObjectDescriptionManager<T>::Register(T *objDesc) {

        if (this->Find(objDesc->ClassName()) != NULL) {
            throw vislib::IllegalParamException("objDesc", __FILE__, __LINE__);
        }
        this->descriptions.Add(objDesc);
    }


    /*
     * ObjectDescriptionManager<T>::Unregister
     */
    template<class T>
    void ObjectDescriptionManager<T>::Unregister(const char *classname) {
        vislib::StringA nameA(classname);
        typename vislib::SingleLinkedList<T*>::Iterator iter
            = this->descriptions.GetIterator();
        while (iter.HasNext()) {
            T* obj = iter.Next();
            if (nameA.Equals(obj->ClassName(), false)) {
                this->descriptions.Remove(iter);
                delete obj;
                return;
            }
        }
    }


} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_OBJECTDESCRIPTIONMANAGER_H_INCLUDED */
