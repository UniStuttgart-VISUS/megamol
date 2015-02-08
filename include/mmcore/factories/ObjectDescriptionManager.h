/*
 * ObjectDescriptionManager.h
 *
 * Copyright (C) 2006 - 2015 by MegaMol Consortium
 * All rights reserved. Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_FACTORIES_OBJECTDESCRIPTIONMANAGER_H_INCLUDED
#define MEGAMOLCORE_FACTORIES_OBJECTDESCRIPTIONMANAGER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include <vector>
#include <memory>
#include <algorithm>
#include "vislib/IllegalParamException.h"
#include "vislib/macro_utils.h"


namespace megamol {
namespace core {
namespace factories {

    /**
     * Singelton class template for managing object descriptions.
     * Template parameter T is a class derived from 'ObjectDescription'.
     */
    template<class T> class ObjectDescriptionManager {
    public:

        typedef T description_type;
        typedef ::std::shared_ptr<const T> description_ptr_type;
        typedef ::std::vector<description_ptr_type> description_list_type;
        typedef typename description_list_type::iterator description_iterator_type;
        typedef typename description_list_type::const_iterator description_const_iterator_type;

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
        virtual description_ptr_type Find(const char *classname) const;

        /**
         * Gets an iterator over the registered descriptions.
         *
         * @return The iterator object
         */
        description_iterator_type begin(void);

        /**
         * Gets an iterator over the registered descriptions.
         *
         * @return The iterator object
         */
        description_iterator_type end(void);


        /**
         * Gets an iterator over the registered descriptions.
         *
         * @return The iterator object
         */
        description_const_iterator_type begin(void) const;

        /**
         * Gets an iterator over the registered descriptions.
         *
         * @return The iterator object
         */
        description_const_iterator_type end(void) const;

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
        void Register(description_ptr_type objDesc);

        /**
         * Unregisteres an object. It is safe to unregister an object never
         * registered.
         *
         * @param classname The name of the object to unregister.
         */
        void Unregister(const char *classname);

        /**
         * Answers the number of descriptions stored.
         *
         * @return The number of descriptions stored
         */
        unsigned int Count(void) const;

        /**
         * Removes all descriptions from the manager in preparation of the
         * shutdown.
         */
        void Shutdown(void);

    private:

        /* deleted copy ctor */
        ObjectDescriptionManager(const ObjectDescriptionManager& srv) = delete;

        /* deleted assigmnet operator */
        ObjectDescriptionManager& operator=(const ObjectDescriptionManager& rhs) = delete;

        /** The registered object descriptions */
        VISLIB_MSVC_SUPPRESS_WARNING(4251)
        description_list_type descriptions;

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
        this->descriptions.clear();
    }


    /*
     * ObjectDescriptionManager<T>::Register
     */
    template<class T> 
    typename ObjectDescriptionManager<T>::description_ptr_type
    ObjectDescriptionManager<T>::Find(const char *classname) const {
        vislib::StringA nameA(classname);
        auto d_end = this->descriptions.cend();
        for (auto d_i = this->descriptions.cbegin(); d_i != d_end; ++d_i) {
            if (nameA.Equals((*d_i)->ClassName(), false)) {
                return *d_i;
            }
        }
        return nullptr;
    }


    /*
     * ObjectDescriptionManager<T>::begin
     */
    template<class T> 
    typename ObjectDescriptionManager<T>::description_iterator_type
    ObjectDescriptionManager<T>::begin(void) {
        return this->descriptions.begin();
    }


    /*
     * ObjectDescriptionManager<T>::end
     */
    template<class T> 
    typename ObjectDescriptionManager<T>::description_iterator_type
    ObjectDescriptionManager<T>::end(void) {
        return this->descriptions.end();
    }


    /*
     * ObjectDescriptionManager<T>::begin
     */
    template<class T> 
    typename ObjectDescriptionManager<T>::description_const_iterator_type
    ObjectDescriptionManager<T>::begin(void) const {
        return this->descriptions.begin();
    }


    /*
     * ObjectDescriptionManager<T>::end
     */
    template<class T> 
    typename ObjectDescriptionManager<T>::description_const_iterator_type
    ObjectDescriptionManager<T>::end(void) const {
        return this->descriptions.end();
    }


    /*
     * ObjectDescriptionManager<T>::Register
     */
    template<class T>
    void ObjectDescriptionManager<T>::Register(description_ptr_type objDesc) {
        if (!objDesc) throw vislib::IllegalParamException("objDesc", __FILE__, __LINE__);
        if (this->Find(objDesc->ClassName()) != nullptr) {
            throw vislib::IllegalParamException("objDesc", __FILE__, __LINE__);
        }
        this->descriptions.push_back(objDesc);
    }


    /*
     * ObjectDescriptionManager<T>::Unregister
     */
    template<class T>
    void ObjectDescriptionManager<T>::Unregister(const char *classname) {
        vislib::StringA nameA(classname);
        this->descriptions.erase(
            ::std::remove_if(this->descriptions.begin(), this->descriptions.end(),
            [&nameA](const description_ptr_type& d) {
                return nameA.Equals(d->ClassName(), false);
            }),
            this->descriptions.end());
    }


    /*
     * ObjectDescriptionManager<T>::Count
     */
    template<class T>
    unsigned int ObjectDescriptionManager<T>::Count(void) const {
        return static_cast<unsigned int>(this->descriptions.size());
    }


    /*
     * ObjectDescriptionManager<T>::Shutdown
     */
    template<class T>
    void ObjectDescriptionManager<T>::Shutdown(void) {
        this->descriptions.clear();
    }


} /* end namespace factories */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_FACTORIES_OBJECTDESCRIPTIONMANAGER_H_INCLUDED */
