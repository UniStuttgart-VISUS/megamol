/**
 * MegaMol
 * Copyright (c) 2006-2021, MegaMol Dev Team
 * All rights reserved.
 */

#ifndef MEGAMOLCORE_FACTORIES_OBJECTDESCRIPTIONMANAGER_H_INCLUDED
#define MEGAMOLCORE_FACTORIES_OBJECTDESCRIPTIONMANAGER_H_INCLUDED
#pragma once

#include <algorithm>
#include <cctype>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
bool strEqualIgnoreCase(const std::string& a, const std::string& b) {
    return std::equal(a.begin(), a.end(), b.begin(), b.end(),
        [](const char& a, const char& b) { return std::tolower(a) == std::tolower(b); });
}
} // namespace

namespace megamol::core::factories {

/**
 * Class template for managing object descriptions.
 * Template parameter T is a class derived from 'ObjectDescription'.
 */
template<class T>
class ObjectDescriptionManager {
public:
    typedef std::shared_ptr<const T> description_ptr_type;
    typedef std::vector<description_ptr_type> description_list_type;
    typedef typename description_list_type::iterator description_iterator_type;
    typedef typename description_list_type::const_iterator description_const_iterator_type;

    /** ctor */
    ObjectDescriptionManager();

    /** dtor */
    virtual ~ObjectDescriptionManager();

    /* deleted copy ctor */
    ObjectDescriptionManager(const ObjectDescriptionManager& other) = delete;

    /* deleted assigmnet operator */
    ObjectDescriptionManager& operator=(const ObjectDescriptionManager& other) = delete;

    /**
     * Registers an object description.
     *
     * @param objDesc A pointer to the object description object to be
     *                registered.
     *
     * @throws std::invalid_argument if there already is an object
     *         registered with the same name (case insensitive).
     */
    void Register(description_ptr_type objDesc);

    /**
     * Unregisters an object. It is safe to unregister an object never
     * registered.
     *
     * @param classname The name of the object to unregister.
     */
    void Unregister(const char* classname);

    /**
     * Answers the number of descriptions stored.
     *
     * @return The number of descriptions stored
     */
    unsigned int Count() const;

    /**
     * Removes all descriptions from the manager in preparation of the
     * shutdown.
     */
    void Shutdown();

    /**
     * Searches for an object description object with the given name.
     *
     * @param classname The class name to search for.
     *
     * @return The found object description object or NULL if the name is
     *         not found.
     */
    virtual description_ptr_type Find(const char* classname) const;

    /**
     * Gets an iterator over the registered descriptions.
     *
     * @return The iterator object
     */
    description_iterator_type begin();

    /**
     * Gets an iterator over the registered descriptions.
     *
     * @return The iterator object
     */
    description_iterator_type end();


    /**
     * Gets an iterator over the registered descriptions.
     *
     * @return The iterator object
     */
    description_const_iterator_type begin() const;

    /**
     * Gets an iterator over the registered descriptions.
     *
     * @return The iterator object
     */
    description_const_iterator_type end() const;

private:
    /** The registered object descriptions */
    description_list_type descriptions_;
};

/*
 * ObjectDescriptionManager::ObjectDescriptionManager
 */
template<class T>
ObjectDescriptionManager<T>::ObjectDescriptionManager() : descriptions_() {}

/*
 * ObjectDescriptionManager::~ObjectDescriptionManager
 */
template<class T>
ObjectDescriptionManager<T>::~ObjectDescriptionManager() {
    descriptions_.clear();
}

/*
 * ObjectDescriptionManager<T>::Register
 */
template<class T>
void ObjectDescriptionManager<T>::Register(description_ptr_type objDesc) {
    if (!objDesc)
        throw std::runtime_error("No object description given!");
    if (this->Find(objDesc->ClassName()) != nullptr) {
        throw std::invalid_argument("Class name of object description is already registered!");
    }
    descriptions_.push_back(objDesc);
}

/*
 * ObjectDescriptionManager<T>::Unregister
 */
template<class T>
void ObjectDescriptionManager<T>::Unregister(const char* classname) {
    std::string name(classname);
    descriptions_.erase(
        std::remove_if(descriptions_.begin(), descriptions_.end(),
            [&name](const description_ptr_type& d) { return strEqualIgnoreCase(name, d->ClassName()); }),
        descriptions_.end());
}

/*
 * ObjectDescriptionManager<T>::Count
 */
template<class T>
unsigned int ObjectDescriptionManager<T>::Count() const {
    return static_cast<unsigned int>(descriptions_.size());
}

/*
 * ObjectDescriptionManager<T>::Shutdown
 */
template<class T>
void ObjectDescriptionManager<T>::Shutdown() {
    descriptions_.clear();
}

/*
 * ObjectDescriptionManager<T>::Find
 */
template<class T>
typename ObjectDescriptionManager<T>::description_ptr_type ObjectDescriptionManager<T>::Find(
    const char* classname) const {
    for (auto& desc : descriptions_) {
        if (strEqualIgnoreCase(std::string(classname), std::string(desc->ClassName()))) {
            return desc;
        }
    }
    return nullptr;
}

/*
 * ObjectDescriptionManager<T>::begin
 */
template<class T>
typename ObjectDescriptionManager<T>::description_iterator_type ObjectDescriptionManager<T>::begin() {
    return descriptions_.begin();
}

/*
 * ObjectDescriptionManager<T>::end
 */
template<class T>
typename ObjectDescriptionManager<T>::description_iterator_type ObjectDescriptionManager<T>::end() {
    return descriptions_.end();
}

/*
 * ObjectDescriptionManager<T>::begin
 */
template<class T>
typename ObjectDescriptionManager<T>::description_const_iterator_type ObjectDescriptionManager<T>::begin() const {
    return descriptions_.begin();
}

/*
 * ObjectDescriptionManager<T>::end
 */
template<class T>
typename ObjectDescriptionManager<T>::description_const_iterator_type ObjectDescriptionManager<T>::end() const {
    return descriptions_.end();
}
} // namespace megamol::core::factories

#endif // MEGAMOLCORE_FACTORIES_OBJECTDESCRIPTIONMANAGER_H_INCLUDED
