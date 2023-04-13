/*
 * Listenable.h
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#pragma once
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/ConstIterator.h"
#include "vislib/Iterator.h"
#include "vislib/SingleLinkedList.h"


namespace vislib {


/**
 * Base class for listenable classes
 *
 * Template parameter C is the derived class to make the listeners classes
 * a bit type-safe
 */
template<class C>
class Listenable {
public:
    /**
     * Base class for listeners classes
     */
    class Listener {
    public:
        /** Ctor */
        Listener();

        /** Dtor */
        virtual ~Listener();
    };

    /** Iterator type */
    typedef typename SingleLinkedList<Listener*>::Iterator ListenerIterator;

    /** Const iterator type */
    typedef ConstIterator<ListenerIterator> ListenerConstIterator;

    /** Ctor. */
    Listenable();

    /** Dtor. */
    ~Listenable();

    /**
     * Adds a listener to this object. The ownership of the memory of the
     * listener will not change. The caller must ensure the pointer to the
     * listener remains valid as long as it is used. If 'listener' is
     * already a listener of this object, nothing changes.
     *
     * @param listener Pointer to the new listener
     */
    void AddListener(Listener* listener);

    /**
     * Removes all listeners. Does not delete the listener objects since
     * their memory is not owned by this object.
     */
    void ClearListeners();

    /**
     * answer wether 'listener' is a listener of this object
     *
     * @param listener The listener object to test
     *
     * @return True if 'listener' is a listener of this object
     */
    bool IsListener(Listener* listener) const;

    /**
     * Removes a listener from this object. Does not delete the listener
     * object since its memory is not owned by this object.
     */
    void RemoveListener(Listener* listener);

protected:
    /**
     * Answer a const iterator of all listeners of this object
     *
     * @return A const iterator of all listeners of this object
     */
    ListenerConstIterator GetConstListeners() const;

    /**
     * Answer an iterator of all listeners of this object
     *
     * @return An iterator of all listeners of this object
     */
    ListenerIterator GetListeners();

private:
    /** The list of listeners of this object */
    SingleLinkedList<Listener*> listeners;
};


/*
 * Listenable<C>::Listener::Listener
 */
template<class C>
Listenable<C>::Listener::Listener() {
    // intentionally empty
}


/*
 * Listenable<C>::Listener::~Listener
 */
template<class C>
Listenable<C>::Listener::~Listener() {
    // intentionally empty
}


/*
 * Listenable<C>::Listenable
 */
template<class C>
Listenable<C>::Listenable() : listeners() {
    // intentionally empty
}


/*
 * Listenable<C>::~Listenable
 */
template<class C>
Listenable<C>::~Listenable() {
    this->listeners.Clear(); // DO NOT DELETE Items
}


/*
 * Listenable<C>::AddListener
 */
template<class C>
void Listenable<C>::AddListener(Listener* listener) {
    if (!this->listeners.Contains(listener)) {
        this->listeners.Add(listener);
    }
}


/*
 * Listenable<C>::ClearListeners
 */
template<class C>
void Listenable<C>::ClearListeners() {
    this->listeners.Clear(); // DO NOT DELETE Items
}


/*
 * Listenable<C>::IsListener
 */
template<class C>
bool Listenable<C>::IsListener(Listener* listener) const {
    return this->listeners.Contains(listener);
}


/*
 * Listenable<C>::RemoveListener
 */
template<class C>
void Listenable<C>::RemoveListener(Listener* listener) {
    this->listeners.RemoveAll(listener);
}


/*
 * Listenable<C>::GetConstListeners
 */
template<class C>
typename Listenable<C>::ListenerConstIterator Listenable<C>::GetConstListeners() const {
    return this->listeners.GetConstIterator();
}


/*
 * Listenable<C>::GetListeners
 */
template<class C>
typename Listenable<C>::ListenerIterator Listenable<C>::GetListeners() {
    return this->listeners.GetIterator();
}


} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
