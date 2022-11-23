/**
 * MegaMol
 * Copyright (c) 2009, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "factories/CallDescription.h"
#include "mmcore/AbstractNamedObject.h"
#include "vislib/SingleLinkedList.h"
#include "vislib/String.h"

namespace megamol::core {

/**
 * Abstract base class for all module slots.
 *
 * Note that the owner is not set in the ctor for a very good reason: If
 * the slot is a non-pointer member of the module class, initialising the
 * owner in the ctor would require the use of 'this' in the initialisation
 * list, which is not allowed.
 */
class AbstractSlot : public AbstractNamedObject {
public:
    /**
     * interface class for listener objects
     */
    class Listener {
    public:
        /**
         * Ctor.
         */
        inline Listener() {
            // intentionally empty
        }

        /**
         * Dtor.
         */
        virtual ~Listener() {
            // intentionally empty
        }

        /**
         * This method is called when an object connects to the slot.
         *
         * @param slot The slot that triggered this event.
         */
        virtual void OnConnect(AbstractSlot& slot) {
            // empty default implementation
        }

        /**
         * This method is called when an object disconnects from the slot.
         *
         * @param slot The slot that triggered this event.
         */
        virtual void OnDisconnect(AbstractSlot& slot) {
            // empty default implementation
        }
    };

    /** Possible status values for the slot */
    enum SlotStatus { STATUS_UNAVAILABLE, STATUS_ENABLED, STATUS_CONNECTED };

    /** Dtor. */
    virtual ~AbstractSlot();

    /**
     * Gets a human readable description of this slot.
     *
     * @return A human readable description of this slot.
     */
    inline const vislib::StringA& Description() const {
        return this->desc;
    }

    /**
     * Answer the status of the slot.
     *
     * @return The status of the slot.
     */
    inline const SlotStatus& GetStatus() const {
        return this->status;
    }

    /**
     * Makes this slot available. After this method was called the
     * settings of the slot can no longer be changed.
     */
    virtual void MakeAvailable();
    virtual void MakeUnavailable();

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

    /**
     * Adds a listener to the list of registered listeners. The caller
     * remains owner of the listener object and must ensure that the
     * object remains valid as long as it is registered at this slot.
     *
     * @param listener Pointer to the listener to be added
     */
    inline void AddListener(Listener* listener) {
        this->listeners.Add(listener);
    }

    /**
     * Removes a listener from the list of registered listeners.
     *
     * @param listener Pointer to the listener to be removed.
     */
    inline void RemoveListener(Listener* listener) {
        this->listeners.Remove(listener);
    }

    /**
     * Answers whether a given call is compatible with this slot.
     *
     * @param desc The description of the call to test.
     *
     * @return 'true' if the call is compatible, 'false' otherwise.
     */
    virtual bool IsCallCompatible(factories::CallDescription::ptr desc) const {
        return false;
    };

protected:
    /**
     * Ctor.
     *
     * @param name The name of this slot.
     * @param desc A human readable description of this slot.
     */
    AbstractSlot(const vislib::StringA& name, const vislib::StringA& desc);

    /**
     * Sets the status of the slot to indicate the status of its
     * connection.
     *
     * @param connected If 'true' the status of the slot will be set to
     *                  indicate that the slot is connected, if 'false'
     *                  the slot is no longer connected.
     */
    void SetStatusConnected(bool connected = true);

    /**
     * Sets the status of the Slot to indicate that the slot is no
     * longer connected.
     */
    inline void SetStatusDisconnected() {
        this->SetStatusConnected(false);
    }

private:
    /** forbidden copy ctor */
    AbstractSlot(const AbstractSlot& src);

    /** forbidden assignment */
    AbstractSlot& operator=(const AbstractSlot& rhs);

    /** A human readable description of the slot */
    vislib::StringA desc;

    /** The status of the slot */
    SlotStatus status;

    /** Counts the connected references */
    unsigned int connectedRefCnt;

    /** List of listeners */
    vislib::SingleLinkedList<Listener*> listeners;
};


} // namespace megamol::core
