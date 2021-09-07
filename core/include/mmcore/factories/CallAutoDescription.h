/**
 * MegaMol
 * Copyright (c) 2008-2021, MegaMol Dev Team
 * All rights reserved.
 */

#ifndef MEGAMOLCORE_FACTORIES_CALLAUTODESCRIPTION_H_INCLUDED
#define MEGAMOLCORE_FACTORIES_CALLAUTODESCRIPTION_H_INCLUDED
#pragma once

#include "CallDescription.h"

namespace megamol::core::factories {

    /**
     * Description class for the call T
     */
    template<class T>
    class CallAutoDescription : public CallDescription {
    public:
        /** Ctor. */
        CallAutoDescription() : CallDescription() {}

        /** Dtor. */
        ~CallAutoDescription() override = default;

        /**
         * Answer the name of the objects of this description.
         *
         * @return The name of the objects of this description.
         */
        const char* ClassName() const override {
            return T::ClassName();
        }

        /**
         * Gets a human readable description of the module.
         *
         * @return A human readable description of the module.
         */
        const char* Description() const override {
            return T::Description();
        }

        /**
         * Creates a new call object.
         *
         * @return The newly created call object.
         */
        Call* CreateCall() const override {
            T* c = new T();
            c->SetClassName(this->ClassName());
            return this->describeCall(c);
        }

        /**
         * Answer the number of functions used for this call.
         *
         * @return The number of functions used for this call.
         */
        unsigned int FunctionCount() const override {
            return T::FunctionCount();
        }

        /**
         * Answer the name of the function used for this call.
         *
         * @param idx The index of the function to return it's name.
         *
         * @return The name of the requested function.
         */
        const char* FunctionName(unsigned int idx) const override {
            return T::FunctionName(idx);
        }

        /**
         * Answers whether this description is describing the class of
         * 'call'.
         *
         * @param call The call to test.
         *
         * @return 'true' if 'call' is described by this description,
         *         'false' otherwise.
         */
        bool IsDescribing(const Call* call) const override {
            return dynamic_cast<const T*>(call) != nullptr;
        }

        /**
         * Assignment crowbar
         *
         * @param tar The targeted object
         * @param src The source object
         */
        void AssignmentCrowbar(Call* tar, const Call* src) const override {
            T* t = dynamic_cast<T*>(tar);
            const T* s = dynamic_cast<const T*>(src);
            *t = *s;
        }
    };

} // namespace megamol::core::factories

#endif // MEGAMOLCORE_FACTORIES_CALLAUTODESCRIPTION_H_INCLUDED
