/*
 * Delegate.h
 *
 * Copyright (C) 2006 - 2011 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_DELEGATE_H_INCLUDED
#define VISLIB_DELEGATE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/IllegalStateException.h"
#include "vislib/memutils.h"


namespace vislib {


    /**
     * Class representing a single callback target, which can either be a
     * function (c or static member) or a member-object pair, both with an
     * optional context object (usually a pointer).
     *
     * The first template parameter specifies the return value.
     * The other template parameters define the parameter list of the
     * callback, while void is used to (implicitly) terminate the list.
     * The parameter list may only have a maximum of 10 elements.
     */
    template<class Rv = void, class P1 = void, class P2 = void, class P3 = void, class P4 = void, class P5 = void, class P6 = void, class P7 = void, class P8 = void, class P9 = void, class P10 = void>
    class Delegate {
    public:

        Delegate(void) : callee(NULL) {
        }

        Delegate(const Delegate& rhs) : callee(NULL) {
            (*this) = rhs;
        }

        Delegate(Rv (*funcPtr)(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10))
                : callee(new FunctionCallee(funcPtr)) {
        }

        template<class CT1, class CT2>
        Delegate(Rv (*funcPtr)(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, CT1), CT2 ctxt)
                : callee(new FunctionContextCallee<CT1>(funcPtr, ctxt)) {
        }

        template<class C>
        Delegate(C& obj, Rv (C::*methPtr)(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10))
                : callee(new MethodCallee<C>(obj, methPtr)) {
        }

        template<class C, class CT1, class CT2>
        Delegate(C& obj, Rv (C::*methPtr)(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, CT1), CT2 ctxt)
                : callee(new MethodContextCallee<C, CT1>(obj, methPtr, ctxt)) {
        }

        ~Delegate(void) {
            SAFE_DELETE(this->callee);
        }

        inline bool IsTargetSet(void) const {
            return this->callee != NULL;
        }

        void Set(Rv (*funcPtr)(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10)) {
            SAFE_DELETE(this->callee);
            this->callee = new FunctionCallee(funcPtr);
        }

        template<class CT1, class CT2>
        void Set(Rv (*funcPtr)(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, CT1), CT2 ctxt) {
            SAFE_DELETE(this->callee);
            this->callee = new FunctionContextCallee<CT1>(funcPtr, ctxt);
        }

        template<class C>
        void Set(C& obj, Rv (C::*methPtr)(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10)) {
            SAFE_DELETE(this->callee);
            this->callee = new MethodCallee<C>(obj, methPtr);
        }

        template<class C, class CT1, class CT2>
        void Set(C& obj, Rv (C::*methPtr)(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, CT1), CT2 ctxt) {
            SAFE_DELETE(this->callee);
            this->callee = new MethodContextCallee<C, CT1>(obj, methPtr, ctxt);
        }

        void Unset(void) {
            SAFE_DELETE(this->callee);
        }

        Rv operator()(P1 p1, P2 p2, P3 p3, P4 p4, P5 p5, P6 p6, P7 p7, P8 p8, P9 p9, P10 p10) {
            if (this->callee == NULL) {
                throw vislib::IllegalStateException("Delegate target not set", __FILE__, __LINE__);
            }
            return this->callee->Call(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10);
        }

        bool operator==(const Delegate& rhs) const {
            if (this->callee == NULL) {
                return (rhs.callee == NULL);
            }
            return this->callee->Equals(*rhs.callee);
        }

        Delegate& operator=(const Delegate& rhs) {
            SAFE_DELETE(this->callee);
            if (rhs.callee != NULL) {
                this->callee = rhs.callee->Clone();
            }
            return *this;
        }

    private:

        /**
         * base class for callee implementations
         */
        class AbstractCallee {
        public:

            /** Ctor */
            AbstractCallee(void) {
                // intentionally empty
            }

            /** Dtor */
            virtual ~AbstractCallee(void) {
                // intentionally empty
            }

            /** Call */
            virtual Rv Call(P1 p1, P2 p2, P3 p3, P4 p4, P5 p5, P6 p6, P7 p7, P8 p8, P9 p9, P10 p10) = 0;

            /**
             * Clones this object
             *
             * return A clone
             */
            virtual AbstractCallee * Clone(void) = 0;

            /**
             * Returns if this and rhs are equal
             *
             * param rhs The right hand side operand
             *
             * return True if this and rhs are equal
             */
            virtual bool Equals(const AbstractCallee& rhs) = 0;

        };

        class FunctionCallee : public AbstractCallee {
        public:
            FunctionCallee(Rv (*func)(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10)) : AbstractCallee(), func(func) {
                // Intentionally empty
            }
            virtual ~FunctionCallee(void) {
                this->func = NULL; // DO NOT DELETE
            }
            virtual Rv Call(P1 p1, P2 p2, P3 p3, P4 p4, P5 p5, P6 p6, P7 p7, P8 p8, P9 p9, P10 p10) {
                return this->func(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10);
            }
            virtual AbstractCallee * Clone(void) {
                return new FunctionCallee(this->func);
            }
            virtual bool Equals(const AbstractCallee& rhs) {
                const FunctionCallee *r= dynamic_cast<const FunctionCallee*>(&rhs);
                return (r != NULL)
                    && (r->func == this->func);
            }
        private:
            Rv (*func)(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10);
        };

        template<class CT>
        class FunctionContextCallee : public AbstractCallee {
        public:
            FunctionContextCallee(Rv (*func)(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, CT), CT ctxt) : AbstractCallee(), func(func), ctxt(ctxt) {
                // Intentionally empty
            }
            virtual ~FunctionContextCallee(void) {
                this->func = NULL; // DO NOT DELETE
            }
            virtual Rv Call(P1 p1, P2 p2, P3 p3, P4 p4, P5 p5, P6 p6, P7 p7, P8 p8, P9 p9, P10 p10) {
                return this->func(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, this->ctxt);
            }
            virtual AbstractCallee * Clone(void) {
                return new FunctionContextCallee(this->func, this->ctxt);
            }
            virtual bool Equals(const AbstractCallee& rhs) {
                const FunctionContextCallee *r= dynamic_cast<const FunctionContextCallee*>(&rhs);
                return (r != NULL)
                    && (r->func == this->func)
                    && (r->ctxt == this->ctxt);
            }
        private:
            Rv (*func)(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, CT);
            CT ctxt;
        };

        template<class C>
        class MethodCallee : public AbstractCallee {
        public:
            MethodCallee(C& obj, Rv (C::*meth)(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10)) : AbstractCallee(), obj(obj), meth(meth) {
                // Intentionally empty
            }
            virtual ~MethodCallee(void) {
                this->meth = NULL; // DO NOT DELETE
            }
            virtual Rv Call(P1 p1, P2 p2, P3 p3, P4 p4, P5 p5, P6 p6, P7 p7, P8 p8, P9 p9, P10 p10) {
                return (this->obj.*this->meth)(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10);
            }
            virtual AbstractCallee * Clone(void) {
                return new MethodCallee<C>(this->obj, this->meth);
            }
            virtual bool Equals(const AbstractCallee& rhs) {
                const MethodCallee *r= dynamic_cast<const MethodCallee*>(&rhs);
                return (r != NULL)
                    && (&r->obj == &this->obj)
                    && (r->meth == this->meth);
            }
        private:
            C& obj;
            Rv (C::*meth)(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10);
        };

        template<class C, class CT>
        class MethodContextCallee : public AbstractCallee {
        public:
            MethodContextCallee(C& obj, Rv (C::*meth)(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, CT), CT ctxt) : AbstractCallee(), obj(obj), meth(meth), ctxt(ctxt) {
                // Intentionally empty
            }
            virtual ~MethodContextCallee(void) {
                this->meth = NULL; // DO NOT DELETE
            }
            virtual Rv Call(P1 p1, P2 p2, P3 p3, P4 p4, P5 p5, P6 p6, P7 p7, P8 p8, P9 p9, P10 p10) {
                return (this->obj.*this->meth)(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, this->ctxt);
            }
            virtual AbstractCallee * Clone(void) {
                return new MethodContextCallee<C, CT>(this->obj, this->meth, this->ctxt);
            }
            virtual bool Equals(const AbstractCallee& rhs) {
                const MethodContextCallee *r= dynamic_cast<const MethodContextCallee*>(&rhs);
                return (r != NULL)
                    && (&r->obj == &this->obj)
                    && (r->meth == this->meth)
                    && (r->ctxt == this->ctxt);
            }
        private:
            C& obj;
            Rv (C::*meth)(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, CT);
            CT ctxt;
        };

        /** The callee target */
        AbstractCallee *callee;

    };


    /**
     * Template specialication for delegate functions with 9 parameters
     */
    template<class Rv, class P1, class P2, class P3, class P4, class P5, class P6, class P7, class P8, class P9>
    class Delegate<Rv, P1, P2, P3, P4, P5, P6, P7, P8, P9, void> {
    public:

        Delegate(void) : callee(NULL) {
        }

        Delegate(const Delegate& rhs) : callee(NULL) {
            (*this) = rhs;
        }

        Delegate(Rv (*funcPtr)(P1, P2, P3, P4, P5, P6, P7, P8, P9))
                : callee(new FunctionCallee(funcPtr)) {
        }

        template<class CT1, class CT2>
        Delegate(Rv (*funcPtr)(P1, P2, P3, P4, P5, P6, P7, P8, P9, CT1), CT2 ctxt)
                : callee(new FunctionContextCallee<CT1>(funcPtr, ctxt)) {
        }

        template<class C>
        Delegate(C& obj, Rv (C::*methPtr)(P1, P2, P3, P4, P5, P6, P7, P8, P9))
                : callee(new MethodCallee<C>(obj, methPtr)) {
        }

        template<class C, class CT1, class CT2>
        Delegate(C& obj, Rv (C::*methPtr)(P1, P2, P3, P4, P5, P6, P7, P8, P9, CT1), CT2 ctxt)
                : callee(new MethodContextCallee<C, CT1>(obj, methPtr, ctxt)) {
        }

        ~Delegate(void) {
            SAFE_DELETE(this->callee);
        }

        inline bool IsTargetSet(void) const {
            return this->callee != NULL;
        }

        void Set(Rv (*funcPtr)(P1, P2, P3, P4, P5, P6, P7, P8, P9)) {
            SAFE_DELETE(this->callee);
            this->callee = new FunctionCallee(funcPtr);
        }

        template<class CT1, class CT2>
        void Set(Rv (*funcPtr)(P1, P2, P3, P4, P5, P6, P7, P8, P9, CT1), CT2 ctxt) {
            SAFE_DELETE(this->callee);
            this->callee = new FunctionContextCallee<CT1>(funcPtr, ctxt);
        }

        template<class C>
        void Set(C& obj, Rv (C::*methPtr)(P1, P2, P3, P4, P5, P6, P7, P8, P9)) {
            SAFE_DELETE(this->callee);
            this->callee = new MethodCallee<C>(obj, methPtr);
        }

        template<class C, class CT1, class CT2>
        void Set(C& obj, Rv (C::*methPtr)(P1, P2, P3, P4, P5, P6, P7, P8, P9, CT1), CT2 ctxt) {
            SAFE_DELETE(this->callee);
            this->callee = new MethodContextCallee<C, CT1>(obj, methPtr, ctxt);
        }

        void Unset(void) {
            SAFE_DELETE(this->callee);
        }

        Rv operator()(P1 p1, P2 p2, P3 p3, P4 p4, P5 p5, P6 p6, P7 p7, P8 p8, P9 p9) {
            if (this->callee == NULL) {
                throw vislib::IllegalStateException("Delegate target not set", __FILE__, __LINE__);
            }
            return this->callee->Call(p1, p2, p3, p4, p5, p6, p7, p8, p9);
        }

        bool operator==(const Delegate& rhs) const {
            if (this->callee == NULL) {
                return (rhs.callee == NULL);
            }
            return this->callee->Equals(*rhs.callee);
        }

        Delegate& operator=(const Delegate& rhs) {
            SAFE_DELETE(this->callee);
            if (rhs.callee != NULL) {
                this->callee = rhs.callee->Clone();
            }
            return *this;
        }

    private:

        /**
         * base class for callee implementations
         */
        class AbstractCallee {
        public:

            /** Ctor */
            AbstractCallee(void) {
                // intentionally empty
            }

            /** Dtor */
            virtual ~AbstractCallee(void) {
                // intentionally empty
            }

            /** Call */
            virtual Rv Call(P1 p1, P2 p2, P3 p3, P4 p4, P5 p5, P6 p6, P7 p7, P8 p8, P9 p9) = 0;

            /**
             * Clones this object
             *
             * return A clone
             */
            virtual AbstractCallee * Clone(void) = 0;

            /**
             * Returns if this and rhs are equal
             *
             * param rhs The right hand side operand
             *
             * return True if this and rhs are equal
             */
            virtual bool Equals(const AbstractCallee& rhs) = 0;

        };

        class FunctionCallee : public AbstractCallee {
        public:
            FunctionCallee(Rv (*func)(P1, P2, P3, P4, P5, P6, P7, P8, P9)) : AbstractCallee(), func(func) {
                // Intentionally empty
            }
            virtual ~FunctionCallee(void) {
                this->func = NULL; // DO NOT DELETE
            }
            virtual Rv Call(P1 p1, P2 p2, P3 p3, P4 p4, P5 p5, P6 p6, P7 p7, P8 p8, P9 p9) {
                return this->func(p1, p2, p3, p4, p5, p6, p7, p8, p9);
            }
            virtual AbstractCallee * Clone(void) {
                return new FunctionCallee(this->func);
            }
            virtual bool Equals(const AbstractCallee& rhs) {
                const FunctionCallee *r= dynamic_cast<const FunctionCallee*>(&rhs);
                return (r != NULL)
                    && (r->func == this->func);
            }
        private:
            Rv (*func)(P1, P2, P3, P4, P5, P6, P7, P8, P9);
        };

        template<class CT>
        class FunctionContextCallee : public AbstractCallee {
        public:
            FunctionContextCallee(Rv (*func)(P1, P2, P3, P4, P5, P6, P7, P8, P9, CT), CT ctxt) : AbstractCallee(), func(func), ctxt(ctxt) {
                // Intentionally empty
            }
            virtual ~FunctionContextCallee(void) {
                this->func = NULL; // DO NOT DELETE
            }
            virtual Rv Call(P1 p1, P2 p2, P3 p3, P4 p4, P5 p5, P6 p6, P7 p7, P8 p8, P9 p9) {
                return this->func(p1, p2, p3, p4, p5, p6, p7, p8, p9, this->ctxt);
            }
            virtual AbstractCallee * Clone(void) {
                return new FunctionContextCallee(this->func, this->ctxt);
            }
            virtual bool Equals(const AbstractCallee& rhs) {
                const FunctionContextCallee *r= dynamic_cast<const FunctionContextCallee*>(&rhs);
                return (r != NULL)
                    && (r->func == this->func)
                    && (r->ctxt == this->ctxt);
            }
        private:
            Rv (*func)(P1, P2, P3, P4, P5, P6, P7, P8, P9, CT);
            CT ctxt;
        };

        template<class C>
        class MethodCallee : public AbstractCallee {
        public:
            MethodCallee(C& obj, Rv (C::*meth)(P1, P2, P3, P4, P5, P6, P7, P8, P9)) : AbstractCallee(), obj(obj), meth(meth) {
                // Intentionally empty
            }
            virtual ~MethodCallee(void) {
                this->meth = NULL; // DO NOT DELETE
            }
            virtual Rv Call(P1 p1, P2 p2, P3 p3, P4 p4, P5 p5, P6 p6, P7 p7, P8 p8, P9 p9) {
                return (this->obj.*this->meth)(p1, p2, p3, p4, p5, p6, p7, p8, p9);
            }
            virtual AbstractCallee * Clone(void) {
                return new MethodCallee<C>(this->obj, this->meth);
            }
            virtual bool Equals(const AbstractCallee& rhs) {
                const MethodCallee *r= dynamic_cast<const MethodCallee*>(&rhs);
                return (r != NULL)
                    && (&r->obj == &this->obj)
                    && (r->meth == this->meth);
            }
        private:
            C& obj;
            Rv (C::*meth)(P1, P2, P3, P4, P5, P6, P7, P8, P9);
        };

        template<class C, class CT>
        class MethodContextCallee : public AbstractCallee {
        public:
            MethodContextCallee(C& obj, Rv (C::*meth)(P1, P2, P3, P4, P5, P6, P7, P8, P9, CT), CT ctxt) : AbstractCallee(), obj(obj), meth(meth), ctxt(ctxt) {
                // Intentionally empty
            }
            virtual ~MethodContextCallee(void) {
                this->meth = NULL; // DO NOT DELETE
            }
            virtual Rv Call(P1 p1, P2 p2, P3 p3, P4 p4, P5 p5, P6 p6, P7 p7, P8 p8, P9 p9) {
                return (this->obj.*this->meth)(p1, p2, p3, p4, p5, p6, p7, p8, p9, this->ctxt);
            }
            virtual AbstractCallee * Clone(void) {
                return new MethodContextCallee<C, CT>(this->obj, this->meth, this->ctxt);
            }
            virtual bool Equals(const AbstractCallee& rhs) {
                const MethodContextCallee *r= dynamic_cast<const MethodContextCallee*>(&rhs);
                return (r != NULL)
                    && (&r->obj == &this->obj)
                    && (r->meth == this->meth)
                    && (r->ctxt == this->ctxt);
            }
        private:
            C& obj;
            Rv (C::*meth)(P1, P2, P3, P4, P5, P6, P7, P8, P9, CT);
            CT ctxt;
        };

        /** The callee target */
        AbstractCallee *callee;

    };


    /**
     * Template specialication for delegate functions with 8 parameters
     */
    template<class Rv, class P1, class P2, class P3, class P4, class P5, class P6, class P7, class P8>
    class Delegate<Rv, P1, P2, P3, P4, P5, P6, P7, P8, void, void> {
    public:

        Delegate(void) : callee(NULL) {
        }

        Delegate(const Delegate& rhs) : callee(NULL) {
            (*this) = rhs;
        }

        Delegate(Rv (*funcPtr)(P1, P2, P3, P4, P5, P6, P7, P8))
                : callee(new FunctionCallee(funcPtr)) {
        }

        template<class CT1, class CT2>
        Delegate(Rv (*funcPtr)(P1, P2, P3, P4, P5, P6, P7, P8, CT1), CT2 ctxt)
                : callee(new FunctionContextCallee<CT1>(funcPtr, ctxt)) {
        }

        template<class C>
        Delegate(C& obj, Rv (C::*methPtr)(P1, P2, P3, P4, P5, P6, P7, P8))
                : callee(new MethodCallee<C>(obj, methPtr)) {
        }

        template<class C, class CT1, class CT2>
        Delegate(C& obj, Rv (C::*methPtr)(P1, P2, P3, P4, P5, P6, P7, P8, CT1), CT2 ctxt)
                : callee(new MethodContextCallee<C, CT1>(obj, methPtr, ctxt)) {
        }

        ~Delegate(void) {
            SAFE_DELETE(this->callee);
        }

        inline bool IsTargetSet(void) const {
            return this->callee != NULL;
        }

        void Set(Rv (*funcPtr)(P1, P2, P3, P4, P5, P6, P7, P8)) {
            SAFE_DELETE(this->callee);
            this->callee = new FunctionCallee(funcPtr);
        }

        template<class CT1, class CT2>
        void Set(Rv (*funcPtr)(P1, P2, P3, P4, P5, P6, P7, P8, CT1), CT2 ctxt) {
            SAFE_DELETE(this->callee);
            this->callee = new FunctionContextCallee<CT1>(funcPtr, ctxt);
        }

        template<class C>
        void Set(C& obj, Rv (C::*methPtr)(P1, P2, P3, P4, P5, P6, P7, P8)) {
            SAFE_DELETE(this->callee);
            this->callee = new MethodCallee<C>(obj, methPtr);
        }

        template<class C, class CT1, class CT2>
        void Set(C& obj, Rv (C::*methPtr)(P1, P2, P3, P4, P5, P6, P7, P8, CT1), CT2 ctxt) {
            SAFE_DELETE(this->callee);
            this->callee = new MethodContextCallee<C, CT1>(obj, methPtr, ctxt);
        }

        void Unset(void) {
            SAFE_DELETE(this->callee);
        }

        Rv operator()(P1 p1, P2 p2, P3 p3, P4 p4, P5 p5, P6 p6, P7 p7, P8 p8) {
            if (this->callee == NULL) {
                throw vislib::IllegalStateException("Delegate target not set", __FILE__, __LINE__);
            }
            return this->callee->Call(p1, p2, p3, p4, p5, p6, p7, p8);
        }

        bool operator==(const Delegate& rhs) const {
            if (this->callee == NULL) {
                return (rhs.callee == NULL);
            }
            return this->callee->Equals(*rhs.callee);
        }

        Delegate& operator=(const Delegate& rhs) {
            SAFE_DELETE(this->callee);
            if (rhs.callee != NULL) {
                this->callee = rhs.callee->Clone();
            }
            return *this;
        }

    private:

        /**
         * base class for callee implementations
         */
        class AbstractCallee {
        public:

            /** Ctor */
            AbstractCallee(void) {
                // intentionally empty
            }

            /** Dtor */
            virtual ~AbstractCallee(void) {
                // intentionally empty
            }

            /** Call */
            virtual Rv Call(P1 p1, P2 p2, P3 p3, P4 p4, P5 p5, P6 p6, P7 p7, P8 p8) = 0;

            /**
             * Clones this object
             *
             * return A clone
             */
            virtual AbstractCallee * Clone(void) = 0;

            /**
             * Returns if this and rhs are equal
             *
             * param rhs The right hand side operand
             *
             * return True if this and rhs are equal
             */
            virtual bool Equals(const AbstractCallee& rhs) = 0;

        };

        class FunctionCallee : public AbstractCallee {
        public:
            FunctionCallee(Rv (*func)(P1, P2, P3, P4, P5, P6, P7, P8)) : AbstractCallee(), func(func) {
                // Intentionally empty
            }
            virtual ~FunctionCallee(void) {
                this->func = NULL; // DO NOT DELETE
            }
            virtual Rv Call(P1 p1, P2 p2, P3 p3, P4 p4, P5 p5, P6 p6, P7 p7, P8 p8) {
                return this->func(p1, p2, p3, p4, p5, p6, p7, p8);
            }
            virtual AbstractCallee * Clone(void) {
                return new FunctionCallee(this->func);
            }
            virtual bool Equals(const AbstractCallee& rhs) {
                const FunctionCallee *r= dynamic_cast<const FunctionCallee*>(&rhs);
                return (r != NULL)
                    && (r->func == this->func);
            }
        private:
            Rv (*func)(P1, P2, P3, P4, P5, P6, P7, P8);
        };

        template<class CT>
        class FunctionContextCallee : public AbstractCallee {
        public:
            FunctionContextCallee(Rv (*func)(P1, P2, P3, P4, P5, P6, P7, P8, CT), CT ctxt) : AbstractCallee(), func(func), ctxt(ctxt) {
                // Intentionally empty
            }
            virtual ~FunctionContextCallee(void) {
                this->func = NULL; // DO NOT DELETE
            }
            virtual Rv Call(P1 p1, P2 p2, P3 p3, P4 p4, P5 p5, P6 p6, P7 p7, P8 p8) {
                return this->func(p1, p2, p3, p4, p5, p6, p7, p8, this->ctxt);
            }
            virtual AbstractCallee * Clone(void) {
                return new FunctionContextCallee(this->func, this->ctxt);
            }
            virtual bool Equals(const AbstractCallee& rhs) {
                const FunctionContextCallee *r= dynamic_cast<const FunctionContextCallee*>(&rhs);
                return (r != NULL)
                    && (r->func == this->func)
                    && (r->ctxt == this->ctxt);
            }
        private:
            Rv (*func)(P1, P2, P3, P4, P5, P6, P7, P8, CT);
            CT ctxt;
        };

        template<class C>
        class MethodCallee : public AbstractCallee {
        public:
            MethodCallee(C& obj, Rv (C::*meth)(P1, P2, P3, P4, P5, P6, P7, P8)) : AbstractCallee(), obj(obj), meth(meth) {
                // Intentionally empty
            }
            virtual ~MethodCallee(void) {
                this->meth = NULL; // DO NOT DELETE
            }
            virtual Rv Call(P1 p1, P2 p2, P3 p3, P4 p4, P5 p5, P6 p6, P7 p7, P8 p8) {
                return (this->obj.*this->meth)(p1, p2, p3, p4, p5, p6, p7, p8);
            }
            virtual AbstractCallee * Clone(void) {
                return new MethodCallee<C>(this->obj, this->meth);
            }
            virtual bool Equals(const AbstractCallee& rhs) {
                const MethodCallee *r= dynamic_cast<const MethodCallee*>(&rhs);
                return (r != NULL)
                    && (&r->obj == &this->obj)
                    && (r->meth == this->meth);
            }
        private:
            C& obj;
            Rv (C::*meth)(P1, P2, P3, P4, P5, P6, P7, P8);
        };

        template<class C, class CT>
        class MethodContextCallee : public AbstractCallee {
        public:
            MethodContextCallee(C& obj, Rv (C::*meth)(P1, P2, P3, P4, P5, P6, P7, P8, CT), CT ctxt) : AbstractCallee(), obj(obj), meth(meth), ctxt(ctxt) {
                // Intentionally empty
            }
            virtual ~MethodContextCallee(void) {
                this->meth = NULL; // DO NOT DELETE
            }
            virtual Rv Call(P1 p1, P2 p2, P3 p3, P4 p4, P5 p5, P6 p6, P7 p7, P8 p8) {
                return (this->obj.*this->meth)(p1, p2, p3, p4, p5, p6, p7, p8, this->ctxt);
            }
            virtual AbstractCallee * Clone(void) {
                return new MethodContextCallee<C, CT>(this->obj, this->meth, this->ctxt);
            }
            virtual bool Equals(const AbstractCallee& rhs) {
                const MethodContextCallee *r= dynamic_cast<const MethodContextCallee*>(&rhs);
                return (r != NULL)
                    && (&r->obj == &this->obj)
                    && (r->meth == this->meth)
                    && (r->ctxt == this->ctxt);
            }
        private:
            C& obj;
            Rv (C::*meth)(P1, P2, P3, P4, P5, P6, P7, P8, CT);
            CT ctxt;
        };

        /** The callee target */
        AbstractCallee *callee;

    };


    /**
     * Template specialication for delegate functions with 7 parameters
     */
    template<class Rv, class P1, class P2, class P3, class P4, class P5, class P6, class P7>
    class Delegate<Rv, P1, P2, P3, P4, P5, P6, P7, void, void, void> {
    public:

        Delegate(void) : callee(NULL) {
        }

        Delegate(const Delegate& rhs) : callee(NULL) {
            (*this) = rhs;
        }

        Delegate(Rv (*funcPtr)(P1, P2, P3, P4, P5, P6, P7))
                : callee(new FunctionCallee(funcPtr)) {
        }

        template<class CT1, class CT2>
        Delegate(Rv (*funcPtr)(P1, P2, P3, P4, P5, P6, P7, CT1), CT2 ctxt)
                : callee(new FunctionContextCallee<CT1>(funcPtr, ctxt)) {
        }

        template<class C>
        Delegate(C& obj, Rv (C::*methPtr)(P1, P2, P3, P4, P5, P6, P7))
                : callee(new MethodCallee<C>(obj, methPtr)) {
        }

        template<class C, class CT1, class CT2>
        Delegate(C& obj, Rv (C::*methPtr)(P1, P2, P3, P4, P5, P6, P7, CT1), CT2 ctxt)
                : callee(new MethodContextCallee<C, CT1>(obj, methPtr, ctxt)) {
        }

        ~Delegate(void) {
            SAFE_DELETE(this->callee);
        }

        inline bool IsTargetSet(void) const {
            return this->callee != NULL;
        }

        void Set(Rv (*funcPtr)(P1, P2, P3, P4, P5, P6, P7)) {
            SAFE_DELETE(this->callee);
            this->callee = new FunctionCallee(funcPtr);
        }

        template<class CT1, class CT2>
        void Set(Rv (*funcPtr)(P1, P2, P3, P4, P5, P6, P7, CT1), CT2 ctxt) {
            SAFE_DELETE(this->callee);
            this->callee = new FunctionContextCallee<CT1>(funcPtr, ctxt);
        }

        template<class C>
        void Set(C& obj, Rv (C::*methPtr)(P1, P2, P3, P4, P5, P6, P7)) {
            SAFE_DELETE(this->callee);
            this->callee = new MethodCallee<C>(obj, methPtr);
        }

        template<class C, class CT1, class CT2>
        void Set(C& obj, Rv (C::*methPtr)(P1, P2, P3, P4, P5, P6, P7, CT1), CT2 ctxt) {
            SAFE_DELETE(this->callee);
            this->callee = new MethodContextCallee<C, CT1>(obj, methPtr, ctxt);
        }

        void Unset(void) {
            SAFE_DELETE(this->callee);
        }

        Rv operator()(P1 p1, P2 p2, P3 p3, P4 p4, P5 p5, P6 p6, P7 p7) {
            if (this->callee == NULL) {
                throw vislib::IllegalStateException("Delegate target not set", __FILE__, __LINE__);
            }
            return this->callee->Call(p1, p2, p3, p4, p5, p6, p7);
        }

        bool operator==(const Delegate& rhs) const {
            if (this->callee == NULL) {
                return (rhs.callee == NULL);
            }
            return this->callee->Equals(*rhs.callee);
        }

        Delegate& operator=(const Delegate& rhs) {
            SAFE_DELETE(this->callee);
            if (rhs.callee != NULL) {
                this->callee = rhs.callee->Clone();
            }
            return *this;
        }

    private:

        /**
         * base class for callee implementations
         */
        class AbstractCallee {
        public:

            /** Ctor */
            AbstractCallee(void) {
                // intentionally empty
            }

            /** Dtor */
            virtual ~AbstractCallee(void) {
                // intentionally empty
            }

            /** Call */
            virtual Rv Call(P1 p1, P2 p2, P3 p3, P4 p4, P5 p5, P6 p6, P7 p7) = 0;

            /**
             * Clones this object
             *
             * return A clone
             */
            virtual AbstractCallee * Clone(void) = 0;

            /**
             * Returns if this and rhs are equal
             *
             * param rhs The right hand side operand
             *
             * return True if this and rhs are equal
             */
            virtual bool Equals(const AbstractCallee& rhs) = 0;

        };

        class FunctionCallee : public AbstractCallee {
        public:
            FunctionCallee(Rv (*func)(P1, P2, P3, P4, P5, P6, P7)) : AbstractCallee(), func(func) {
                // Intentionally empty
            }
            virtual ~FunctionCallee(void) {
                this->func = NULL; // DO NOT DELETE
            }
            virtual Rv Call(P1 p1, P2 p2, P3 p3, P4 p4, P5 p5, P6 p6, P7 p7) {
                return this->func(p1, p2, p3, p4, p5, p6, p7);
            }
            virtual AbstractCallee * Clone(void) {
                return new FunctionCallee(this->func);
            }
            virtual bool Equals(const AbstractCallee& rhs) {
                const FunctionCallee *r= dynamic_cast<const FunctionCallee*>(&rhs);
                return (r != NULL)
                    && (r->func == this->func);
            }
        private:
            Rv (*func)(P1, P2, P3, P4, P5, P6, P7);
        };

        template<class CT>
        class FunctionContextCallee : public AbstractCallee {
        public:
            FunctionContextCallee(Rv (*func)(P1, P2, P3, P4, P5, P6, P7, CT), CT ctxt) : AbstractCallee(), func(func), ctxt(ctxt) {
                // Intentionally empty
            }
            virtual ~FunctionContextCallee(void) {
                this->func = NULL; // DO NOT DELETE
            }
            virtual Rv Call(P1 p1, P2 p2, P3 p3, P4 p4, P5 p5, P6 p6, P7 p7) {
                return this->func(p1, p2, p3, p4, p5, p6, p7, this->ctxt);
            }
            virtual AbstractCallee * Clone(void) {
                return new FunctionContextCallee(this->func, this->ctxt);
            }
            virtual bool Equals(const AbstractCallee& rhs) {
                const FunctionContextCallee *r= dynamic_cast<const FunctionContextCallee*>(&rhs);
                return (r != NULL)
                    && (r->func == this->func)
                    && (r->ctxt == this->ctxt);
            }
        private:
            Rv (*func)(P1, P2, P3, P4, P5, P6, P7, CT);
            CT ctxt;
        };

        template<class C>
        class MethodCallee : public AbstractCallee {
        public:
            MethodCallee(C& obj, Rv (C::*meth)(P1, P2, P3, P4, P5, P6, P7)) : AbstractCallee(), obj(obj), meth(meth) {
                // Intentionally empty
            }
            virtual ~MethodCallee(void) {
                this->meth = NULL; // DO NOT DELETE
            }
            virtual Rv Call(P1 p1, P2 p2, P3 p3, P4 p4, P5 p5, P6 p6, P7 p7) {
                return (this->obj.*this->meth)(p1, p2, p3, p4, p5, p6, p7);
            }
            virtual AbstractCallee * Clone(void) {
                return new MethodCallee<C>(this->obj, this->meth);
            }
            virtual bool Equals(const AbstractCallee& rhs) {
                const MethodCallee *r= dynamic_cast<const MethodCallee*>(&rhs);
                return (r != NULL)
                    && (&r->obj == &this->obj)
                    && (r->meth == this->meth);
            }
        private:
            C& obj;
            Rv (C::*meth)(P1, P2, P3, P4, P5, P6, P7);
        };

        template<class C, class CT>
        class MethodContextCallee : public AbstractCallee {
        public:
            MethodContextCallee(C& obj, Rv (C::*meth)(P1, P2, P3, P4, P5, P6, P7, CT), CT ctxt) : AbstractCallee(), obj(obj), meth(meth), ctxt(ctxt) {
                // Intentionally empty
            }
            virtual ~MethodContextCallee(void) {
                this->meth = NULL; // DO NOT DELETE
            }
            virtual Rv Call(P1 p1, P2 p2, P3 p3, P4 p4, P5 p5, P6 p6, P7 p7) {
                return (this->obj.*this->meth)(p1, p2, p3, p4, p5, p6, p7, this->ctxt);
            }
            virtual AbstractCallee * Clone(void) {
                return new MethodContextCallee<C, CT>(this->obj, this->meth, this->ctxt);
            }
            virtual bool Equals(const AbstractCallee& rhs) {
                const MethodContextCallee *r= dynamic_cast<const MethodContextCallee*>(&rhs);
                return (r != NULL)
                    && (&r->obj == &this->obj)
                    && (r->meth == this->meth)
                    && (r->ctxt == this->ctxt);
            }
        private:
            C& obj;
            Rv (C::*meth)(P1, P2, P3, P4, P5, P6, P7, CT);
            CT ctxt;
        };

        /** The callee target */
        AbstractCallee *callee;

    };


    /**
     * Template specialication for delegate functions with 6 parameters
     */
    template<class Rv, class P1, class P2, class P3, class P4, class P5, class P6>
    class Delegate<Rv, P1, P2, P3, P4, P5, P6, void, void, void, void> {
    public:

        Delegate(void) : callee(NULL) {
        }

        Delegate(const Delegate& rhs) : callee(NULL) {
            (*this) = rhs;
        }

        Delegate(Rv (*funcPtr)(P1, P2, P3, P4, P5, P6))
                : callee(new FunctionCallee(funcPtr)) {
        }

        template<class CT1, class CT2>
        Delegate(Rv (*funcPtr)(P1, P2, P3, P4, P5, P6, CT1), CT2 ctxt)
                : callee(new FunctionContextCallee<CT1>(funcPtr, ctxt)) {
        }

        template<class C>
        Delegate(C& obj, Rv (C::*methPtr)(P1, P2, P3, P4, P5, P6))
                : callee(new MethodCallee<C>(obj, methPtr)) {
        }

        template<class C, class CT1, class CT2>
        Delegate(C& obj, Rv (C::*methPtr)(P1, P2, P3, P4, P5, P6, CT1), CT2 ctxt)
                : callee(new MethodContextCallee<C, CT1>(obj, methPtr, ctxt)) {
        }

        ~Delegate(void) {
            SAFE_DELETE(this->callee);
        }

        inline bool IsTargetSet(void) const {
            return this->callee != NULL;
        }

        void Set(Rv (*funcPtr)(P1, P2, P3, P4, P5, P6)) {
            SAFE_DELETE(this->callee);
            this->callee = new FunctionCallee(funcPtr);
        }

        template<class CT1, class CT2>
        void Set(Rv (*funcPtr)(P1, P2, P3, P4, P5, P6, CT1), CT2 ctxt) {
            SAFE_DELETE(this->callee);
            this->callee = new FunctionContextCallee<CT1>(funcPtr, ctxt);
        }

        template<class C>
        void Set(C& obj, Rv (C::*methPtr)(P1, P2, P3, P4, P5, P6)) {
            SAFE_DELETE(this->callee);
            this->callee = new MethodCallee<C>(obj, methPtr);
        }

        template<class C, class CT1, class CT2>
        void Set(C& obj, Rv (C::*methPtr)(P1, P2, P3, P4, P5, P6, CT1), CT2 ctxt) {
            SAFE_DELETE(this->callee);
            this->callee = new MethodContextCallee<C, CT1>(obj, methPtr, ctxt);
        }

        void Unset(void) {
            SAFE_DELETE(this->callee);
        }

        Rv operator()(P1 p1, P2 p2, P3 p3, P4 p4, P5 p5, P6 p6) {
            if (this->callee == NULL) {
                throw vislib::IllegalStateException("Delegate target not set", __FILE__, __LINE__);
            }
            return this->callee->Call(p1, p2, p3, p4, p5, p6);
        }

        bool operator==(const Delegate& rhs) const {
            if (this->callee == NULL) {
                return (rhs.callee == NULL);
            }
            return this->callee->Equals(*rhs.callee);
        }

        Delegate& operator=(const Delegate& rhs) {
            SAFE_DELETE(this->callee);
            if (rhs.callee != NULL) {
                this->callee = rhs.callee->Clone();
            }
            return *this;
        }

    private:

        /**
         * base class for callee implementations
         */
        class AbstractCallee {
        public:

            /** Ctor */
            AbstractCallee(void) {
                // intentionally empty
            }

            /** Dtor */
            virtual ~AbstractCallee(void) {
                // intentionally empty
            }

            /** Call */
            virtual Rv Call(P1 p1, P2 p2, P3 p3, P4 p4, P5 p5, P6 p6) = 0;

            /**
             * Clones this object
             *
             * return A clone
             */
            virtual AbstractCallee * Clone(void) = 0;

            /**
             * Returns if this and rhs are equal
             *
             * param rhs The right hand side operand
             *
             * return True if this and rhs are equal
             */
            virtual bool Equals(const AbstractCallee& rhs) = 0;

        };

        class FunctionCallee : public AbstractCallee {
        public:
            FunctionCallee(Rv (*func)(P1, P2, P3, P4, P5, P6)) : AbstractCallee(), func(func) {
                // Intentionally empty
            }
            virtual ~FunctionCallee(void) {
                this->func = NULL; // DO NOT DELETE
            }
            virtual Rv Call(P1 p1, P2 p2, P3 p3, P4 p4, P5 p5, P6 p6) {
                return this->func(p1, p2, p3, p4, p5, p6);
            }
            virtual AbstractCallee * Clone(void) {
                return new FunctionCallee(this->func);
            }
            virtual bool Equals(const AbstractCallee& rhs) {
                const FunctionCallee *r= dynamic_cast<const FunctionCallee*>(&rhs);
                return (r != NULL)
                    && (r->func == this->func);
            }
        private:
            Rv (*func)(P1, P2, P3, P4, P5, P6);
        };

        template<class CT>
        class FunctionContextCallee : public AbstractCallee {
        public:
            FunctionContextCallee(Rv (*func)(P1, P2, P3, P4, P5, P6, CT), CT ctxt) : AbstractCallee(), func(func), ctxt(ctxt) {
                // Intentionally empty
            }
            virtual ~FunctionContextCallee(void) {
                this->func = NULL; // DO NOT DELETE
            }
            virtual Rv Call(P1 p1, P2 p2, P3 p3, P4 p4, P5 p5, P6 p6) {
                return this->func(p1, p2, p3, p4, p5, p6, this->ctxt);
            }
            virtual AbstractCallee * Clone(void) {
                return new FunctionContextCallee(this->func, this->ctxt);
            }
            virtual bool Equals(const AbstractCallee& rhs) {
                const FunctionContextCallee *r= dynamic_cast<const FunctionContextCallee*>(&rhs);
                return (r != NULL)
                    && (r->func == this->func)
                    && (r->ctxt == this->ctxt);
            }
        private:
            Rv (*func)(P1, P2, P3, P4, P5, P6, CT);
            CT ctxt;
        };

        template<class C>
        class MethodCallee : public AbstractCallee {
        public:
            MethodCallee(C& obj, Rv (C::*meth)(P1, P2, P3, P4, P5, P6)) : AbstractCallee(), obj(obj), meth(meth) {
                // Intentionally empty
            }
            virtual ~MethodCallee(void) {
                this->meth = NULL; // DO NOT DELETE
            }
            virtual Rv Call(P1 p1, P2 p2, P3 p3, P4 p4, P5 p5, P6 p6) {
                return (this->obj.*this->meth)(p1, p2, p3, p4, p5, p6);
            }
            virtual AbstractCallee * Clone(void) {
                return new MethodCallee<C>(this->obj, this->meth);
            }
            virtual bool Equals(const AbstractCallee& rhs) {
                const MethodCallee *r= dynamic_cast<const MethodCallee*>(&rhs);
                return (r != NULL)
                    && (&r->obj == &this->obj)
                    && (r->meth == this->meth);
            }
        private:
            C& obj;
            Rv (C::*meth)(P1, P2, P3, P4, P5, P6);
        };

        template<class C, class CT>
        class MethodContextCallee : public AbstractCallee {
        public:
            MethodContextCallee(C& obj, Rv (C::*meth)(P1, P2, P3, P4, P5, P6, CT), CT ctxt) : AbstractCallee(), obj(obj), meth(meth), ctxt(ctxt) {
                // Intentionally empty
            }
            virtual ~MethodContextCallee(void) {
                this->meth = NULL; // DO NOT DELETE
            }
            virtual Rv Call(P1 p1, P2 p2, P3 p3, P4 p4, P5 p5, P6 p6) {
                return (this->obj.*this->meth)(p1, p2, p3, p4, p5, p6, this->ctxt);
            }
            virtual AbstractCallee * Clone(void) {
                return new MethodContextCallee<C, CT>(this->obj, this->meth, this->ctxt);
            }
            virtual bool Equals(const AbstractCallee& rhs) {
                const MethodContextCallee *r= dynamic_cast<const MethodContextCallee*>(&rhs);
                return (r != NULL)
                    && (&r->obj == &this->obj)
                    && (r->meth == this->meth)
                    && (r->ctxt == this->ctxt);
            }
        private:
            C& obj;
            Rv (C::*meth)(P1, P2, P3, P4, P5, P6, CT);
            CT ctxt;
        };

        /** The callee target */
        AbstractCallee *callee;

    };


    /**
     * Template specialication for delegate functions with 5 parameters
     */
    template<class Rv, class P1, class P2, class P3, class P4, class P5>
    class Delegate<Rv, P1, P2, P3, P4, P5, void, void, void, void, void> {
    public:

        Delegate(void) : callee(NULL) {
        }

        Delegate(const Delegate& rhs) : callee(NULL) {
            (*this) = rhs;
        }

        Delegate(Rv (*funcPtr)(P1, P2, P3, P4, P5))
                : callee(new FunctionCallee(funcPtr)) {
        }

        template<class CT1, class CT2>
        Delegate(Rv (*funcPtr)(P1, P2, P3, P4, P5, CT1), CT2 ctxt)
                : callee(new FunctionContextCallee<CT1>(funcPtr, ctxt)) {
        }

        template<class C>
        Delegate(C& obj, Rv (C::*methPtr)(P1, P2, P3, P4, P5))
                : callee(new MethodCallee<C>(obj, methPtr)) {
        }

        template<class C, class CT1, class CT2>
        Delegate(C& obj, Rv (C::*methPtr)(P1, P2, P3, P4, P5, CT1), CT2 ctxt)
                : callee(new MethodContextCallee<C, CT1>(obj, methPtr, ctxt)) {
        }

        ~Delegate(void) {
            SAFE_DELETE(this->callee);
        }

        inline bool IsTargetSet(void) const {
            return this->callee != NULL;
        }

        void Set(Rv (*funcPtr)(P1, P2, P3, P4, P5)) {
            SAFE_DELETE(this->callee);
            this->callee = new FunctionCallee(funcPtr);
        }

        template<class CT1, class CT2>
        void Set(Rv (*funcPtr)(P1, P2, P3, P4, P5, CT1), CT2 ctxt) {
            SAFE_DELETE(this->callee);
            this->callee = new FunctionContextCallee<CT1>(funcPtr, ctxt);
        }

        template<class C>
        void Set(C& obj, Rv (C::*methPtr)(P1, P2, P3, P4, P5)) {
            SAFE_DELETE(this->callee);
            this->callee = new MethodCallee<C>(obj, methPtr);
        }

        template<class C, class CT1, class CT2>
        void Set(C& obj, Rv (C::*methPtr)(P1, P2, P3, P4, P5, CT1), CT2 ctxt) {
            SAFE_DELETE(this->callee);
            this->callee = new MethodContextCallee<C, CT1>(obj, methPtr, ctxt);
        }

        void Unset(void) {
            SAFE_DELETE(this->callee);
        }

        Rv operator()(P1 p1, P2 p2, P3 p3, P4 p4, P5 p5) {
            if (this->callee == NULL) {
                throw vislib::IllegalStateException("Delegate target not set", __FILE__, __LINE__);
            }
            return this->callee->Call(p1, p2, p3, p4, p5);
        }

        bool operator==(const Delegate& rhs) const {
            if (this->callee == NULL) {
                return (rhs.callee == NULL);
            }
            return this->callee->Equals(*rhs.callee);
        }

        Delegate& operator=(const Delegate& rhs) {
            SAFE_DELETE(this->callee);
            if (rhs.callee != NULL) {
                this->callee = rhs.callee->Clone();
            }
            return *this;
        }

    private:

        /**
         * base class for callee implementations
         */
        class AbstractCallee {
        public:

            /** Ctor */
            AbstractCallee(void) {
                // intentionally empty
            }

            /** Dtor */
            virtual ~AbstractCallee(void) {
                // intentionally empty
            }

            /** Call */
            virtual Rv Call(P1 p1, P2 p2, P3 p3, P4 p4, P5 p5) = 0;

            /**
             * Clones this object
             *
             * return A clone
             */
            virtual AbstractCallee * Clone(void) = 0;

            /**
             * Returns if this and rhs are equal
             *
             * param rhs The right hand side operand
             *
             * return True if this and rhs are equal
             */
            virtual bool Equals(const AbstractCallee& rhs) = 0;

        };

        class FunctionCallee : public AbstractCallee {
        public:
            FunctionCallee(Rv (*func)(P1, P2, P3, P4, P5)) : AbstractCallee(), func(func) {
                // Intentionally empty
            }
            virtual ~FunctionCallee(void) {
                this->func = NULL; // DO NOT DELETE
            }
            virtual Rv Call(P1 p1, P2 p2, P3 p3, P4 p4, P5 p5) {
                return this->func(p1, p2, p3, p4, p5);
            }
            virtual AbstractCallee * Clone(void) {
                return new FunctionCallee(this->func);
            }
            virtual bool Equals(const AbstractCallee& rhs) {
                const FunctionCallee *r= dynamic_cast<const FunctionCallee*>(&rhs);
                return (r != NULL)
                    && (r->func == this->func);
            }
        private:
            Rv (*func)(P1, P2, P3, P4, P5);
        };

        template<class CT>
        class FunctionContextCallee : public AbstractCallee {
        public:
            FunctionContextCallee(Rv (*func)(P1, P2, P3, P4, P5, CT), CT ctxt) : AbstractCallee(), func(func), ctxt(ctxt) {
                // Intentionally empty
            }
            virtual ~FunctionContextCallee(void) {
                this->func = NULL; // DO NOT DELETE
            }
            virtual Rv Call(P1 p1, P2 p2, P3 p3, P4 p4, P5 p5) {
                return this->func(p1, p2, p3, p4, p5, this->ctxt);
            }
            virtual AbstractCallee * Clone(void) {
                return new FunctionContextCallee(this->func, this->ctxt);
            }
            virtual bool Equals(const AbstractCallee& rhs) {
                const FunctionContextCallee *r= dynamic_cast<const FunctionContextCallee*>(&rhs);
                return (r != NULL)
                    && (r->func == this->func)
                    && (r->ctxt == this->ctxt);
            }
        private:
            Rv (*func)(P1, P2, P3, P4, P5, CT);
            CT ctxt;
        };

        template<class C>
        class MethodCallee : public AbstractCallee {
        public:
            MethodCallee(C& obj, Rv (C::*meth)(P1, P2, P3, P4, P5)) : AbstractCallee(), obj(obj), meth(meth) {
                // Intentionally empty
            }
            virtual ~MethodCallee(void) {
                this->meth = NULL; // DO NOT DELETE
            }
            virtual Rv Call(P1 p1, P2 p2, P3 p3, P4 p4, P5 p5) {
                return (this->obj.*this->meth)(p1, p2, p3, p4, p5);
            }
            virtual AbstractCallee * Clone(void) {
                return new MethodCallee<C>(this->obj, this->meth);
            }
            virtual bool Equals(const AbstractCallee& rhs) {
                const MethodCallee *r= dynamic_cast<const MethodCallee*>(&rhs);
                return (r != NULL)
                    && (&r->obj == &this->obj)
                    && (r->meth == this->meth);
            }
        private:
            C& obj;
            Rv (C::*meth)(P1, P2, P3, P4, P5);
        };

        template<class C, class CT>
        class MethodContextCallee : public AbstractCallee {
        public:
            MethodContextCallee(C& obj, Rv (C::*meth)(P1, P2, P3, P4, P5, CT), CT ctxt) : AbstractCallee(), obj(obj), meth(meth), ctxt(ctxt) {
                // Intentionally empty
            }
            virtual ~MethodContextCallee(void) {
                this->meth = NULL; // DO NOT DELETE
            }
            virtual Rv Call(P1 p1, P2 p2, P3 p3, P4 p4, P5 p5) {
                return (this->obj.*this->meth)(p1, p2, p3, p4, p5, this->ctxt);
            }
            virtual AbstractCallee * Clone(void) {
                return new MethodContextCallee<C, CT>(this->obj, this->meth, this->ctxt);
            }
            virtual bool Equals(const AbstractCallee& rhs) {
                const MethodContextCallee *r= dynamic_cast<const MethodContextCallee*>(&rhs);
                return (r != NULL)
                    && (&r->obj == &this->obj)
                    && (r->meth == this->meth)
                    && (r->ctxt == this->ctxt);
            }
        private:
            C& obj;
            Rv (C::*meth)(P1, P2, P3, P4, P5, CT);
            CT ctxt;
        };

        /** The callee target */
        AbstractCallee *callee;

    };


    /**
     * Template specialication for delegate functions with 4 parameters
     */
    template<class Rv, class P1, class P2, class P3, class P4>
    class Delegate<Rv, P1, P2, P3, P4, void, void, void, void, void, void> {
    public:

        Delegate(void) : callee(NULL) {
        }

        Delegate(const Delegate& rhs) : callee(NULL) {
            (*this) = rhs;
        }

        Delegate(Rv (*funcPtr)(P1, P2, P3, P4))
                : callee(new FunctionCallee(funcPtr)) {
        }

        template<class CT1, class CT2>
        Delegate(Rv (*funcPtr)(P1, P2, P3, P4, CT1), CT2 ctxt)
                : callee(new FunctionContextCallee<CT1>(funcPtr, ctxt)) {
        }

        template<class C>
        Delegate(C& obj, Rv (C::*methPtr)(P1, P2, P3, P4))
                : callee(new MethodCallee<C>(obj, methPtr)) {
        }

        template<class C, class CT1, class CT2>
        Delegate(C& obj, Rv (C::*methPtr)(P1, P2, P3, P4, CT1), CT2 ctxt)
                : callee(new MethodContextCallee<C, CT1>(obj, methPtr, ctxt)) {
        }

        ~Delegate(void) {
            SAFE_DELETE(this->callee);
        }

        inline bool IsTargetSet(void) const {
            return this->callee != NULL;
        }

        void Set(Rv (*funcPtr)(P1, P2, P3, P4)) {
            SAFE_DELETE(this->callee);
            this->callee = new FunctionCallee(funcPtr);
        }

        template<class CT1, class CT2>
        void Set(Rv (*funcPtr)(P1, P2, P3, P4, CT1), CT2 ctxt) {
            SAFE_DELETE(this->callee);
            this->callee = new FunctionContextCallee<CT1>(funcPtr, ctxt);
        }

        template<class C>
        void Set(C& obj, Rv (C::*methPtr)(P1, P2, P3, P4)) {
            SAFE_DELETE(this->callee);
            this->callee = new MethodCallee<C>(obj, methPtr);
        }

        template<class C, class CT1, class CT2>
        void Set(C& obj, Rv (C::*methPtr)(P1, P2, P3, P4, CT1), CT2 ctxt) {
            SAFE_DELETE(this->callee);
            this->callee = new MethodContextCallee<C, CT1>(obj, methPtr, ctxt);
        }

        void Unset(void) {
            SAFE_DELETE(this->callee);
        }

        Rv operator()(P1 p1, P2 p2, P3 p3, P4 p4) {
            if (this->callee == NULL) {
                throw vislib::IllegalStateException("Delegate target not set", __FILE__, __LINE__);
            }
            return this->callee->Call(p1, p2, p3, p4);
        }

        bool operator==(const Delegate& rhs) const {
            if (this->callee == NULL) {
                return (rhs.callee == NULL);
            }
            return this->callee->Equals(*rhs.callee);
        }

        Delegate& operator=(const Delegate& rhs) {
            SAFE_DELETE(this->callee);
            if (rhs.callee != NULL) {
                this->callee = rhs.callee->Clone();
            }
            return *this;
        }

    private:

        /**
         * base class for callee implementations
         */
        class AbstractCallee {
        public:

            /** Ctor */
            AbstractCallee(void) {
                // intentionally empty
            }

            /** Dtor */
            virtual ~AbstractCallee(void) {
                // intentionally empty
            }

            /** Call */
            virtual Rv Call(P1 p1, P2 p2, P3 p3, P4 p4) = 0;

            /**
             * Clones this object
             *
             * return A clone
             */
            virtual AbstractCallee * Clone(void) = 0;

            /**
             * Returns if this and rhs are equal
             *
             * param rhs The right hand side operand
             *
             * return True if this and rhs are equal
             */
            virtual bool Equals(const AbstractCallee& rhs) = 0;

        };

        class FunctionCallee : public AbstractCallee {
        public:
            FunctionCallee(Rv (*func)(P1, P2, P3, P4)) : AbstractCallee(), func(func) {
                // Intentionally empty
            }
            virtual ~FunctionCallee(void) {
                this->func = NULL; // DO NOT DELETE
            }
            virtual Rv Call(P1 p1, P2 p2, P3 p3, P4 p4) {
                return this->func(p1, p2, p3, p4);
            }
            virtual AbstractCallee * Clone(void) {
                return new FunctionCallee(this->func);
            }
            virtual bool Equals(const AbstractCallee& rhs) {
                const FunctionCallee *r= dynamic_cast<const FunctionCallee*>(&rhs);
                return (r != NULL)
                    && (r->func == this->func);
            }
        private:
            Rv (*func)(P1, P2, P3, P4);
        };

        template<class CT>
        class FunctionContextCallee : public AbstractCallee {
        public:
            FunctionContextCallee(Rv (*func)(P1, P2, P3, P4, CT), CT ctxt) : AbstractCallee(), func(func), ctxt(ctxt) {
                // Intentionally empty
            }
            virtual ~FunctionContextCallee(void) {
                this->func = NULL; // DO NOT DELETE
            }
            virtual Rv Call(P1 p1, P2 p2, P3 p3, P4 p4) {
                return this->func(p1, p2, p3, p4, this->ctxt);
            }
            virtual AbstractCallee * Clone(void) {
                return new FunctionContextCallee(this->func, this->ctxt);
            }
            virtual bool Equals(const AbstractCallee& rhs) {
                const FunctionContextCallee *r= dynamic_cast<const FunctionContextCallee*>(&rhs);
                return (r != NULL)
                    && (r->func == this->func)
                    && (r->ctxt == this->ctxt);
            }
        private:
            Rv (*func)(P1, P2, P3, P4, CT);
            CT ctxt;
        };

        template<class C>
        class MethodCallee : public AbstractCallee {
        public:
            MethodCallee(C& obj, Rv (C::*meth)(P1, P2, P3, P4)) : AbstractCallee(), obj(obj), meth(meth) {
                // Intentionally empty
            }
            virtual ~MethodCallee(void) {
                this->meth = NULL; // DO NOT DELETE
            }
            virtual Rv Call(P1 p1, P2 p2, P3 p3, P4 p4) {
                return (this->obj.*this->meth)(p1, p2, p3, p4);
            }
            virtual AbstractCallee * Clone(void) {
                return new MethodCallee<C>(this->obj, this->meth);
            }
            virtual bool Equals(const AbstractCallee& rhs) {
                const MethodCallee *r= dynamic_cast<const MethodCallee*>(&rhs);
                return (r != NULL)
                    && (&r->obj == &this->obj)
                    && (r->meth == this->meth);
            }
        private:
            C& obj;
            Rv (C::*meth)(P1, P2, P3, P4);
        };

        template<class C, class CT>
        class MethodContextCallee : public AbstractCallee {
        public:
            MethodContextCallee(C& obj, Rv (C::*meth)(P1, P2, P3, P4, CT), CT ctxt) : AbstractCallee(), obj(obj), meth(meth), ctxt(ctxt) {
                // Intentionally empty
            }
            virtual ~MethodContextCallee(void) {
                this->meth = NULL; // DO NOT DELETE
            }
            virtual Rv Call(P1 p1, P2 p2, P3 p3, P4 p4) {
                return (this->obj.*this->meth)(p1, p2, p3, p4, this->ctxt);
            }
            virtual AbstractCallee * Clone(void) {
                return new MethodContextCallee<C, CT>(this->obj, this->meth, this->ctxt);
            }
            virtual bool Equals(const AbstractCallee& rhs) {
                const MethodContextCallee *r= dynamic_cast<const MethodContextCallee*>(&rhs);
                return (r != NULL)
                    && (&r->obj == &this->obj)
                    && (r->meth == this->meth)
                    && (r->ctxt == this->ctxt);
            }
        private:
            C& obj;
            Rv (C::*meth)(P1, P2, P3, P4, CT);
            CT ctxt;
        };

        /** The callee target */
        AbstractCallee *callee;

    };


    /**
     * Template specialication for delegate functions with 3 parameters
     */
    template<class Rv, class P1, class P2, class P3>
    class Delegate<Rv, P1, P2, P3, void, void, void, void, void, void, void> {
    public:

        Delegate(void) : callee(NULL) {
        }

        Delegate(const Delegate& rhs) : callee(NULL) {
            (*this) = rhs;
        }

        Delegate(Rv (*funcPtr)(P1, P2, P3))
                : callee(new FunctionCallee(funcPtr)) {
        }

        template<class CT1, class CT2>
        Delegate(Rv (*funcPtr)(P1, P2, P3, CT1), CT2 ctxt)
                : callee(new FunctionContextCallee<CT1>(funcPtr, ctxt)) {
        }

        template<class C>
        Delegate(C& obj, Rv (C::*methPtr)(P1, P2, P3))
                : callee(new MethodCallee<C>(obj, methPtr)) {
        }

        template<class C, class CT1, class CT2>
        Delegate(C& obj, Rv (C::*methPtr)(P1, P2, P3, CT1), CT2 ctxt)
                : callee(new MethodContextCallee<C, CT1>(obj, methPtr, ctxt)) {
        }

        ~Delegate(void) {
            SAFE_DELETE(this->callee);
        }

        inline bool IsTargetSet(void) const {
            return this->callee != NULL;
        }

        void Set(Rv (*funcPtr)(P1, P2, P3)) {
            SAFE_DELETE(this->callee);
            this->callee = new FunctionCallee(funcPtr);
        }

        template<class CT1, class CT2>
        void Set(Rv (*funcPtr)(P1, P2, P3, CT1), CT2 ctxt) {
            SAFE_DELETE(this->callee);
            this->callee = new FunctionContextCallee<CT1>(funcPtr, ctxt);
        }

        template<class C>
        void Set(C& obj, Rv (C::*methPtr)(P1, P2, P3)) {
            SAFE_DELETE(this->callee);
            this->callee = new MethodCallee<C>(obj, methPtr);
        }

        template<class C, class CT1, class CT2>
        void Set(C& obj, Rv (C::*methPtr)(P1, P2, P3, CT1), CT2 ctxt) {
            SAFE_DELETE(this->callee);
            this->callee = new MethodContextCallee<C, CT1>(obj, methPtr, ctxt);
        }

        void Unset(void) {
            SAFE_DELETE(this->callee);
        }

        Rv operator()(P1 p1, P2 p2, P3 p3) {
            if (this->callee == NULL) {
                throw vislib::IllegalStateException("Delegate target not set", __FILE__, __LINE__);
            }
            return this->callee->Call(p1, p2, p3);
        }

        bool operator==(const Delegate& rhs) const {
            if (this->callee == NULL) {
                return (rhs.callee == NULL);
            }
            return this->callee->Equals(*rhs.callee);
        }

        Delegate& operator=(const Delegate& rhs) {
            SAFE_DELETE(this->callee);
            if (rhs.callee != NULL) {
                this->callee = rhs.callee->Clone();
            }
            return *this;
        }

    private:

        /**
         * base class for callee implementations
         */
        class AbstractCallee {
        public:

            /** Ctor */
            AbstractCallee(void) {
                // intentionally empty
            }

            /** Dtor */
            virtual ~AbstractCallee(void) {
                // intentionally empty
            }

            /** Call */
            virtual Rv Call(P1 p1, P2 p2, P3 p3) = 0;

            /**
             * Clones this object
             *
             * return A clone
             */
            virtual AbstractCallee * Clone(void) = 0;

            /**
             * Returns if this and rhs are equal
             *
             * param rhs The right hand side operand
             *
             * return True if this and rhs are equal
             */
            virtual bool Equals(const AbstractCallee& rhs) = 0;

        };

        class FunctionCallee : public AbstractCallee {
        public:
            FunctionCallee(Rv (*func)(P1, P2, P3)) : AbstractCallee(), func(func) {
                // Intentionally empty
            }
            virtual ~FunctionCallee(void) {
                this->func = NULL; // DO NOT DELETE
            }
            virtual Rv Call(P1 p1, P2 p2, P3 p3) {
                return this->func(p1, p2, p3);
            }
            virtual AbstractCallee * Clone(void) {
                return new FunctionCallee(this->func);
            }
            virtual bool Equals(const AbstractCallee& rhs) {
                const FunctionCallee *r= dynamic_cast<const FunctionCallee*>(&rhs);
                return (r != NULL)
                    && (r->func == this->func);
            }
        private:
            Rv (*func)(P1, P2, P3);
        };

        template<class CT>
        class FunctionContextCallee : public AbstractCallee {
        public:
            FunctionContextCallee(Rv (*func)(P1, P2, P3, CT), CT ctxt) : AbstractCallee(), func(func), ctxt(ctxt) {
                // Intentionally empty
            }
            virtual ~FunctionContextCallee(void) {
                this->func = NULL; // DO NOT DELETE
            }
            virtual Rv Call(P1 p1, P2 p2, P3 p3) {
                return this->func(p1, p2, p3, this->ctxt);
            }
            virtual AbstractCallee * Clone(void) {
                return new FunctionContextCallee(this->func, this->ctxt);
            }
            virtual bool Equals(const AbstractCallee& rhs) {
                const FunctionContextCallee *r= dynamic_cast<const FunctionContextCallee*>(&rhs);
                return (r != NULL)
                    && (r->func == this->func)
                    && (r->ctxt == this->ctxt);
            }
        private:
            Rv (*func)(P1, P2, P3, CT);
            CT ctxt;
        };

        template<class C>
        class MethodCallee : public AbstractCallee {
        public:
            MethodCallee(C& obj, Rv (C::*meth)(P1, P2, P3)) : AbstractCallee(), obj(obj), meth(meth) {
                // Intentionally empty
            }
            virtual ~MethodCallee(void) {
                this->meth = NULL; // DO NOT DELETE
            }
            virtual Rv Call(P1 p1, P2 p2, P3 p3) {
                return (this->obj.*this->meth)(p1, p2, p3);
            }
            virtual AbstractCallee * Clone(void) {
                return new MethodCallee<C>(this->obj, this->meth);
            }
            virtual bool Equals(const AbstractCallee& rhs) {
                const MethodCallee *r= dynamic_cast<const MethodCallee*>(&rhs);
                return (r != NULL)
                    && (&r->obj == &this->obj)
                    && (r->meth == this->meth);
            }
        private:
            C& obj;
            Rv (C::*meth)(P1, P2, P3);
        };

        template<class C, class CT>
        class MethodContextCallee : public AbstractCallee {
        public:
            MethodContextCallee(C& obj, Rv (C::*meth)(P1, P2, P3, CT), CT ctxt) : AbstractCallee(), obj(obj), meth(meth), ctxt(ctxt) {
                // Intentionally empty
            }
            virtual ~MethodContextCallee(void) {
                this->meth = NULL; // DO NOT DELETE
            }
            virtual Rv Call(P1 p1, P2 p2, P3 p3) {
                return (this->obj.*this->meth)(p1, p2, p3, this->ctxt);
            }
            virtual AbstractCallee * Clone(void) {
                return new MethodContextCallee<C, CT>(this->obj, this->meth, this->ctxt);
            }
            virtual bool Equals(const AbstractCallee& rhs) {
                const MethodContextCallee *r= dynamic_cast<const MethodContextCallee*>(&rhs);
                return (r != NULL)
                    && (&r->obj == &this->obj)
                    && (r->meth == this->meth)
                    && (r->ctxt == this->ctxt);
            }
        private:
            C& obj;
            Rv (C::*meth)(P1, P2, P3, CT);
            CT ctxt;
        };

        /** The callee target */
        AbstractCallee *callee;

    };


    /**
     * Template specialication for delegate functions with 2 parameters
     */
    template<class Rv, class P1, class P2>
    class Delegate<Rv, P1, P2, void, void, void, void, void, void, void, void> {
    public:

        Delegate(void) : callee(NULL) {
        }

        Delegate(const Delegate& rhs) : callee(NULL) {
            (*this) = rhs;
        }

        Delegate(Rv (*funcPtr)(P1, P2))
                : callee(new FunctionCallee(funcPtr)) {
        }

        template<class CT1, class CT2>
        Delegate(Rv (*funcPtr)(P1, P2, CT1), CT2 ctxt)
                : callee(new FunctionContextCallee<CT1>(funcPtr, ctxt)) {
        }

        template<class C>
        Delegate(C& obj, Rv (C::*methPtr)(P1, P2))
                : callee(new MethodCallee<C>(obj, methPtr)) {
        }

        template<class C, class CT1, class CT2>
        Delegate(C& obj, Rv (C::*methPtr)(P1, P2, CT1), CT2 ctxt)
                : callee(new MethodContextCallee<C, CT1>(obj, methPtr, ctxt)) {
        }

        ~Delegate(void) {
            SAFE_DELETE(this->callee);
        }

        inline bool IsTargetSet(void) const {
            return this->callee != NULL;
        }

        void Set(Rv (*funcPtr)(P1, P2)) {
            SAFE_DELETE(this->callee);
            this->callee = new FunctionCallee(funcPtr);
        }

        template<class CT1, class CT2>
        void Set(Rv (*funcPtr)(P1, P2, CT1), CT2 ctxt) {
            SAFE_DELETE(this->callee);
            this->callee = new FunctionContextCallee<CT1>(funcPtr, ctxt);
        }

        template<class C>
        void Set(C& obj, Rv (C::*methPtr)(P1, P2)) {
            SAFE_DELETE(this->callee);
            this->callee = new MethodCallee<C>(obj, methPtr);
        }

        template<class C, class CT1, class CT2>
        void Set(C& obj, Rv (C::*methPtr)(P1, P2, CT1), CT2 ctxt) {
            SAFE_DELETE(this->callee);
            this->callee = new MethodContextCallee<C, CT1>(obj, methPtr, ctxt);
        }

        void Unset(void) {
            SAFE_DELETE(this->callee);
        }

        Rv operator()(P1 p1, P2 p2) {
            if (this->callee == NULL) {
                throw vislib::IllegalStateException("Delegate target not set", __FILE__, __LINE__);
            }
            return this->callee->Call(p1, p2);
        }

        bool operator==(const Delegate& rhs) const {
            if (this->callee == NULL) {
                return (rhs.callee == NULL);
            }
            return this->callee->Equals(*rhs.callee);
        }

        Delegate& operator=(const Delegate& rhs) {
            SAFE_DELETE(this->callee);
            if (rhs.callee != NULL) {
                this->callee = rhs.callee->Clone();
            }
            return *this;
        }

    private:

        /**
         * base class for callee implementations
         */
        class AbstractCallee {
        public:

            /** Ctor */
            AbstractCallee(void) {
                // intentionally empty
            }

            /** Dtor */
            virtual ~AbstractCallee(void) {
                // intentionally empty
            }

            /** Call */
            virtual Rv Call(P1 p1, P2 p2) = 0;

            /**
             * Clones this object
             *
             * return A clone
             */
            virtual AbstractCallee * Clone(void) = 0;

            /**
             * Returns if this and rhs are equal
             *
             * param rhs The right hand side operand
             *
             * return True if this and rhs are equal
             */
            virtual bool Equals(const AbstractCallee& rhs) = 0;

        };

        class FunctionCallee : public AbstractCallee {
        public:
            FunctionCallee(Rv (*func)(P1, P2)) : AbstractCallee(), func(func) {
                // Intentionally empty
            }
            virtual ~FunctionCallee(void) {
                this->func = NULL; // DO NOT DELETE
            }
            virtual Rv Call(P1 p1, P2 p2) {
                return this->func(p1, p2);
            }
            virtual AbstractCallee * Clone(void) {
                return new FunctionCallee(this->func);
            }
            virtual bool Equals(const AbstractCallee& rhs) {
                const FunctionCallee *r= dynamic_cast<const FunctionCallee*>(&rhs);
                return (r != NULL)
                    && (r->func == this->func);
            }
        private:
            Rv (*func)(P1, P2);
        };

        template<class CT>
        class FunctionContextCallee : public AbstractCallee {
        public:
            FunctionContextCallee(Rv (*func)(P1, P2, CT), CT ctxt) : AbstractCallee(), func(func), ctxt(ctxt) {
                // Intentionally empty
            }
            virtual ~FunctionContextCallee(void) {
                this->func = NULL; // DO NOT DELETE
            }
            virtual Rv Call(P1 p1, P2 p2) {
                return this->func(p1, p2, this->ctxt);
            }
            virtual AbstractCallee * Clone(void) {
                return new FunctionContextCallee(this->func, this->ctxt);
            }
            virtual bool Equals(const AbstractCallee& rhs) {
                const FunctionContextCallee *r= dynamic_cast<const FunctionContextCallee*>(&rhs);
                return (r != NULL)
                    && (r->func == this->func)
                    && (r->ctxt == this->ctxt);
            }
        private:
            Rv (*func)(P1, P2, CT);
            CT ctxt;
        };

        template<class C>
        class MethodCallee : public AbstractCallee {
        public:
            MethodCallee(C& obj, Rv (C::*meth)(P1, P2)) : AbstractCallee(), obj(obj), meth(meth) {
                // Intentionally empty
            }
            virtual ~MethodCallee(void) {
                this->meth = NULL; // DO NOT DELETE
            }
            virtual Rv Call(P1 p1, P2 p2) {
                return (this->obj.*this->meth)(p1, p2);
            }
            virtual AbstractCallee * Clone(void) {
                return new MethodCallee<C>(this->obj, this->meth);
            }
            virtual bool Equals(const AbstractCallee& rhs) {
                const MethodCallee *r= dynamic_cast<const MethodCallee*>(&rhs);
                return (r != NULL)
                    && (&r->obj == &this->obj)
                    && (r->meth == this->meth);
            }
        private:
            C& obj;
            Rv (C::*meth)(P1, P2);
        };

        template<class C, class CT>
        class MethodContextCallee : public AbstractCallee {
        public:
            MethodContextCallee(C& obj, Rv (C::*meth)(P1, P2, CT), CT ctxt) : AbstractCallee(), obj(obj), meth(meth), ctxt(ctxt) {
                // Intentionally empty
            }
            virtual ~MethodContextCallee(void) {
                this->meth = NULL; // DO NOT DELETE
            }
            virtual Rv Call(P1 p1, P2 p2) {
                return (this->obj.*this->meth)(p1, p2, this->ctxt);
            }
            virtual AbstractCallee * Clone(void) {
                return new MethodContextCallee<C, CT>(this->obj, this->meth, this->ctxt);
            }
            virtual bool Equals(const AbstractCallee& rhs) {
                const MethodContextCallee *r= dynamic_cast<const MethodContextCallee*>(&rhs);
                return (r != NULL)
                    && (&r->obj == &this->obj)
                    && (r->meth == this->meth)
                    && (r->ctxt == this->ctxt);
            }
        private:
            C& obj;
            Rv (C::*meth)(P1, P2, CT);
            CT ctxt;
        };

        /** The callee target */
        AbstractCallee *callee;

    };


    /**
     * Template specialication for delegate functions with 1 parameters
     */
    template<class Rv, class P1>
    class Delegate<Rv, P1, void, void, void, void, void, void, void, void, void> {
    public:

        Delegate(void) : callee(NULL) {
        }

        Delegate(const Delegate& rhs) : callee(NULL) {
            (*this) = rhs;
        }

        Delegate(Rv (*funcPtr)(P1))
                : callee(new FunctionCallee(funcPtr)) {
        }

        template<class CT1, class CT2>
        Delegate(Rv (*funcPtr)(P1, CT1), CT2 ctxt)
                : callee(new FunctionContextCallee<CT1>(funcPtr, ctxt)) {
        }

        template<class C>
        Delegate(C& obj, Rv (C::*methPtr)(P1))
                : callee(new MethodCallee<C>(obj, methPtr)) {
        }

        template<class C, class CT1, class CT2>
        Delegate(C& obj, Rv (C::*methPtr)(P1, CT1), CT2 ctxt)
                : callee(new MethodContextCallee<C, CT1>(obj, methPtr, ctxt)) {
        }

        ~Delegate(void) {
            SAFE_DELETE(this->callee);
        }

        inline bool IsTargetSet(void) const {
            return this->callee != NULL;
        }

        void Set(Rv (*funcPtr)(P1)) {
            SAFE_DELETE(this->callee);
            this->callee = new FunctionCallee(funcPtr);
        }

        template<class CT1, class CT2>
        void Set(Rv (*funcPtr)(P1, CT1), CT2 ctxt) {
            SAFE_DELETE(this->callee);
            this->callee = new FunctionContextCallee<CT1>(funcPtr, ctxt);
        }

        template<class C>
        void Set(C& obj, Rv (C::*methPtr)(P1)) {
            SAFE_DELETE(this->callee);
            this->callee = new MethodCallee<C>(obj, methPtr);
        }

        template<class C, class CT1, class CT2>
        void Set(C& obj, Rv (C::*methPtr)(P1, CT1), CT2 ctxt) {
            SAFE_DELETE(this->callee);
            this->callee = new MethodContextCallee<C, CT1>(obj, methPtr, ctxt);
        }

        void Unset(void) {
            SAFE_DELETE(this->callee);
        }

        Rv operator()(P1 p1) {
            if (this->callee == NULL) {
                throw vislib::IllegalStateException("Delegate target not set", __FILE__, __LINE__);
            }
            return this->callee->Call(p1);
        }

        bool operator==(const Delegate& rhs) const {
            if (this->callee == NULL) {
                return (rhs.callee == NULL);
            }
            return this->callee->Equals(*rhs.callee);
        }

        Delegate& operator=(const Delegate& rhs) {
            SAFE_DELETE(this->callee);
            if (rhs.callee != NULL) {
                this->callee = rhs.callee->Clone();
            }
            return *this;
        }

    private:

        /**
         * base class for callee implementations
         */
        class AbstractCallee {
        public:

            /** Ctor */
            AbstractCallee(void) {
                // intentionally empty
            }

            /** Dtor */
            virtual ~AbstractCallee(void) {
                // intentionally empty
            }

            /** Call */
            virtual Rv Call(P1 p1) = 0;

            /**
             * Clones this object
             *
             * return A clone
             */
            virtual AbstractCallee * Clone(void) = 0;

            /**
             * Returns if this and rhs are equal
             *
             * param rhs The right hand side operand
             *
             * return True if this and rhs are equal
             */
            virtual bool Equals(const AbstractCallee& rhs) = 0;

        };

        class FunctionCallee : public AbstractCallee {
        public:
            FunctionCallee(Rv (*func)(P1)) : AbstractCallee(), func(func) {
                // Intentionally empty
            }
            virtual ~FunctionCallee(void) {
                this->func = NULL; // DO NOT DELETE
            }
            virtual Rv Call(P1 p1) {
                return this->func(p1);
            }
            virtual AbstractCallee * Clone(void) {
                return new FunctionCallee(this->func);
            }
            virtual bool Equals(const AbstractCallee& rhs) {
                const FunctionCallee *r= dynamic_cast<const FunctionCallee*>(&rhs);
                return (r != NULL)
                    && (r->func == this->func);
            }
        private:
            Rv (*func)(P1);
        };

        template<class CT>
        class FunctionContextCallee : public AbstractCallee {
        public:
            FunctionContextCallee(Rv (*func)(P1, CT), CT ctxt) : AbstractCallee(), func(func), ctxt(ctxt) {
                // Intentionally empty
            }
            virtual ~FunctionContextCallee(void) {
                this->func = NULL; // DO NOT DELETE
            }
            virtual Rv Call(P1 p1) {
                return this->func(p1, this->ctxt);
            }
            virtual AbstractCallee * Clone(void) {
                return new FunctionContextCallee(this->func, this->ctxt);
            }
            virtual bool Equals(const AbstractCallee& rhs) {
                const FunctionContextCallee *r= dynamic_cast<const FunctionContextCallee*>(&rhs);
                return (r != NULL)
                    && (r->func == this->func)
                    && (r->ctxt == this->ctxt);
            }
        private:
            Rv (*func)(P1, CT);
            CT ctxt;
        };

        template<class C>
        class MethodCallee : public AbstractCallee {
        public:
            MethodCallee(C& obj, Rv (C::*meth)(P1)) : AbstractCallee(), obj(obj), meth(meth) {
                // Intentionally empty
            }
            virtual ~MethodCallee(void) {
                this->meth = NULL; // DO NOT DELETE
            }
            virtual Rv Call(P1 p1) {
                return (this->obj.*this->meth)(p1);
            }
            virtual AbstractCallee * Clone(void) {
                return new MethodCallee<C>(this->obj, this->meth);
            }
            virtual bool Equals(const AbstractCallee& rhs) {
                const MethodCallee *r= dynamic_cast<const MethodCallee*>(&rhs);
                return (r != NULL)
                    && (&r->obj == &this->obj)
                    && (r->meth == this->meth);
            }
        private:
            C& obj;
            Rv (C::*meth)(P1);
        };

        template<class C, class CT>
        class MethodContextCallee : public AbstractCallee {
        public:
            MethodContextCallee(C& obj, Rv (C::*meth)(P1, CT), CT ctxt) : AbstractCallee(), obj(obj), meth(meth), ctxt(ctxt) {
                // Intentionally empty
            }
            virtual ~MethodContextCallee(void) {
                this->meth = NULL; // DO NOT DELETE
            }
            virtual Rv Call(P1 p1) {
                return (this->obj.*this->meth)(p1, this->ctxt);
            }
            virtual AbstractCallee * Clone(void) {
                return new MethodContextCallee<C, CT>(this->obj, this->meth, this->ctxt);
            }
            virtual bool Equals(const AbstractCallee& rhs) {
                const MethodContextCallee *r= dynamic_cast<const MethodContextCallee*>(&rhs);
                return (r != NULL)
                    && (&r->obj == &this->obj)
                    && (r->meth == this->meth)
                    && (r->ctxt == this->ctxt);
            }
        private:
            C& obj;
            Rv (C::*meth)(P1, CT);
            CT ctxt;
        };

        /** The callee target */
        AbstractCallee *callee;

    };


    /**
     * Template specialication for delegate functions without parameters
     */
    template<>
    class Delegate<void, void, void, void, void, void, void, void, void, void> {
    public:

        Delegate(void) : callee(NULL) {
        }

        Delegate(const Delegate& rhs) : callee(NULL) {
            (*this) = rhs;
        }

        Delegate(void (*funcPtr)(void))
                : callee(new FunctionCallee(funcPtr)) {
        }

        template<class CT1, class CT2>
        Delegate(void (*funcPtr)(CT1), CT2 ctxt)
                : callee(new FunctionContextCallee<CT1>(funcPtr, ctxt)) {
        }

        template<class C>
        Delegate(C& obj, void (C::*methPtr)(void))
                : callee(new MethodCallee<C>(obj, methPtr)) {
        }

        template<class C, class CT1, class CT2>
        Delegate(C& obj, void (C::*methPtr)(CT1), CT2 ctxt)
                : callee(new MethodContextCallee<C, CT1>(obj, methPtr, ctxt)) {
        }

        ~Delegate(void) {
            SAFE_DELETE(this->callee);
        }

        inline bool IsTargetSet(void) const {
            return this->callee != NULL;
        }

        void Set(void (*funcPtr)(void)) {
            SAFE_DELETE(this->callee);
            this->callee = new FunctionCallee(funcPtr);
        }

        template<class CT1, class CT2>
        void Set(void (*funcPtr)(CT1), CT2 ctxt) {
            SAFE_DELETE(this->callee);
            this->callee = new FunctionContextCallee<CT1>(funcPtr, ctxt);
        }

        template<class C>
        void Set(C& obj, void (C::*methPtr)(void)) {
            SAFE_DELETE(this->callee);
            this->callee = new MethodCallee<C>(obj, methPtr);
        }

        template<class C, class CT1, class CT2>
        void Set(C& obj, void (C::*methPtr)(CT1), CT2 ctxt) {
            SAFE_DELETE(this->callee);
            this->callee = new MethodContextCallee<C, CT1>(obj, methPtr, ctxt);
        }

        void Unset(void) {
            SAFE_DELETE(this->callee);
        }

        void operator()(void) {
            if (this->callee == NULL) {
                throw vislib::IllegalStateException("Delegate target not set", __FILE__, __LINE__);
            }
            this->callee->Call();
        }

        bool operator==(const Delegate& rhs) const {
            if (this->callee == NULL) {
                return (rhs.callee == NULL);
            }
            return this->callee->Equals(*rhs.callee);
        }

        Delegate& operator=(const Delegate& rhs) {
            SAFE_DELETE(this->callee);
            if (rhs.callee != NULL) {
                this->callee = rhs.callee->Clone();
            }
            return *this;
        }

    private:

        /**
         * base class for callee implementations
         */
        class AbstractCallee {
        public:

            /** Ctor */
            AbstractCallee(void) {
                // intentionally empty
            }

            /** Dtor */
            virtual ~AbstractCallee(void) {
                // intentionally empty
            }

            /** Call */
            virtual void Call(void) = 0;

            /**
             * Clones this object
             *
             * return A clone
             */
            virtual AbstractCallee * Clone(void) = 0;

            /**
             * Returns if this and rhs are equal
             *
             * param rhs The right hand side operand
             *
             * return True if this and rhs are equal
             */
            virtual bool Equals(const AbstractCallee& rhs) = 0;

        };

        class FunctionCallee : public AbstractCallee {
        public:
            FunctionCallee(void (*func)(void)) : AbstractCallee(), func(func) {
                // Intentionally empty
            }
            virtual ~FunctionCallee(void) {
                this->func = NULL; // DO NOT DELETE
            }
            virtual void Call(void) {
                this->func();
            }
            virtual AbstractCallee * Clone(void) {
                return new FunctionCallee(this->func);
            }
            virtual bool Equals(const AbstractCallee& rhs) {
                const FunctionCallee *r= dynamic_cast<const FunctionCallee*>(&rhs);
                return (r != NULL)
                    && (r->func == this->func);
            }
        private:
            void (*func)(void);
        };

        template<class CT>
        class FunctionContextCallee : public AbstractCallee {
        public:
            FunctionContextCallee(void (*func)(CT), CT ctxt) : AbstractCallee(), func(func), ctxt(ctxt) {
                // Intentionally empty
            }
            virtual ~FunctionContextCallee(void) {
                this->func = NULL; // DO NOT DELETE
            }
            virtual void Call(void) {
                this->func(this->ctxt);
            }
            virtual AbstractCallee * Clone(void) {
                return new FunctionContextCallee(this->func, this->ctxt);
            }
            virtual bool Equals(const AbstractCallee& rhs) {
                const FunctionContextCallee *r= dynamic_cast<const FunctionContextCallee*>(&rhs);
                return (r != NULL)
                    && (r->func == this->func)
                    && (r->ctxt == this->ctxt);
            }
        private:
            void (*func)(CT);
            CT ctxt;
        };

        template<class C>
        class MethodCallee : public AbstractCallee {
        public:
            MethodCallee(C& obj, void (C::*meth)(void)) : AbstractCallee(), obj(obj), meth(meth) {
                // Intentionally empty
            }
            virtual ~MethodCallee(void) {
                this->meth = NULL; // DO NOT DELETE
            }
            virtual void Call(void) {
                (this->obj.*this->meth)();
            }
            virtual AbstractCallee * Clone(void) {
                return new MethodCallee<C>(this->obj, this->meth);
            }
            virtual bool Equals(const AbstractCallee& rhs) {
                const MethodCallee *r= dynamic_cast<const MethodCallee*>(&rhs);
                return (r != NULL)
                    && (&r->obj == &this->obj)
                    && (r->meth == this->meth);
            }
        private:
            C& obj;
            void (C::*meth)(void);
        };

        template<class C, class CT>
        class MethodContextCallee : public AbstractCallee {
        public:
            MethodContextCallee(C& obj, void (C::*meth)(CT), CT ctxt) : AbstractCallee(), obj(obj), meth(meth), ctxt(ctxt) {
                // Intentionally empty
            }
            virtual ~MethodContextCallee(void) {
                this->meth = NULL; // DO NOT DELETE
            }
            virtual void Call(void) {
                (this->obj.*this->meth)(this->ctxt);
            }
            virtual AbstractCallee * Clone(void) {
                return new MethodContextCallee<C, CT>(this->obj, this->meth, this->ctxt);
            }
            virtual bool Equals(const AbstractCallee& rhs) {
                const MethodContextCallee *r= dynamic_cast<const MethodContextCallee*>(&rhs);
                return (r != NULL)
                    && (&r->obj == &this->obj)
                    && (r->meth == this->meth)
                    && (r->ctxt == this->ctxt);
            }
        private:
            C& obj;
            void (C::*meth)(CT);
            CT ctxt;
        };

        /** The callee target */
        AbstractCallee *callee;

    };


} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_DELEGATE_H_INCLUDED */
