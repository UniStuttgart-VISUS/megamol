#
# implementDelegate.pl
#
# Copyright (C) 2011 by VISUS (Universitaet Stuttgart).
# Alle Rechte vorbehalten.
#
use strict;


#
# Setup:
##############################################################################
my $outputFile = "include/vislib/Delegate.h";
my $paramCnt = 10;


#
# Output:
##############################################################################
open (my $out, ">$outputFile") || die "Cannot write output file \"$outputFile\"\n";


#
# Functions
##############################################################################
sub WriteClass {
    # parameters
    my $out = shift;
    my $paramCnt = shift;
    my $pCnt = shift;

    # class header
    if ($paramCnt eq $pCnt) {
        print $out "\n    template<class Rv = void";
        for (my $i = 1; $i <= $paramCnt; $i++) {
            print $out ", class P$i = void";
        }
        print $out ">\n    class Delegate {";
    } else {
        print $out "\n    template<";
        if ($pCnt >= 0) {
            print $out "class Rv";
            for (my $i = 1; $i <= $pCnt; $i++) {
                print $out ", class P$i";
            }
        }
        print $out ">\n    class Delegate<";
        if ($pCnt >= 0) {
            print $out "Rv";
        } else {
            print $out "void";
        }
        for (my $i = 1; $i <= $pCnt; $i++) {
            print $out ", P$i";
        }
        for (my $i = $pCnt; $i < $paramCnt; $i++) {
            if ($i >= 0) {
                print $out ", void";
            }
        }
        print $out "> {";
    }

    # text strings
    my $paramList = "";
    my $paramTypeList = "";
    my $paramNameList = "";
    my $returnType = ($pCnt >= 0) ? "Rv" : "void";
    my $returnStatement = ($pCnt >= 0) ? "return " : "";
    if ($pCnt > 0) {
        for (my $i = 1; $i <= $pCnt; $i++) {
            if ($i > 1) {
                $paramList .= ", ";
                $paramTypeList .= ", ";
                $paramNameList .= ", ";
            }
            $paramList .= "P$i p$i";
            $paramTypeList .= "P$i";
            $paramNameList .= "p$i";
        }
    } else {
        $paramList = "void";
        $paramTypeList = "void";
        $paramNameList = "";
    }
    my $paramCtxtTypeList = $paramTypeList;
    my $paramCtxtNameList = $paramNameList;
    if ($paramCtxtTypeList eq "void") {
        $paramCtxtTypeList = "CT1";
        $paramCtxtNameList = "this->ctxt";
    } else {
        $paramCtxtTypeList .= ", CT1";
        $paramCtxtNameList .= ", this->ctxt";
    }

    # class implementation
    print $out qq§
    public:

        /**
         * Ctor
         */
        Delegate(void) : callee(NULL) {
            // intentionally empty
        }

        /**
         * Copy ctor
         *
         * \@param src The object to clone from
         */
        Delegate(const Delegate& src) : callee(NULL) {
            (*this) = src;
        }

        /**
         * Ctor
         *
         * \@param funcPtr Function pointer to be set
         */
        Delegate($returnType (*funcPtr)($paramTypeList))
                : callee((funcPtr == NULL)
                    ? NULL
                    : new FunctionCallee(funcPtr)) {
            // intentionally empty
        }

        /**
         * Ctor
         *
         * \@param funcPtr Function pointer to be set
         * \@param ctxt The user data context used when calling the function
         */
        template<class CT1, class CT2>
        Delegate($returnType (*funcPtr)($paramCtxtTypeList), CT2 ctxt)
                : callee((funcPtr == NULL)
                    ? NULL
                    : new FunctionContextCallee<CT1>(funcPtr, ctxt)) {
            // intentionally empty
        }

        /**
         * Ctor
         *
         * \@param obj The object to call the member of
         * \@param methPtr The method class pointer to be set (must not be NULL)
         */
        template<class C>
        Delegate(C& obj, $returnType (C::*methPtr)($paramTypeList))
                : callee(new MethodCallee<C>(obj, methPtr)) {
            // intentionally empty
        }

        /**
         * Ctor
         *
         * \@param obj The object to call the member of
         * \@param methPtr The method class pointer to be set (must not be NULL)
         * \@param ctxt The user data context used when calling the method
         */
        template<class C, class CT1, class CT2>
        Delegate(C& obj, $returnType (C::*methPtr)($paramCtxtTypeList), CT2 ctxt)
                : callee(new MethodContextCallee<C, CT1>(obj, methPtr, ctxt)) {
            // intentionally empty
        }

        /**
         * Dtor
         *
         * Note that no memory will be freed (user data context, etc.)
         */
        ~Delegate(void) {
            SAFE_DELETE(this->callee);
        }

        /**
         * Answer whether the target for this delegate is set
         *
         * \@return True if the target for this delegate is set
         */
        inline bool IsTargetSet(void) const {
            return this->callee != NULL;
        }

        /**
         * Sets the target for this delegate
         *
         * \@param funcPtr Function pointer to be set
         */
        void Set($returnType (*funcPtr)($paramTypeList)) {
            SAFE_DELETE(this->callee);
            if (funcPtr != NULL) {
                this->callee = new FunctionCallee(funcPtr);
            }
        }

        /**
         * Sets the target for this delegate
         *
         * \@param funcPtr Function pointer to be set
         * \@param ctxt The user data context used when calling the function
         */
        template<class CT1, class CT2>
        void Set($returnType (*funcPtr)($paramCtxtTypeList), CT2 ctxt) {
            SAFE_DELETE(this->callee);
            if (funcPtr != NULL) {
                this->callee = new FunctionContextCallee<CT1>(funcPtr, ctxt);
            }
        }

        /**
         * Sets the target for this delegate
         *
         * \@param obj The object to call the member of
         * \@param methPtr The method class pointer to be set (must not be NULL)
         */
        template<class C>
        void Set(C& obj, $returnType (C::*methPtr)($paramTypeList)) {
            SAFE_DELETE(this->callee);
            this->callee = new MethodCallee<C>(obj, methPtr);
        }

        /**
         * Sets the target for this delegate
         *
         * \@param obj The object to call the member of
         * \@param methPtr The method class pointer to be set (must not be NULL)
         * \@param ctxt The user data context used when calling the method
         */
        template<class C, class CT1, class CT2>
        void Set(C& obj, $returnType (C::*methPtr)($paramCtxtTypeList), CT2 ctxt) {
            SAFE_DELETE(this->callee);
            this->callee = new MethodContextCallee<C, CT1>(obj, methPtr, ctxt);
        }

        /**
         * Unsets the target for this delegate
         *
         * Note that no memory will be freed (user data context, etc.)
         */
        void Unset(void) {
            SAFE_DELETE(this->callee);
        }

        /**
         * Calls the delegate's target
         *
         * Parameter and return value depend on the template arguments of the delegate
         */
        $returnType operator()($paramList) {
            if (this->callee == NULL) {
                throw vislib::IllegalStateException("Delegate target not set", __FILE__, __LINE__);
            }
            ${returnStatement}this->callee->Call($paramNameList);
        }

        /**
         * Test for equality
         *
         * \@param rhs The right hand side operand
         *
         * \@return True if this and rhs are equal
         */
        bool operator==(const Delegate& rhs) const {
            if (this->callee == NULL) {
                return (rhs.callee == NULL);
            }
            return this->callee->Equals(*rhs.callee);
        }

        /**
         * Assignment operator
         *
         * \@param rhs The right hand side operand
         *
         * \@return A reference to this object
         */
        Delegate& operator=(const Delegate& rhs) {
            SAFE_DELETE(this->callee);
            if (rhs.callee != NULL) {
                this->callee = rhs.callee->Clone();
            }
            return *this;
        }

        /**
         * Sets the target for this delegate
         *
         * \@param funcPtr Function pointer to be set
         *
         * \@return A reference to this object
         */
        Delegate& operator=($returnType (*funcPtr)($paramTypeList)) {
            this->Set(funcPtr);
            return *this;
        }

    private:

        /**
         * abstract base class for callee implementations
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
            virtual $returnType Call($paramList) = 0;

            /**
             * Clones this object
             *
             * \@return A clone
             */
            virtual AbstractCallee * Clone(void) = 0;

            /**
             * Test for equality
             *
             * \@param rhs The right hand side operand
             *
             * \@return True if this and rhs are equal
             */
            virtual bool Equals(const AbstractCallee& rhs) = 0;

        };

        /**
         * Callee implementation for functions
         */
        class FunctionCallee : public AbstractCallee {
        public:

            /**
             * Ctor
             *
             * \@param func The function pointer (must not be NULL)
             */
            FunctionCallee($returnType (*func)($paramTypeList)) : AbstractCallee(), func(func) {
                ASSERT(this->func != NULL);
            }

            /** Dtor */
            virtual ~FunctionCallee(void) {
                this->func = NULL; // DO NOT DELETE
            }

            /** Call */
            virtual $returnType Call($paramList) {
                ${returnStatement}this->func($paramNameList);
            }

            /**
             * Clones this object
             *
             * \@return A clone
             */
            virtual AbstractCallee * Clone(void) {
                return new FunctionCallee(this->func);
            }

            /**
             * Test for equality
             *
             * \@param rhs The right hand side operand
             *
             * \@return True if this and rhs are equal
             */
            virtual bool Equals(const AbstractCallee& rhs) {
                const FunctionCallee *r= dynamic_cast<const FunctionCallee*>(&rhs);
                return (r != NULL)
                    && (r->func == this->func);
            }

        private:

            /** The function pointer */
            $returnType (*func)($paramTypeList);

        };

        /**
         * Callee implementation for functions with user data context
         */
        template<class CT1>
        class FunctionContextCallee : public AbstractCallee {
        public:

            /**
             * Ctor
             *
             * \@param func The function pointer (must not be NULL)
             * \@param ctxt The user data context
             */
            FunctionContextCallee($returnType (*func)($paramCtxtTypeList), CT1 ctxt) : AbstractCallee(), func(func), ctxt(ctxt) {
                ASSERT(this->func != NULL);
            }

            /** Dtor */
            virtual ~FunctionContextCallee(void) {
                this->func = NULL; // DO NOT DELETE
            }

            /** Call */
            virtual $returnType Call($paramList) {
                ${returnStatement}this->func($paramCtxtNameList);
            }

            /**
             * Clones this object
             *
             * \@return A clone
             */
            virtual AbstractCallee * Clone(void) {
                return new FunctionContextCallee<CT1>(this->func, this->ctxt);
            }

            /**
             * Test for equality
             *
             * \@param rhs The right hand side operand
             *
             * \@return True if this and rhs are equal
             */
            virtual bool Equals(const AbstractCallee& rhs) {
                const FunctionContextCallee *r= dynamic_cast<const FunctionContextCallee*>(&rhs);
                return (r != NULL)
                    && (r->func == this->func)
                    && (r->ctxt == this->ctxt);
            }

        private:

            /** The function pointer */
            $returnType (*func)($paramCtxtTypeList);

            /** The user data context */
            CT1 ctxt;

        };

        /**
         * Callee implementation for methods
         */
        template<class C>
        class MethodCallee : public AbstractCallee {
        public:

            /**
             * Ctor
             *
             * \@param func The method pointer (must not be NULL)
             * \@param ctxt The user data context
             */
            MethodCallee(C& obj, $returnType (C::*meth)($paramTypeList)) : AbstractCallee(), obj(obj), meth(meth) {
                ASSERT(this->meth != NULL);
            }

            /** Dtor */
            virtual ~MethodCallee(void) {
                this->meth = NULL; // DO NOT DELETE
            }

            /** Call */
            virtual $returnType Call($paramList) {
                ${returnStatement}(this->obj.*this->meth)($paramNameList);
            }

            /**
             * Clones this object
             *
             * \@return A clone
             */
            virtual AbstractCallee * Clone(void) {
                return new MethodCallee<C>(this->obj, this->meth);
            }

            /**
             * Test for equality
             *
             * \@param rhs The right hand side operand
             *
             * \@return True if this and rhs are equal
             */
            virtual bool Equals(const AbstractCallee& rhs) {
                const MethodCallee *r= dynamic_cast<const MethodCallee*>(&rhs);
                return (r != NULL)
                    && (&r->obj == &this->obj)
                    && (r->meth == this->meth);
            }

        private:

            /** The object */
            C& obj;

            /** The method pointer */
            $returnType (C::*meth)($paramTypeList);

        };

        /**
         * Callee implementation for methods with user data context
         */
        template<class C, class CT1>
        class MethodContextCallee : public AbstractCallee {
        public:

            /**
             * Ctor
             *
             * \@param obj The object
             * \@param func The method pointer (must not be NULL)
             * \@param ctxt The user data context
             */
            MethodContextCallee(C& obj, $returnType (C::*meth)($paramCtxtTypeList), CT1 ctxt) : AbstractCallee(), obj(obj), meth(meth), ctxt(ctxt) {
                ASSERT(this->meth != NULL);
            }

            /** Dtor */
            virtual ~MethodContextCallee(void) {
                this->meth = NULL; // DO NOT DELETE
            }

            /** Call */
            virtual $returnType Call($paramList) {
                ${returnStatement}(this->obj.*this->meth)($paramCtxtNameList);
            }

            /**
             * Clones this object
             *
             * \@return A clone
             */
            virtual AbstractCallee * Clone(void) {
                return new MethodContextCallee<C, CT1>(this->obj, this->meth, this->ctxt);
            }

            /**
             * Test for equality
             *
             * \@param rhs The right hand side operand
             *
             * \@return True if this and rhs are equal
             */
            virtual bool Equals(const AbstractCallee& rhs) {
                const MethodContextCallee *r= dynamic_cast<const MethodContextCallee*>(&rhs);
                return (r != NULL)
                    && (&r->obj == &this->obj)
                    && (r->meth == this->meth)
                    && (r->ctxt == this->ctxt);
            }

        private:

            /** The object */
            C& obj;

            /** The method pointer */
            $returnType (C::*meth)($paramCtxtTypeList);

            /** The user data context */
            CT1 ctxt;

        };

        /** The callee target */
        AbstractCallee *callee;

§;

    # class footer
    print $out "    };\n\n";
}

#
# Header
##############################################################################
print $out qq§/*
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

#include "vislib/assert.h"
#include "vislib/IllegalStateException.h"
#include "vislib/memutils.h"


/*
 * This file is generated by 'implementDelegate.pl'
 */


namespace vislib {

§;


#
# full class
##############################################################################
print $out qq§
    /**
     * Class representing a single callback target, which can either be a
     * function (c or static member) or a member-object pair, both with an
     * optional context object (usually a pointer).
     *
     * The first template parameter specifies the return value.
     * The other template parameters define the parameter list of the
     * callback, while void is used to (implicitly) terminate the list.
     * The parameter list may only have a maximum of $paramCnt elements.
     */§;
WriteClass($out, $paramCnt, $paramCnt);

#
# specialization classes for smaller parameter lists
##############################################################################
for (my $pcnt = $paramCnt - 1; $pcnt >= 0; $pcnt--) {
    print $out qq§
    /**
     * Template specialication for delegate functions with $pcnt parameters
     */§;
    WriteClass($out, $paramCnt, $pcnt);
}

#
# specialization class for empty parameter lists
##############################################################################
print $out qq§
    /**
     * Template specialication for delegate functions without parameters and
     * without return value.
     */§;
WriteClass($out, $paramCnt, -1);


#
# footer
##############################################################################
print $out qq§} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_DELEGATE_H_INCLUDED */
§;


close ($out);
exit 0;
