/*
 * NodeIdentifier.h
 *
 * Copyright (C) 2006 - 2011 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_NODEIDENTIFIER_H_INCLUDED
#define VISLIB_NODEIDENTIFIER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */


#include "vislib/IPEndPoint.h"
#include "vislib/StackTrace.h"


namespace vislib {
namespace net {
namespace cluster {


    /**
     * This class implements a unique identifier for a cluster node or
     * process.
     */
    class NodeIdentifier {

    public:

        /**
         * Create a new, undeterminate identifier.
         */
        NodeIdentifier(void);

        /**
         * Clone 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        inline NodeIdentifier(const NodeIdentifier& rhs) {
            VLSTACKTRACE("NodeIdentifier::NodeIdentifier", __FILE__, __LINE__);
            *this = rhs;
        }

        /**
         * Dtor.
         */
        ~NodeIdentifier(void);

        /**
         * Answer whether the identifier is undeterminate, i. e. does not 
         * represent a node.
         *
         * @return true if the identifier is undeterminate, false otherwise.
         */
        inline bool IsNull(void) const {
            VLSTACKTRACE("NodeIdentifier::IsNull", __FILE__, __LINE__);
            return (this->id.GetIPAddress().IsAny() 
                && (this->id.GetPort() == 0));
        }

        /**
         * Create a string representation of the ID.
         *
         * @return A string representation of the ID.
         */
        StringA ToStringA(void) const {
            VLSTACKTRACE("NodeIdentifier::ToStringA", __FILE__, __LINE__);
            return this->id.ToStringA();
        }

        /**
         * Create a string representation of the ID.
         *
         * @return A string representation of the ID.
         */
        StringW ToStringW(void) const {
            VLSTACKTRACE("NodeIdentifier::ToStringW", __FILE__, __LINE__);
            return this->id.ToStringW();
        }

        /**
         * Test for equality.
         *
         * @param rhs The right hand side operand.
         *
         * @return true if this object and 'rhs' are equal, false otherwise.
         */
        inline bool operator ==(const NodeIdentifier& rhs) const {
            VLSTACKTRACE("NodeIdentifier::operator ==", __FILE__, __LINE__);
            return (this->id == rhs.id);
        }

        /**
         * Test for inequality.
         *
         * @param rhs The right hand side operand.
         *
         * @return true if this object and 'rhs' are not equal, false otherwise.
         */
        inline bool operator !=(const NodeIdentifier& rhs) const {
            VLSTACKTRACE("NodeIdentifier::operator !=", __FILE__, __LINE__);
            return (this->id != rhs.id);
        }

        /**
         * Assignment operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        NodeIdentifier& operator =(const NodeIdentifier& rhs);

    private:

        /** The IP end point address is the unique identifier. */
        IPEndPoint id;

    };

} /* end namespace cluster */
} /* end namespace net */
} /* end namespace vislib */

#endif /* VISLIB_NODEIDENTIFIER_H_INCLUDED */
