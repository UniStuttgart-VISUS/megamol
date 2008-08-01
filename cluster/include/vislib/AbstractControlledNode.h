/*
 * AbstractControlledNode.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTCONTROLLEDNODE_H_INCLUDED
#define VISLIB_ABSTRACTCONTROLLEDNODE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/Socket.h"  // Must be first
#include "vislib/AbstractClusterNode.h"
#include "vislib/assert.h"
#include "vislib/Camera.h"


namespace vislib {
namespace net {
namespace cluster {


    /**
     * This class implements the behaviour of a remote controlled camera. It
     * accepts a reference counted pointer to CameraParameters and updates 
     * those once it receives messages from an AbstractControllerNode.
     *
     * Subclasses must ensure that they call onMessageReceived() in order to
     * give AbstractControlledNode the possibility to filter out camera-related
     * messages.
     */
    class AbstractControlledNode : public virtual AbstractClusterNode {

    public:

        /** Dtor. */
        virtual ~AbstractControlledNode(void);

    protected:

        /**
         * Ctor.
         */
        AbstractControlledNode(void);

        /**
         * Answer the camera parameters.
         *
         * @return A pointer to the parameters.
         */
        inline const SmartPtr<graphics::CameraParameters>& getParameters(
                void) const {
            return this->parameters;
        }

        /**
         * Answer the camera parameters.
         *
         * @return A pointer to the parameters.
         */
        inline SmartPtr<graphics::CameraParameters>& getParameters(void) {
            return this->parameters;
        }

        /**
         * This method is called when data have been received and a valid 
         * message has been found in the packet.
         *
         * @param src     The socket the message has been received from.
         * @param msgId   The message ID.
         * @param body    Pointer to the message body.
         * @param cntBody The number of bytes designated by 'body'.
         *
         * @return true in order to signal that the message has been processed, 
         *         false if the implementation did ignore it.
         */
        virtual bool onMessageReceived(const Socket& src, const UINT msgId,
            const BYTE *body, const SIZE_T cntBody);

        /**
         * Set new camera parameters to update.
         *
         * @param params The new parameters.
         */
        inline void setParameters(
                const SmartPtr<graphics::CameraParameters>& params) {
            ASSERT(!params.IsNull());
            this->parameters = params;
        }

        /**
         * Set new camera parameters to update.
         *
         * @param camera The camera whose parameters should be udpated.
         */
        inline void setParameters(const graphics::Camera& camera) {
            this->parameters = camera.Parameters();
        }

        /**
         * Assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        AbstractControlledNode& operator =(const AbstractControlledNode& rhs);

    private:

        /** The camera parameters that are to be updated via the network. */
        SmartPtr<graphics::CameraParameters> parameters;

    };
    
} /* end namespace cluster */
} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTCONTROLLEDNODE_H_INCLUDED */
