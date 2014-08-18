/*
 * AbstractControllerNode.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTCONTROLLERNODE_H_INCLUDED
#define VISLIB_ABSTRACTCONTROLLERNODE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/Socket.h"  // Must be first
#include "vislib/AbstractClusterNode.h"
#include "vislib/CameraParameterObserver.h"
#include "vislib/clustermessages.h"
#include "vislib/ObservableCameraParams.h"
#include "vislib/SmartPtr.h"
#include "vislib/types.h"


namespace vislib {
namespace net {
namespace cluster {


    /**
     * By inheriting from this class, it is possible to make a cluster node the 
     * controlling node that propagates the camera settings to all other nodes
     * in the cluster application.
     *
     * It is recommended to combine this behaviour with a server node that runs
     * as the master of a set of slave machines controlled by the reference
     * camera of the master node.
     *
     * Subclasses must pass messages received in their onMessageReceived() 
     * method to the AbstractControllerNode in order to allow the controller to
     * check for camera messages and process them.
     *
     * This class uses virtual inheritance from AbstractClusterNode to implement
     * the "delegate to sister" pattern.
     */
    class AbstractControllerNode : public virtual AbstractClusterNode,
            public graphics::CameraParameterObserver {

    public:

        /** Dtor. */
        virtual ~AbstractControllerNode(void);

        /**
         * This method is called if the aperture angle changed.
         *
         * @param newValue The new aperture angle.
         */
        virtual void OnApertureAngleChanged(const math::AngleDeg newValue);

        /**
         * This method is called if the stereo eye changed.
         *
         * @param newValue The new stereo eye.
         */
        virtual void OnEyeChanged(
            const graphics::CameraParameters::StereoEye newValue);

        /**
         * This method is called if the far clipping plane changed.
         *
         * @param newValue The new far clipping plane.
         */
        virtual void OnFarClipChanged(const graphics::SceneSpaceType newValue);

        /**
         * This method is called if the focal distance changed.
         *
         * @param newValue The new forcal distance.
         */
        virtual void OnFocalDistanceChanged(
            const graphics::SceneSpaceType newValue);

        //virtual void OnLimitsChanged(const SmartPtr<CameraParameterLimits>& limits) = 0;

        /**
         * This method is called if the look at point changed.
         *
         * @param newValue The new look at point.
         */
        virtual void OnLookAtChanged(
            const graphics::SceneSpacePoint3D& newValue);

        /**
         * This method is called if the near clipping plane changed.
         *
         * @param newValue The new near clipping plane.
         */
        virtual void OnNearClipChanged(const graphics::SceneSpaceType newValue);

        /**
         * This method is called if the camera position changed.
         *
         * @param newValue The new camera position.
         */
        virtual void OnPositionChanged(
            const graphics::SceneSpacePoint3D& newValue);

        /**
         * This method is called if the projection type changed.
         *
         * @param newValue The new projection type.
         */
        virtual void OnProjectionChanged(
            const graphics::CameraParameters::ProjectionType newValue);

        /**
         * This method is called if the stereo disparity changed.
         *
         * @param newValue The new stereo disparity.
         */
        virtual void OnStereoDisparityChanged(
            const graphics::SceneSpaceType newValue);

        /**
         * This method is called if the screen tile changed.
         *
         * @param newValue The new screen tile.
         */
        virtual void OnTileRectChanged(
            const graphics::ImageSpaceRectangle& newValue);

        /**
         * This method is called if the camera up vector changed.
         *
         * @param newValue The new camera up vector.
         */
        virtual void OnUpChanged(
            const graphics::SceneSpaceVector3D& newValue);

        /**
         * This method is called if the virtual screen size changed.
         *
         * @param newValue The new virtual screen size.
         */
        virtual void OnVirtualViewSizeChanged(
            const graphics::ImageSpaceDimension& newValue);

    protected:

        /** 
         * Ctor. 
         *
         * @param params The camera parameters that are to be observed. 
         *               This must be an instance of ObservableCameraParams 
         *               or a derived class.
         */
        AbstractControllerNode(SmartPtr<graphics::CameraParameters> params);

        /**
         * Copy ctor.
         *
         * @param rhs The object to be cloned.
         */
        AbstractControllerNode(const AbstractControllerNode& rhs);

        /**
         * Answer the camera parameters that are observed.
         *
         * @return A pointer to the parameters that are observed.
         */
        inline const graphics::ObservableCameraParams *
                getObservableParameters(void) const {
            return this->parameters.DynamicCast<
                graphics::ObservableCameraParams>();
        }

        /**
         * Answer the camera parameters that are observed.
         *
         * @return A pointer to the parameters that are observed.
         */
        inline graphics::ObservableCameraParams *getObservableParameters(void) {
            return this->parameters.DynamicCast<
                graphics::ObservableCameraParams>();
        }

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
         * If a peer node connects, send initial camera configuration to it.
         *
         * This method and all overriding implementations must not throw
         * an exception!
         *
         * @param peerId The identifier of the new peer node.
         */
        virtual void onPeerConnected(const PeerIdentifier& peerId) throw();

        /**
         * Send all parameters to all peer nodes.
         * This method fails silently.
         *
         * @param peerId If not NULL, send the parameters only to the specified
         *               node. Otherwise, send it to all.
         */
        void sendAllParameters(const PeerIdentifier *peerId = NULL);

        /**
         * Set new camera parameters to observe. 
         * This method unregisters as listener of a possible old set of 
         * parameters and registers as listener of the new one.
         *
         * @param params The new camera parameters to be observed.
         *               This must be an instance of ObservableCameraParams 
         *               or a derived class.
         */
        void setParameters(const SmartPtr<graphics::CameraParameters>&
            params);

        /**
         * Assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        AbstractControllerNode& operator =(const AbstractControllerNode& rhs);

    private:

        /**
         * Send a camera parameter that consists of an integral variable of 
         * type T.
         *
         * @param msgId The message ID.
         * @param value The value.
         */
        template<class T> 
        inline void sendIntegralCamParam(const UINT32 msgId, const T value) {
            BYTE msg[sizeof(MessageHeader) + sizeof(T)];
            MessageHeader *header = reinterpret_cast<MessageHeader *>(msg);
            T *body = reinterpret_cast<T *>(msg + sizeof(MessageHeader));

            InitialiseMessageHeader(*header);
            header->Header.BlockId = msgId;
            header->Header.BlockLength = sizeof(T);
            *body = value;

            this->sendToEachPeer(msg, sizeof(msg));
        }


        /**
         * Send a camera parameter that is an array of D elements of type T.
         *
         * @param msgId The message ID.
         * @param value Pointer to the value.
         */
        template<class T, UINT D>
        inline void sendVectorialCamParam(const UINT32 msgId, const T *value) {
            BYTE msg[sizeof(MessageHeader) + D * sizeof(T)];
            MessageHeader *header = reinterpret_cast<MessageHeader *>(msg);
            T *body = reinterpret_cast<T *>(msg + sizeof(MessageHeader));

            InitialiseMessageHeader(*header);
            header->Header.BlockId = msgId;
            header->Header.BlockLength = D * sizeof(T);
            ::memcpy(body, value, D * sizeof(T));

            this->sendToEachPeer(msg, sizeof(msg));
        }

        /** The camera parameters that are to be observed. */
        SmartPtr<graphics::CameraParameters> parameters;

    };
    
} /* end namespace cluster */
} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTCONTROLLERNODE_H_INCLUDED */
