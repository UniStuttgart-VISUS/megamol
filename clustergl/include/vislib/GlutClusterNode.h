/*
 * GlutClusterNode.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_GLUTCLUSTERNODE_H_INCLUDED
#define VISLIB_GLUTCLUSTERNODE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislibGlutInclude.h"

#include "vislib/AbstractClusterNode.h"
#include "vislib/UnsupportedOperationException.h"
#include "vislib/Trace.h"


namespace vislib {
namespace net {
namespace cluster {


    /**
     * This class defines the interface of a GLUT-powered OpenGL cluster node.
     * It handles OpenGL initialisation and provides virtual method for the
     * drawing and event callbacks.
     *
     * The template must be instantiated with the inheriting class as template
     * parameter. This template parameters is used to implement the singleton
     * pattern here. The inheriting class must make this class a friend of 
     * itself or provide a public default ctor in order for this to work.
     *
     * It is recommended to use the specialised GlutServerNode or GlutClientNode
     * as these resolve ambiguities in the inheritance tree and are therefore
     * easier to use.
     *
     * The GlutServerNode also inherits from AbstractControllerNode and 
     * therefore allows for easily replicating the camera parameters to other 
     * cluster nodes.
     */
    template<class T> class GlutClusterNode {

    public:

        /**
         * Answer the only instance of T.
         *
         * @return The only instance of T.
         */
        static T& GetInstance(void);

        /** Dtor. */
        virtual ~GlutClusterNode(void);

        /**
         * Initialise the node.
         *
         * Implementing subclasses should build an initialisation chain by 
         * calling their parent class implementation first.
         *
         * @param inOutCmdLine The command line passed to the containing
         *                     application. The method might alter the command
         *                     line and remove consumed options.
         *
         * @throws
         */
        virtual void Initialise(sys::CmdLineProviderA& inOutCmdLine);

        /**
         * Initialise the node.
         *
         * Implementing subclasses should build an initialisation chain by 
         * calling their parent class implementation first.
         *
         * @param inOutCmdLine The command line passed to the containing
         *                     application. The method might alter the command
         *                     line and remove consumed options.
         *
         * @throws
         */
        virtual void Initialise(sys::CmdLineProviderW& inOutCmdLine);

        virtual DWORD Run(void);

    protected:

        /** Ctor. */
        GlutClusterNode(void);

        /**
         * This method is called when the next frame must be drawn.
         */
        virtual void onFrameRender(void);

        /**
         * This method is called when the message loop is idling.
         */
        virtual void onIdle(void);

        /**
         * This method is called after the OpenGL context and the window have
         * been created, but before the message loop is entered.
         */
        virtual void onInitialise(void);

        /**
         * The method is called when a mouse button is pressed or released.
         *
         * @param button The button that has been pressed or released.
         * @param state  The new button state.
         * @param x      The mouse x-coordinates where the event occurred.
         * @param y      The mouse y-coordinates where the event occurred.
         */
        virtual void onMouseButton(const int button, const int state,
            const int x, const int y);
    
        /**
         * This method is called when the mouse is moved.
         *
         * @param x The new x-coordinates of the mouse cursor.
         * @param y The new y-coordiantes of the mouse cursor.
         */
        virtual void onMouseMove(const int x, const int y);

        /**
         * This method is called when a character message was received.
         *
         * @param key The key that has been pressed.
         * @param x   The cursor x-coordinates at the time when the key was 
         *            pressed.
         * @param y   The cursor y-coordinates at the time when the key was
         *            pressed.
         */
        virtual void onKeyDown(const unsigned char key, const int x, 
            const int y);

        /**
         * This method is called before the GLUT window is created.
         */
        virtual void onPreCreateWindow(void);

        /**
         * This method is called when the window was resized.
         *
         * @param width  The new width of the window.
         * @param height The new height of the window.
         */
        virtual void onResize(const int width, const int height);

        /**
         * This method is called when a special key was pressed.
         *
         * @param key The virtual key code of the special key that was pressed.
         * @param x   The cursor x-coordinates at the time when the key was 
         *            pressed.
         * @param y   The cursor y-coordinates at the time when the key was
         *            pressed.
         */
        virtual void onSpecialKeyDown(const int key, const int x, const int y);

        /** 
         * Remember whether the windows was opened and the GL context is 
         * available. 
         */
        bool isWindowReady;

    private:

        /**
         * GLUT callback for draw events.
         * This function redirects the call to the instance callback of the 
         * singleton.
         */
        static inline void frameRender(void) {
            GetInstance().onFrameRender();
        }

        /**
         * GLUT callback for idle events.
         * This function redirects the call to the instance callback of the 
         * singleton.
         */
        static inline void idle(void) {
            GetInstance().onIdle();
        }

        /**
         * GLUT callback for mouse click events.
         * This function redirects the call to the instance callback of the 
         * singleton.
         *
         * @param button The button that has been pressed or released.
         * @param state  The new button state.
         * @param x      The mouse x-coordinates where the event occurred.
         * @param y      The mouse y-coordinates where the event occurred.
         */
        static inline void mouseButton(const int button, const int state,
                const int x, const int y) {
            GetInstance().onMouseButton(button, state, x, y);
        }
    
        /**
         * GLUT callback for mouse motion events.
         * This function redirects the call to the instance callback of the 
         * singleton.
         *
         * @param x The new x-coordinates of the mouse cursor.
         * @param y The new y-coordiantes of the mouse cursor.
         */
        static inline void mouseMove(const int x, const int y) {
            GetInstance().onMouseMove(x, y);
        }

        /**
         * GLUT callback for key down events.
         * This function redirects the call to the instance callback of the 
         * singleton.
         *
         * @param key The key that has been pressed.
         * @param x   The cursor x-coordinates at the time when the key was 
         *            pressed.
         * @param y   The cursor y-coordinates at the time when the key was
         *            pressed.
         */
        static inline void keyDown(const unsigned char key, const int x, 
                const int y) {
            GetInstance().onKeyDown(key, x, y);
        }

        /**
         * GLUT callback for window resize events.
         * This function redirects the call to the instance callback of the 
         * singleton.
         *
         * @param width  The new width of the window.
         * @param height The new height of the window.
         */
        static inline void resize(const int width, const int height) {
            GetInstance().onResize(width, height);
        }

        /**
         * GLUT callback for special key down events.
         * This function redirects the call to the instance callback of the 
         * singleton.
         *
         * @param key The virtual key code of the special key that was pressed.
         * @param x   The cursor x-coordinates at the time when the key was 
         *            pressed.
         * @param y   The cursor y-coordinates at the time when the key was
         *            pressed.
         */
        static inline void specialKeyDown(const int key, const int x, 
                const int y) {
            GetInstance().onSpecialKeyDown(key, x, y);
        }

        /** The only instance of the inheriting class. */
        static T *instance;
    };


    /*
     * vislib::net::cluster::GlutClusterNode<T>::GetInstance
     */
    template<class T> T& GlutClusterNode<T>::GetInstance(void) {
        if (GlutClusterNode::instance == NULL) {
            GlutClusterNode::instance = new T();
        }

        return *GlutClusterNode::instance;
    }


    /*
     * vislib::net::cluster::GlutClusterNode<T>::~GlutClusterNode
     */
    template<class T> GlutClusterNode<T>::~GlutClusterNode(void) {
    }


    /*
     * vislib::net::cluster::GlutClusterNode<T>::Initialise
     */
    template<class T> 
    void GlutClusterNode<T>::Initialise(sys::CmdLineProviderA& inOutCmdLine) {
        int argc = inOutCmdLine.ArgC();
        ::glutInit(&argc, inOutCmdLine.ArgV());
    }


    /*
     * vislib::net::cluster::GlutClusterNode<T>::Initialise
     */
    template<class T> 
    void GlutClusterNode<T>::Initialise(sys::CmdLineProviderW& inOutCmdLine) {
        throw UnsupportedOperationException("glutInit is not supported for "
            "wchar_t.", __FILE__, __LINE__);
    }


    /*
     * vislib::net::cluster::GlutClusterNode<T>::Run
     */
    template<class T> DWORD GlutClusterNode<T>::Run(void) {
        // TODO: Additional initialisation callback (format, title, ...) req.
        this->onPreCreateWindow();
        ::glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
        ::glutCreateWindow("");

        this->onInitialise();

        ::glutDisplayFunc(frameRender);
        ::glutIdleFunc(idle);
        ::glutMouseFunc(mouseButton);
        ::glutMotionFunc(mouseMove);
        ::glutPassiveMotionFunc(mouseMove);
        ::glutKeyboardFunc(keyDown);
        ::glutReshapeFunc(resize);
        ::glutSpecialFunc(specialKeyDown);

        this->isWindowReady = true;

        ::glutMainLoop();
        return 0;
    }


    /*
     * vislib::net::cluster::GlutClusterNode<T>::GlutClusterNode
     */
    template<class T> GlutClusterNode<T>::GlutClusterNode(void) 
            : isWindowReady(false) {
    }


    /*
     * vislib::net::cluster::GlutClusterNode<T>::onFrameRender
     */
    template<class T> void GlutClusterNode<T>::onFrameRender(void) {
        ::glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        ::glFlush();
        ::glutSwapBuffers();
    }


    /*
     * vislib::net::cluster::GlutClusterNode<T>::onIdle
     */
    template<class T> void GlutClusterNode<T>::onIdle(void) {
    }


    /*
     * vislib::net::cluster::GlutClusterNode<T>::onInitialise
     */
    template<class T> void GlutClusterNode<T>::onInitialise(void) {
        ::glClearColor(0.0f, 0.0f, 1.0f, 1.0f);
    }


    /*
     * vislib::net::cluster::GlutClusterNode<T>::onMouseButton
     */
    template<class T> void GlutClusterNode<T>::onMouseButton(const int button, 
            const int state, const int x, const int y) {
    }


    /*
     * vislib::net::cluster::GlutClusterNode<T>::onMouseMove
     */
    template<class T> 
    void GlutClusterNode<T>::onMouseMove(const int x, const int y) {
    }


    /*
     * vislib::net::cluster::GlutClusterNode<T>::onKeyDown
     */
    template<class T> 
    void GlutClusterNode<T>::onKeyDown(const unsigned char key, const int x,
            const int y) {
    }


    /*
     * vislib::net::cluster::GlutClusterNode<T>::onResize
     */
    template<class T> 
    void GlutClusterNode<T>::onResize(const int width, const int height) {
        ::glViewport(0, 0, width, height);
    }


    /*
     * vislib::net::cluster::GlutClusterNode<T>::onResize
     */
    template<class T> 
    void GlutClusterNode<T>::onPreCreateWindow(void) {
    }


    /*
     * vislib::net::cluster::GlutClusterNode<T>::onSpecialKeyDown
     */
    template<class T> void GlutClusterNode<T>::onSpecialKeyDown(const int key,
            const int x, const int y) {
    }


    /*
     * vislib::net::cluster::GlutClusterNode<T>::instance
     */
    template<class T> T *GlutClusterNode<T>::instance = NULL;
    
} /* end namespace cluster */
} /* end namespace net */
} /* end namespace vislib */


#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_GLUTCLUSTERNODE_H_INCLUDED */
