/*
 * FBOTestApp.cpp
 *
 * Copyright (C) 2007 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "FBOTestApp.h"

#include "vislibGlutInclude.h"
#include <iostream>

#include "vislib/glverify.h"


/*
 * FBOTestApp::FBOTestApp
 */
FBOTestApp::FBOTestApp(void) : AbstractGlutApp() {
}


/*
 * FBOTestApp::~FBOTestApp
 */
FBOTestApp::~FBOTestApp(void) {
}


/*
 * FBOTestApp::GLInit
 */
int FBOTestApp::GLInit(void) {
    using vislib::graphics::gl::FramebufferObject;
    
    int retval = 0;

    this->logo.Create();

    ::glEnable(GL_DEPTH_TEST);
    ::glDisable(GL_LIGHTING);

    if (!FramebufferObject::InitialiseExtensions()) {
        retval++;
    } else {
        try {
            FramebufferObject::ColourAttachParams cap[2];
            cap[0].internalFormat = cap[1].internalFormat = GL_RGBA8;
            cap[0].format = cap[1].format = GL_RGBA;
            cap[0].type = cap[1].type = GL_UNSIGNED_BYTE;
            //cap[0].type = GL_FLOAT;

            FramebufferObject::DepthAttachParams dap;
            dap.format = GL_DEPTH_COMPONENT32;
            dap.state = FramebufferObject::ATTACHMENT_TEXTURE;
            FramebufferObject::StencilAttachParams sap;
            sap.state = FramebufferObject::ATTACHMENT_DISABLED;

            ::glDrawBuffer(GL_BACK);
            //if (!this->fbo.Create(512, 512, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, 
            //        FramebufferObject::ATTACHMENT_TEXTURE, GL_DEPTH_COMPONENT32)) {
            if (!this->fbo.Create(512, 512, 2, cap, dap, sap)) {
                std::cout << "Framebuffer object creation failed." << std::endl;
                ASSERT(false);
                retval++;
            }
        } catch (vislib::graphics::gl::OpenGLException e) {
            std::cout << e.GetMsgA() << " @ " << e.GetFile() << ":" << e.GetLine() 
                << std::endl;
            retval++;
        }
    }

    return retval;
}


/*
 * FBOTestApp::GLDeinit
 */
void FBOTestApp::GLDeinit(void) {
    ::glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    this->logo.Release();
}


/*
 * FBOTestApp::Render
 */
void FBOTestApp::Render(void) {
    USES_GL_VERIFY;
   
    try {
        GL_VERIFY_EXPR(this->fbo.Enable(0));
        
        ::glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

        //::glEnable(GL_STENCIL_TEST);
        //::glStencilFunc(GL_ALWAYS, 1, 1);
        //::glStencilOp(GL_KEEP,GL_KEEP, GL_REPLACE);

        ::glMatrixMode(GL_PROJECTION);
        ::glLoadIdentity();

        ::glMatrixMode(GL_MODELVIEW);
        ::glLoadIdentity();

        this->logo.Draw();

        ::glFlush();
        GL_VERIFY_EXPR(this->fbo.Disable());

        ::glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

#if 1
        //GL_VERIFY_EXPR(this->fbo.BindColorTexture(0));
        GL_VERIFY_EXPR(this->fbo.BindDepthTexture());
        ::glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
        ::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        ::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
       
        //::glViewport(0, 0, this->GetWidth(), this->GetHeight());
       
        ::glMatrixMode(GL_PROJECTION);
        ::glLoadIdentity();
        ::gluPerspective(45.0, this->GetAspectRatio(), 0.1, 100.0);

        ::glMatrixMode(GL_MODELVIEW);
        ::glLoadIdentity();
        ::glTranslated(0.0, 0.0, -5.0);
        
        ::glEnable(GL_TEXTURE_2D);
        ::glBegin(GL_QUADS);
        ::glTexCoord2d(0.0, 0.0);
        ::glVertex3d(-1.0, -1.0, 0.0);  // bottom left
        ::glTexCoord2d(1.0, 0.0);
        ::glVertex3d(1.0, -1.0, -3.0);  // bottom right
        ::glTexCoord2d(1.0, 1.0);
        ::glVertex3d(1.0, 1.0, -3.0);   // top right
        ::glTexCoord2d(0.0, 1.0);
        ::glVertex3d(-1.0, 1.0, 0.0);   // top left
        ::glEnd();  

        ::glDisable(GL_TEXTURE_2D);
        ::glColor4f(1.0, 1.0, 1.0, 1.0);
        ::glEnable(GL_POLYGON_OFFSET_LINE);
        ::glPolygonOffset(-1.0f, -1.0f);
        ::glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        ::glBegin(GL_QUADS);
        ::glVertex3d(-1.0, -1.0, 0.0);  // bottom left
        ::glVertex3d(1.0, -1.0, -3.0);  // bottom right
        ::glVertex3d(1.0, 1.0, -3.0);   // top right
        ::glVertex3d(-1.0, 1.0, 0.0);   // top left
        ::glEnd();     
        ::glDisable(GL_POLYGON_OFFSET_LINE);
#else
        this->fbo.DrawColourTexture();
        //this->fbo.DrawDepthTexture(GL_NEAREST, GL_NEAREST);
#endif

        ::glutSwapBuffers();

    } catch (vislib::graphics::gl::OpenGLException e) {
        std::cout << e.GetMsgA() << std::endl;
    }
}
