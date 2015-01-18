#include "CamGlMatrixTest.h"

#include <iostream>
#include "vislibGlutInclude.h"

#include "vislib/assert.h"
#include "vislib/memutils.h"


/*
 * CamGlMatrixTest::CamGlMatrixTest
 */
CamGlMatrixTest::CamGlMatrixTest(void) 
        : AbstractGlutApp(), isManual(false), mia(NULL) {
    using namespace vislib::graphics;
    this->mia = new MouseInteractionAdapter(this->camera.Parameters());
    this->mia->ConfigureZoom(MouseInteractionAdapter::ZOOM_MOVE,
        MouseInteractionAdapter::BUTTON_RIGHT);
}


/*
 * CamGlMatrixTest::~CamGlMatrixTest
 */
CamGlMatrixTest::~CamGlMatrixTest(void) {
    SAFE_DELETE(this->mia);
}


/*
 * CamGlMatrixTest::GLInit
 */
int CamGlMatrixTest::GLInit(void) {
    using namespace vislib::graphics;

    static float lightPos[] = { 10.0f, 10.0f, 0.0f, 1.0f };
    static float lightCol[] = { 1.0f, 1.0f, 1.0f, 1.0f };


    this->vislogo.Create();

    this->camera.Parameters()->SetClip(0.1f, 15.0f);
    this->camera.Parameters()->SetFocalDistance(2.5f);
    this->camera.Parameters()->SetApertureAngle(50.0f);
    this->camera.Parameters()->SetView(SceneSpacePoint3D(0.0f, -5.0f, 0.0f),
        SceneSpacePoint3D(0.0f, 0.0f, 0.0f), 
        SceneSpaceVector3D(0.0f, 0.0f, 1.0f));

    ::glEnable(GL_DEPTH_TEST);
    ::glEnable(GL_LIGHTING);

    ::glEnable(GL_LIGHT0);
    ::glEnable(GL_COLOR_MATERIAL);
    ::glLightfv(GL_LIGHT0, GL_POSITION, lightPos);
    ::glLightfv(GL_LIGHT0, GL_DIFFUSE, lightCol);

    return 0;
}


/*
 * CamGlMatrixTest::GLDeinit
 */
void CamGlMatrixTest::GLDeinit(void) {
    this->vislogo.Release();

    ::glDisable(GL_DEPTH_TEST);
    ::glDisable(GL_LIGHTING);
    ::glDisable(GL_LIGHT0);
    ::glDisable(GL_COLOR_MATERIAL);
}


/*
 * CamGlMatrixTest::OnResize
 */
void CamGlMatrixTest::OnResize(unsigned int w, unsigned int h) {
    AbstractGlutApp::OnResize(w, h);
    this->camera.Parameters()->SetVirtualViewSize(
        static_cast<vislib::graphics::ImageSpaceType>(w),
        static_cast<vislib::graphics::ImageSpaceType>(h));
}


/*
 * CamGlMatrixTest::Render
 */
void CamGlMatrixTest::Render(void) {
    float vm[16], pm[16];

    ::glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (this->isManual) {
        this->camera.ViewMatrix(vm);
        this->camera.ProjectionMatrix(pm);

        ::glMatrixMode(GL_MODELVIEW);
        ::glLoadMatrixf(vm);

        ::glMatrixMode(GL_PROJECTION);
        ::glLoadMatrixf(pm);

    } else {
        ::glMatrixMode(GL_MODELVIEW);
        ::glLoadIdentity();
        this->camera.glMultViewMatrix();

        ::glMatrixMode(GL_PROJECTION);
        ::glLoadIdentity();
        this->camera.glMultProjectionMatrix();
    }

    this->vislogo.Draw();

    ::glutSwapBuffers();
}


/*
 * CamGlMatrixTest::OnKeyPress
 */
bool CamGlMatrixTest::OnKeyPress(unsigned char key, int x, int y) {
    using namespace vislib::graphics;
    this->mia->SetMousePosition(x, y, true);

    static float lightCol[] = { 1.0f, 1.0f, 1.0f, 1.0f };

    switch (key) {
        case 'm':
            this->isManual = !this->isManual;
            std::cout << "isManual = " << this->isManual << std::endl;
            if (this->isManual) {
                lightCol[1] = lightCol[2] = 0.0f;
            } else {
                lightCol[1] = lightCol[2] = 1.0f;
            }
            ::glLightfv(GL_LIGHT0, GL_DIFFUSE, lightCol);
            return true;

        default:
            return false;
    }
}


/*
 * CamGlMatrixTest::OnMouseEvent
 */
void CamGlMatrixTest::OnMouseEvent(int button, int state, int x, int y) {
    using namespace vislib::graphics;
    this->mia->SetMousePosition(x, y, true);

    this->mia->SetMouseButtonState(
        static_cast<MouseInteractionAdapter::Button>(button), 
        (state == GLUT_DOWN));

    int modifiers = glutGetModifiers();
    this->mia->SetModifierState(InputModifiers::MODIFIER_SHIFT,
        (modifiers & GLUT_ACTIVE_SHIFT) == GLUT_ACTIVE_SHIFT);
    this->mia->SetModifierState(InputModifiers::MODIFIER_CTRL,
        (modifiers & GLUT_ACTIVE_CTRL) == GLUT_ACTIVE_CTRL);
    this->mia->SetModifierState(InputModifiers::MODIFIER_ALT,
        (modifiers & GLUT_ACTIVE_ALT) == GLUT_ACTIVE_ALT);
}


/*
 * CamGlMatrixTest::OnMouseMove
 */
void CamGlMatrixTest::OnMouseMove(int x, int y) {
    using namespace vislib::graphics;
    this->mia->SetMousePosition(x, y, true);
    ::glutPostRedisplay();
}


/*
 * CamGlMatrixTest::OnSpecialKey
 */
void CamGlMatrixTest::OnSpecialKey(int key, int x, int y) {
    using namespace vislib::graphics;
    this->mia->SetMousePosition(x, y, true);

    int modifiers = glutGetModifiers();
    this->mia->SetModifierState(InputModifiers::MODIFIER_SHIFT,
        (modifiers & GLUT_ACTIVE_SHIFT) == GLUT_ACTIVE_SHIFT);
    this->mia->SetModifierState(InputModifiers::MODIFIER_CTRL,
        (modifiers & GLUT_ACTIVE_CTRL) == GLUT_ACTIVE_CTRL);
    this->mia->SetModifierState(InputModifiers::MODIFIER_ALT,
        (modifiers & GLUT_ACTIVE_ALT) == GLUT_ACTIVE_ALT);
}
