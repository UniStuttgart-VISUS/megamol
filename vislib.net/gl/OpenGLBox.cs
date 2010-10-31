using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Data;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using vislib.pinvoke;
using System.Runtime.InteropServices;

namespace vislib.gl {

    /// <summary>
    /// Implementation of an open gl rendering box
    /// </summary>
    public partial class OpenGLBox : UserControl {

        /// <summary>
        /// The device context handle
        /// </summary>
        private IntPtr hDC = IntPtr.Zero;

        /// <summary>
        /// The rendering context handle
        /// </summary>
        private IntPtr hRC = IntPtr.Zero;

        /// <summary>
        /// Gets or sets the viewport aspect ratio
        /// </summary>
        public float AspectRatio {
            get;
            private set;
        }

        /// <summary>
        /// Ctor
        /// </summary>
        public OpenGLBox() {
            InitializeComponent();

            this.hDC = user32.GetDC(this.Handle);
            if (this.hDC == IntPtr.Zero) {
                throw new Exception("Gerätekontext nicht verfügbar");
            }

            gdi32.PIXELFORMATDESCRIPTOR pixelFormat = new gdi32.PIXELFORMATDESCRIPTOR();

            pixelFormat.nSize = 40;
            pixelFormat.nVersion = 1;
            pixelFormat.dwFlags = (gdi32.PFD_DRAW_TO_WINDOW | gdi32.PFD_SUPPORT_OPENGL | gdi32.PFD_DOUBLEBUFFER);
            pixelFormat.iPixelType = gdi32.PFD_TYPE_RGBA;
            pixelFormat.cColorBits = 24;
            pixelFormat.cRedBits = 0;
            pixelFormat.cRedShift = 0;
            pixelFormat.cGreenBits = 0;
            pixelFormat.cGreenShift = 0;
            pixelFormat.cBlueBits = 0;
            pixelFormat.cBlueShift = 0;
            pixelFormat.cAlphaBits = 0;
            pixelFormat.cAlphaShift = 0;
            pixelFormat.cAccumBits = 0;
            pixelFormat.cAccumRedBits = 0;
            pixelFormat.cAccumGreenBits = 0;
            pixelFormat.cAccumBlueBits = 0;
            pixelFormat.cAccumAlphaBits = 0;
            pixelFormat.cDepthBits = 24;
            pixelFormat.cStencilBits = 0;
            pixelFormat.cAuxBuffers = 0;
            pixelFormat.iLayerType = gdi32.PFD_MAIN_PLANE;
            pixelFormat.bReserved = 0;
            pixelFormat.dwLayerMask = 0;
            pixelFormat.dwVisibleMask = 0;
            pixelFormat.dwDamageMask = 0;

            int iPixelformat = gdi32.ChoosePixelFormat(this.hDC, pixelFormat);
            if (iPixelformat == 0) {
                pixelFormat.cDepthBits = 32;
                iPixelformat = gdi32.ChoosePixelFormat(this.hDC, pixelFormat);
            }
            if (iPixelformat == 0) {
                pixelFormat.cDepthBits = 16;
                iPixelformat = gdi32.ChoosePixelFormat(this.hDC, pixelFormat);
            }
            if (iPixelformat == 0) {
                user32.ReleaseDC(this.Handle, this.hDC);
                this.hDC = IntPtr.Zero;
                throw new Exception("Pixelformat nicht verfügbar");
            }

            if (gdi32.SetPixelFormat(this.hDC, iPixelformat, pixelFormat) == 0) {
                user32.ReleaseDC(this.Handle, this.hDC);
                this.hDC = IntPtr.Zero;
                throw new Exception("Pixelformat konnte nicht gesetzt werden");
            }

            this.hRC = opengl32.wglCreateContext(this.hDC);
            if (this.hRC == IntPtr.Zero) {
                user32.ReleaseDC(this.Handle, this.hDC);
                this.hDC = IntPtr.Zero;
                //Marshal.ThrowExceptionForHR(Marshal.GetLastWin32Error());
                throw new Exception("Rendering-Kontext konnte nicht erzeugt werden (Code " + Marshal.GetLastWin32Error().ToString() + ")");
            }

            opengl32.wglMakeCurrent(this.hDC, this.hRC);

            this.OnSizeChanged(null);
            this.OnBackColorChanged(null);

            opengl32.glShadeModel(opengl32.SMOOTH);
            opengl32.glClearDepth(1.0f);
            opengl32.glEnable(opengl32.DEPTH_TEST);
            opengl32.glDepthFunc(opengl32.LEQUAL);
            opengl32.glHint(opengl32.PERSPECTIVE_CORRECTION_HINT, opengl32.NICEST);

        }

        /// <summary>
        /// Raises the Paint event.
        /// </summary>
        /// <param name="e">A PaintEventArgs that contains the event data.</param>
        protected override void OnPaint(PaintEventArgs e) {
            if (this.hRC != IntPtr.Zero) {
                opengl32.wglMakeCurrent(this.hDC, this.hRC);

                if (this.OpenGLRender != null) {
                    this.OpenGLRender(this, e);
                }

                gdi32.SwapBuffers(this.hDC);

            } else {
                base.OnPaint(e);
            }
        }

        /// <summary>
        /// Raises the SizeChanged event.
        /// </summary>
        /// <param name="e">An EventArgs that contains the event data.</param>
        protected override void OnSizeChanged(EventArgs e) {
            base.OnSizeChanged(e);
            if (this.hRC != IntPtr.Zero) {
                opengl32.glViewport(0, 0, this.Width, this.Height);

                float w = (float)this.Width;
                float h = (float)this.Height;
                if (h < 1.0f) h = 1.0f;

                this.AspectRatio = w / h;

                this.Invalidate();
            }
        }

        /// <summary>
        /// This API supports the Team Foundation Server infrastructure and is not intended to be used directly from your code.
        /// </summary>
        /// <param name="e">Paint event parameters</param>
        protected override void OnPaintBackground(PaintEventArgs e) {
            // We do not paint the background (not calling 'base') to avoid flickering.
        }

        /// <summary>
        /// Event called weenever OpenGL drawing can/should occur.
        /// </summary>
        [Description("Called whenever OpenGL drawing can/should occur."), Category("OpenGL")]
        public event PaintEventHandler OpenGLRender;

        /// <summary>
        /// Auto refresh for continous drawing
        /// </summary>
        /// <param name="sender">Sender of the event</param>
        /// <param name="e">Arguments of the event</param>
        private void refreshTimer_Tick(object sender, EventArgs e) {
            this.Invalidate();
        }

        /// <summary>
        /// Dtor
        /// </summary>
        ~OpenGLBox() {
            this.Dispose(false);
        }

        /// <summary> 
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing) {
            if (disposing && (components != null)) {
                components.Dispose();
            }

            if (this.PreDispose != null) {
                this.PreDispose(this, new EventArgs());
            }

            if (this.hRC != IntPtr.Zero) {
                opengl32.wglMakeCurrent(this.hDC, IntPtr.Zero);
                opengl32.wglDeleteContext(this.hRC);
                this.hRC = IntPtr.Zero;
            }

            if (this.hDC != IntPtr.Zero) {
                user32.ReleaseDC(this.Handle, this.hDC);
                this.hDC = IntPtr.Zero;
            }

            base.Dispose(disposing);
        }

        /// <summary>
        /// Set the clear colour depending on the background colour
        /// </summary>
        /// <param name="sender">Sender of the event</param>
        /// <param name="e">Arguments of the event</param>
        private void OpenGLBox_BackColorChanged(object sender, EventArgs e) {
            if (this.hRC == IntPtr.Zero) return;
            float backR = (float)this.BackColor.R / 255.0f;
            float backG = (float)this.BackColor.G / 255.0f;
            float backB = (float)this.BackColor.B / 255.0f;
            opengl32.glClearColor(backR, backG, backB, 1.0f);
        }

        /// <summary>
        /// Event fired before the OpenGL box will be disposed
        /// </summary>
        public event EventHandler PreDispose;

    }

}
