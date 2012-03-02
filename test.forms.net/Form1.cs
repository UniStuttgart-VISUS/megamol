using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using vislib.pinvoke;

namespace test.forms.net {

    /// <summary>
    /// Simple testing form
    /// </summary>
    public partial class Form1 : Form {

        /// <summary>
        /// The OpenGL Box
        /// </summary>
        vislib.gl.OpenGLBox oglbox = null;

        /// <summary>
        /// Ctor
        /// </summary>
        public Form1() {
            InitializeComponent();
            this.Icon = Properties.Resources.vis;
        }

        /// <summary>
        /// Create OpenGL control AFTER the form has been shown
        /// </summary>
        /// <param name="sender">Sender of the Event</param>
        /// <param name="e">Argument of the Evnet</param>
        private void Form1_Shown(object sender, EventArgs e) {

            try { // hack required for unknown reason
                this.oglbox = new vislib.gl.OpenGLBox();
            } catch(Exception) { }
            if (this.oglbox == null) {
                this.oglbox = new vislib.gl.OpenGLBox();
            }

            this.oglbox.Parent = this.splitContainer1.Panel1;
            this.oglbox.Dock = DockStyle.Fill;
            this.oglbox.OpenGLRender += oglbox_OpenGLRender;

            this.checkBox1.Checked = this.oglbox.ContinuousRendering;
            this.checkBox2.Checked = this.oglbox.VSync;
        }

        /// <summary>
        /// Simple rendering
        /// </summary>
        /// <param name="sender">Sender of the Event</param>
        /// <param name="e">Argument of the Evnet</param>
        void oglbox_OpenGLRender(object sender, PaintEventArgs e) {
            opengl32.glClearColor(0.1f, 0.1f, 0.2f, 1.0f);
            opengl32.glClear(opengl32.COLOR_BUFFER_BIT | opengl32.DEPTH_BUFFER_BIT);

            float ang = (float)(DateTime.Now.Second * 60)
                + (float)(DateTime.Now.Millisecond * 60) * 0.001f;

            opengl32.glMatrixMode(opengl32.PROJECTION);
            opengl32.glLoadIdentity();
            opengl32.glMatrixMode(opengl32.MODELVIEW);
            opengl32.glLoadIdentity();
            opengl32.glRotatef(ang, 0.0f, 0.0f, 1.0f);

            opengl32.glBegin(opengl32.TRIANGLES);
            opengl32.glColor3ub(255, 0, 0);
            opengl32.glVertex3f(-0.5f, -0.4f, 0.0f);
            opengl32.glColor3ub(0, 255, 0);
            opengl32.glVertex3f(0.0f, 0.4f, 0.0f);
            opengl32.glColor3ub(0, 0, 255);
            opengl32.glVertex3f(0.5f, -0.4f, 0.0f);
            opengl32.glEnd();
        }

        private void fpsLabelUpdateTimer_Tick(object sender, EventArgs e) {
            this.fpslabel.Text = "FPS: " + this.oglbox.FPS.ToString();
        }

        private void checkBox1_CheckedChanged(object sender, EventArgs e) {
            this.oglbox.ContinuousRendering = this.checkBox1.Checked;
            this.checkBox1.Checked = this.oglbox.ContinuousRendering;
        }

        private void checkBox2_CheckedChanged(object sender, EventArgs e) {
            this.oglbox.VSync = this.checkBox2.Checked;
            this.checkBox2.Checked = this.oglbox.VSync;
        }

    }

}
