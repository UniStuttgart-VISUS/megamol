using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Windows.Forms;

namespace MegaMolConf
{
    partial class TransferFunctionDialog : Form
    {
        class DrawState
        {
             public int start_idx = -1;

            public float start_val = 0.0f;
        }
        private uint res;

        private float[] r_histo;
        private float[] g_histo;
        private float[] b_histo;
        private float[] a_histo;

        private DrawState drawState = null;

        private GraphicalModule gm;

        private Form1 parentForm;

        public static bool IsEditable(GraphicalModule gm)
        {
            return gm.Module.Name == "TransferFunction";
        }

        public TransferFunctionDialog(Form1 parentForm, GraphicalModule gm)
        {
            InitializeComponent();

            this.parentForm = parentForm;

            this.gm = gm;
            if (!HistogramFromGM())
            {
                Histogram((int)nUD_Res.Value, 1024);
                HistogramRamp(ref r_histo);
                HistogramRamp(ref g_histo);
                HistogramRamp(ref b_histo);
                HistogramRamp(ref a_histo);
            }
        }

        private void Histogram(int resolution, int maxResolution)
        {
            res = (uint)Math.Max(1, resolution);
            nUD_Res.Value = res;
            nUD_Res.Maximum = maxResolution;
            System.Diagnostics.Debug.WriteLine(resolution + " " + maxResolution);
            r_histo = new float[res];
            g_histo = new float[res];
            b_histo = new float[res];
            a_histo = new float[res];
            pb_TransferFunc.Image = new Bitmap((int)res, 2);
        }

        private static void HistogramRamp(ref float[] histo)
        {
            float el = 1.0f / histo.Count();
            for (int idx = 0; idx < histo.Count(); ++idx)
            {
                histo[idx] = (float)idx * el;
            }
        }

        private static void HistogramSet(ref float[] histo)
        {
            float el = 1.0f / histo.Count();
            for (int idx = 0; idx < histo.Count(); ++idx)
            {
                histo[idx] = (float)idx * el;
            }
        }

        private bool HistogramFromGM()
        {
            Func<string, string> asString = delegate (string name)
            {
                foreach (var parameter in this.gm.ParameterValues)
                {
                    if (parameter.Key.Name == name)
                    {
                        return parameter.Value;
                    }
                }
                return null;
            };

            Func<string, float> asNumber = delegate (string name)
            {
                string value = asString(name);
                if (value != null)
                {
                    return float.Parse(value, CultureInfo.InvariantCulture);
                }
                return float.NaN;
            };

            Func<string, Color> asColor = delegate (string name)
            {
                string value = asString(name);
                if (value != null)
                {
                    return Data.ParamType.Color.FromString(value);
                }
                return Color.Transparent;
            };

            Func<string, bool> asBool = delegate (string name)
            {
                string value = asString(name);
                if (value != null)
                {
                    return bool.Parse(value);
                }
                return false;
            };

            if (this.gm.Module.Name == "TransferFunction")
            {
                List<Color> colors = new List<Color>();
                //List<float> values = new List<float>();
                for (int i = 0; i < 11; ++i)
                {
                    string suffix = (i + 1).ToString("D2");
                    if (asBool("enable" + suffix))
                    {
                        colors.Add(asColor("colour" + suffix));
                        //values.Add(asNumber("value" + suffix));
                    }

                    Histogram(colors.Count(), 11);
                    for (int j = 0; j < colors.Count(); ++j)
                    {
                        r_histo[j] = (float)colors[j].R / 255.0f;
                        g_histo[j] = (float)colors[j].G / 255.0f;
                        b_histo[j] = (float)colors[j].B / 255.0f;
                        a_histo[j] = (float)colors[j].A / 255.0f;
                    }
                }

                return true;
            }
            else
            {
                Debug.Assert(false, "Unknown type of transfer function (should be unrechable)");
            }

            return false;
        }

        public bool HistogramToGM()
        {
            Func<string, string, bool> setString = delegate (string name, string value)
            {
                Data.ParamSlot p = null;
                foreach (var parameter in this.gm.ParameterValues)
                {
                    if (parameter.Key.Name == name)
                    {
                        p = parameter.Key;
                    }
                }
                if (p != null)
                {
                    // this is the apex of shit
                    this.gm.ParameterValues[p] = value;
                    return true;
                }

                return false;
            };

            if (this.gm.Module.Name == "TransferFunction")
            {
                for (int i = 0; i < Math.Min(11, res); ++i)
                {
                    Color c = Color.FromArgb(
                        (int)(255.0 * a_histo[i]),
                        (int)(255.0 * r_histo[i]),
                        (int)(255.0 * g_histo[i]),
                        (int)(255.0 * b_histo[i]));

                    string suffix = (i + 1).ToString("D2");
                    setString("enable" + suffix, "True");
                    setString("colour" + suffix, Data.ParamType.Color.ToString(c));
                }

                this.parentForm.UpdateParameters(this.gm);
            }
            else
            {
                Debug.Assert(false, "Unknown type of transfer function (should be unrechable)");
            }

            return false;
        }

        private void DrawHistogram(Panel p, Graphics g, Color c, float[] values, int channelIndex, int channelCount)
        {
            Brush brush = new SolidBrush(c);
            float channelHeight = 4;
            float pointWidth = (float)p.Width / res;
            float pointHeight = (channelCount * channelHeight);

            for (int i = 0; i < values.Count(); ++i)
            {
                float x = i * pointWidth;
                float y = (int)((1.0f - values[i]) * p.Height / pointHeight) * pointHeight + channelIndex * channelHeight;
                g.FillRectangle(brush, new RectangleF(x, y, pointWidth, channelHeight));
            }
        }

        private void PanelCanvas_Paint(object sender, PaintEventArgs e)
        {
            var p = sender as Panel;
            var g = e.Graphics;

            DrawHistogram(p, g, Color.Red, r_histo, 0, 4);
            DrawHistogram(p, g, Color.Green, g_histo, 1, 4);
            DrawHistogram(p, g, Color.Blue, b_histo, 2, 4);
            DrawHistogram(p, g, Color.Gray, a_histo, 3, 4);

            UpdateTransferFunction();
        }

        private void UpdateTransferFunction()
        {
            Bitmap image = pb_TransferFunc.Image as Bitmap;
            for (int x = 0; x < res; ++x)
            {
                Color c = Color.FromArgb(
                    (int)(255 * a_histo[x]),
                    (int)(255 * r_histo[x]),
                    (int)(255 * g_histo[x]),
                    (int)(255 * b_histo[x]));
                image.SetPixel(x, 0, c);
                image.SetPixel(x, 1, c);
            }

            pb_TransferFunc.Invalidate();
        }

        private void PanelCanvas_Click(object sender, EventArgs e)
        {
            var p = sender as Panel;
            var me = e as MouseEventArgs;

            float el_width = (float)p.Width / res;
            int start_idx = (int)Math.Floor(me.Location.X / el_width);
            float val = 1.0f - ((float)me.Location.Y / p.Height);

            int idx = start_idx;
            if ((idx < res && idx >= 0) && (val <= 1.0f && val >= 0.0f))
            {
                if (b_R.Checked)
                {
                    r_histo[idx] = val;
                }
                if (b_G.Checked)
                {
                    g_histo[idx] = val;
                }
                if (b_B.Checked)
                {
                    b_histo[idx] = val;
                }
                if (b_A.Checked)
                {
                    a_histo[idx] = val;
                }
            }
            panel_Canvas.Invalidate();
            this.UpdateParameters();
        }

        private void NUDRes_ValChanged(object sender, EventArgs e)
        {
            var t = sender as NumericUpDown;

            if ((int)t.Value == res)
            {
                return;
            }

            Histogram((int)t.Value, (int)nUD_Res.Maximum);
            HistogramRamp(ref r_histo);
            HistogramRamp(ref g_histo);
            HistogramRamp(ref b_histo);
            HistogramRamp(ref a_histo);

            panel_Canvas.Refresh();
        }

        private void PanelCanvas_Resize(object sender, EventArgs e)
        {
            panel_Canvas.Refresh();
        }

        private void PanelCanvas_MouseDown(object sender, MouseEventArgs e)
        {

            var p = sender as Panel;

            float el_width = (float)p.Width / res;
            drawState = new DrawState {
                start_idx = (int)Math.Floor(e.Location.X / el_width),
                start_val = 1.0f - ((float)e.Location.Y / p.Height)
            };
        }

        private void PanelCanvas_MouseUp(object sender, MouseEventArgs e)
        {
            drawState = null;
        }

        private void PanelCanvas_MouseMove(object sender, MouseEventArgs e)
        {
            if (drawState == null)
            {
                return;
            }
            var p = sender as Panel;
            var me = e as MouseEventArgs;

            float el_width = (float)p.Width / res;
            int end_idx = (int)Math.Floor(me.Location.X / el_width);
            float end_val = 1.0f - ((float)me.Location.Y / p.Height);

            int step = 1;

            float dy = (end_val - drawState.start_val) / Math.Abs(end_idx - drawState.start_idx);
            if (drawState.start_idx == end_idx)
            {
                PanelCanvas_Click(sender, e);
                return;
            }

            if (drawState.start_idx > end_idx)
            {
                step = -1;
            }

            int idx = drawState.start_idx;
            float val = drawState.start_val;
            do
            {
                if ((idx < res && idx >= 0) && (val <= 1.0f && val >= 0.0f))
                {
                    if (b_R.Checked)
                    {
                        r_histo[idx] = val;
                    }
                    if (b_G.Checked)
                    {
                        g_histo[idx] = val;
                    }
                    if (b_B.Checked)
                    {
                        b_histo[idx] = val;
                    }
                    if (b_A.Checked)
                    {
                        a_histo[idx] = val;
                    }
                }
                idx += step;
                val += dy;
            } while (idx != end_idx);
            panel_Canvas.Invalidate();
            drawState.start_idx = end_idx;
            drawState.start_val = end_val;
            this.UpdateParameters();
        }

        private void btn_Zero_Click(object sender, EventArgs e)
        {
            if (b_R.Checked)
            {
                Array.Clear(r_histo, 0, r_histo.Length);
            }
            if (b_G.Checked)
            {
                Array.Clear(g_histo, 0, g_histo.Length);
            }
            if (b_B.Checked)
            {
                Array.Clear(b_histo, 0, b_histo.Length);
            }
            if (b_A.Checked)
            {
                Array.Clear(a_histo, 0, b_histo.Length);
            }
            panel_Canvas.Invalidate();
        }

        private void btn_Ramp_Click(object sender, EventArgs e)
        {
            if (b_R.Checked)
            {
                HistogramRamp(ref r_histo);
            }
            if (b_G.Checked)
            {
                HistogramRamp(ref g_histo);
            }
            if (b_B.Checked)
            {
                HistogramRamp(ref b_histo);
            }
            if (b_A.Checked)
            {
                HistogramRamp(ref a_histo);
            }
            panel_Canvas.Invalidate();
        }

        private void color_Clicked(object sender, EventArgs e)
        {
            if (ModifierKeys == Keys.Control)
            {
                if (b_R == sender)
                    b_R.Checked = !b_R.Checked;
                if (b_G == sender)
                    b_G.Checked = !b_G.Checked;
                if (b_B == sender)
                    b_B.Checked = !b_B.Checked;
                if (b_A == sender)
                    b_A.Checked = !b_A.Checked;
            }
            else
            {
                zero_Clicked(sender, e);
                b_R.Checked = (b_R == sender);
                b_G.Checked = (b_G == sender);
                b_B.Checked = (b_B == sender);
                b_A.Checked = (b_A == sender);
            }
        }

        private void zero_Clicked(object sender, EventArgs e)
        {
            b_R.Checked = b_G.Checked = b_B.Checked = b_A.Checked = false;
        }

        private void all_Clicked(object sender, EventArgs e)
        {
            b_R.Checked = b_G.Checked = b_B.Checked = b_A.Checked = true;
        }

        private void UpdateParameters()
        {
            if (!this.throttleTimer.Enabled) {
                this.throttleTimer.Start();
            }
        }

        private void throttleTimer_Tick(object sender, EventArgs e)
        {
            HistogramToGM();
            this.parentForm.UpdateParameters(this.gm);
            this.throttleTimer.Stop();
        }

        private void TransferFunctionDialog_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.KeyCode == Keys.D1)
            {
                this.b_R.Checked = !this.b_R.Checked;
            }
            else if (e.KeyCode == Keys.D2)
            {
                this.b_G.Checked = !this.b_G.Checked;
            }
            else if (e.KeyCode == Keys.D3)
            {
                this.b_B.Checked = !this.b_B.Checked;
            }
            else if (e.KeyCode == Keys.D4)
            {
                this.b_A.Checked = !this.b_A.Checked;
            }
        }

        private void TransferFunctionDialog_Load(object sender, EventArgs e)
        {
            this.panel_Canvas.Focus();
        }
    }
}
