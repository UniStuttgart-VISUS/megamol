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
        private uint res;

        private float[] r_histo;
        private float[] g_histo;
        private float[] b_histo;
        private float[] a_histo;

        //private enum Selected_Channel_Enum {
        //    R,
        //    G,
        //    B,
        //    A
        //}

        //private Selected_Channel_Enum selected_channel;

        private bool drawing = false;

        private int start_idx = -1;

        private float start_val = 0.0f;

        private GraphicalModule gm;

        private Form1 parentForm;

        public TransferFunctionDialog(Form1 parentForm, GraphicalModule gm)
        {
            InitializeComponent();

            this.parentForm = parentForm;

            //selected_channel = Selected_Channel_Enum.A;

            this.gm = gm;
            if (!HistogramFromGM())
            {
                Histogram((int)nUD_Res.Value);
                HistogramRamp(ref r_histo);
                HistogramRamp(ref g_histo);
                HistogramRamp(ref b_histo);
                HistogramRamp(ref a_histo);
            }

            pb_TransferFunc.Image = new Bitmap((int)res, 2);
        }

        private void Histogram(int resolution)
        {
            res = (uint)resolution;
            nUD_Res.Value = res;
            r_histo = new float[res];
            g_histo = new float[res];
            b_histo = new float[res];
            a_histo = new float[res];
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
                    return ColorTranslator.FromHtml(value);
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

            if (this.gm.Module.Name == "LinearTransferFunction")
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

                    Histogram(colors.Count());
                    for (int j = 0; j < colors.Count(); ++j)
                    {
                        r_histo[j] = (float)colors[j].R / 255.0f;
                        g_histo[j] = (float)colors[j].G / 255.0f;
                        b_histo[j] = (float)colors[j].B / 255.0f;
                        a_histo[j] = (float)colors[j].A / 255.0f;
                    }
                }
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

            if (this.gm.Module.Name == "LinearTransferFunction")
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
                    setString("colour" + suffix, ColorTranslator.ToHtml(c));
                    System.Diagnostics.Debug.WriteLine(i + " " + c);
                }

                this.parentForm.UpdateParameters(this.gm);
            }
            else
            {
                Debug.Assert(false, "Unknown type of transfer function (should be unrechable)");
            }

            return false;
        }

        private void DrawChannel(Panel p, Graphics g, Color c, float[] values)
        {
            Brush brush = new SolidBrush(c);

            float el_width = (float)p.Width / res;

            int counter = 0;
            foreach (float val in values)
            {
                g.FillRectangle(brush, new RectangleF(counter * el_width, (1.0f - val) * p.Height, el_width, el_width));
                ++counter;
            }
        }

        private void PanelCanvas_Paint(object sender, PaintEventArgs e)
        {
            var p = sender as Panel;
            var g = e.Graphics;

            DrawChannel(p, g, Color.Red, r_histo);
            DrawChannel(p, g, Color.Green, g_histo);
            DrawChannel(p, g, Color.Blue, b_histo);
            DrawChannel(p, g, Color.Gray, a_histo);

            Bitmap image = pb_TransferFunc.Image as Bitmap;
            for (int x = 0; x < res; ++x)
            {
                Color c = Color.FromArgb((int)(255 * a_histo[x]), (int)(255 * r_histo[x]), (int)(255 * g_histo[x]), (int)(255 * b_histo[x]));
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
        }

        private void NUDRes_ValChanged(object sender, EventArgs e)
        {
            var t = sender as NumericUpDown;

            if ((int)t.Value == res)
            {
                return;
            }

            Histogram((int)t.Value);
            pb_TransferFunc.Image = new Bitmap((int)res, 2);

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
            drawing = true;

            var p = sender as Panel;

            float el_width = (float)p.Width / res;
            start_idx = (int)Math.Floor(e.Location.X / el_width);
            start_val = 1.0f - ((float)e.Location.Y / p.Height);
        }

        private void PanelCanvas_MouseUp(object sender, MouseEventArgs e)
        {
            drawing = false;
            start_idx = -1;
            start_val = 0.0f;
        }

        private void PanelCanvas_MouseMove(object sender, MouseEventArgs e)
        {
            if (!drawing)
            {
                return;
            }
            var p = sender as Panel;
            var me = e as MouseEventArgs;

            float el_width = (float)p.Width / res;
            int end_idx = (int)Math.Floor(me.Location.X / el_width);
            float end_val = 1.0f - ((float)me.Location.Y / p.Height);

            int step = 1;

            float dy = (end_val - start_val) / Math.Abs(end_idx - start_idx);
            if (start_idx == end_idx)
            {
                PanelCanvas_Click(sender, e);
                HistogramToGM();
                return;
            }

            if (start_idx > end_idx)
            {
                step = -1;
            }

            int idx = start_idx;
            float val = start_val;
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
            start_idx = end_idx;
            start_val = end_val;
            HistogramToGM();
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
            if ((ModifierKeys & Keys.Control) == 0)
            {
                zero_Clicked(sender, e);
                b_R.Checked = (b_R == sender);
                b_G.Checked = (b_G == sender);
                b_B.Checked = (b_B == sender);
                b_A.Checked = (b_A == sender);
            }
            else
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
        }

        private void zero_Clicked(object sender, EventArgs e)
        {
            b_R.Checked = b_G.Checked = b_B.Checked = b_A.Checked = false;
        }

        private void all_Clicked(object sender, EventArgs e)
        {
            b_R.Checked = b_G.Checked = b_B.Checked = b_A.Checked = true;
        }
    }
}
