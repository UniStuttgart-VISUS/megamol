using System;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;

namespace MegaMolConf {
    partial class TransferFunctionDialog : Form {
        public TransferFunctionDialog(GraphicalModule gm) {
            InitializeComponent();

            //selected_channel = Selected_Channel_Enum.A;

            string old_tf = ""; //TODO: replace all this parsing stuff

            if (String.IsNullOrEmpty(old_tf)) {
                res = (uint) nUD_Res.Value;

                r_histo = new float[res];
                g_histo = new float[res];
                b_histo = new float[res];
                a_histo = new float[res];

                InitHistoWithRamp(ref r_histo);
                InitHistoWithRamp(ref g_histo);
                InitHistoWithRamp(ref b_histo);
                InitHistoWithRamp(ref a_histo);
            } else {
                ParseTF(old_tf);
            }

            pb_TransferFunc.Image = new Bitmap((int)res, 2);
        }


        public void ParseTF(string tf) {
            var toks = tf.Split('\"');
            toks[1] = toks[1].Replace(" ", string.Empty);
            toks[1] = toks[1].Replace("\\", string.Empty);
            toks = toks[1].Split(',');
            var vals = Array.ConvertAll(toks, Double.Parse);

            res = (uint)vals.Length / 4;

            r_histo = new float[res];
            g_histo = new float[res];
            b_histo = new float[res];
            a_histo = new float[res];

            for (uint i = 0; i < res; ++i)
            {
                r_histo[i] = (float)vals[i * 4];
                g_histo[i] = (float)vals[i * 4 + 1];
                b_histo[i] = (float)vals[i * 4 + 2];
                a_histo[i] = (float)vals[i * 4 + 3];
            }
        }


        public string GetSerializedTransferFunction() {
            StringBuilder sb = new StringBuilder();
            sb.Append("mmliParseTF('");

            for (uint i = 0; i < this.res; ++i) {
                sb.AppendFormat("{0}, {1}, {2}, {3}", r_histo[i], g_histo[i], b_histo[i], a_histo[i]);
                if (i < (res - 1)) {
                    sb.Append(", ");
                }
            }

            sb.Append("')");

            return sb.ToString();
        }

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

        private void InitHistoWithRamp(ref float[] histo) {
            float el = 1.0f / histo.Count();
            for (int idx = 0; idx < histo.Count(); ++idx) {
                histo[idx] = (float)idx * el;
            }
        }

        private void DrawHistogram(Panel p, Graphics g, Color c, float[] histo) {
            Brush brush = new SolidBrush(c);

            float el_width = (float)p.Width / res;
            float el_height = (float)p.Height;

            int counter = 0;
            foreach (float val in histo) {
                RectangleF rec = new RectangleF(counter * el_width, (1.0f - val) * p.Height, el_width, val * el_height);
                g.FillRectangle(brush, rec);
                ++counter;
            }
        }

        private void DrawScatterplot(Panel p, Graphics g, Color c, float[] values) {
            Brush brush = new SolidBrush(c);

            float el_width = (float)p.Width / res;

            int counter = 0;
            foreach (float val in values) {
                g.FillRectangle(brush, new RectangleF(counter * el_width, (1.0f - val) * p.Height, el_width, el_width));
                ++counter;
            }
        }

        private void PanelCanvas_Paint(object sender, PaintEventArgs e) {
            var p = sender as Panel;
            var g = e.Graphics;

            //if (selected_channel is Selected_Channel_Enum.R) {
            //    DrawScatterplot(p, g, Color.Red, r_histo);
            //} else if (selected_channel is Selected_Channel_Enum.G) {
            //    DrawScatterplot(p, g, Color.Green, g_histo);
            //} else if (selected_channel is Selected_Channel_Enum.B) {
            //    DrawScatterplot(p, g, Color.Blue, b_histo);
            //} else {
            //    DrawScatterplot(p, g, Color.Gray, a_histo);
            //}

            //if (cb_R.Checked) {
            //    DrawScatterplot(p, g, Color.Red, r_histo);
            //}
            //if (cb_G.Checked) {
            //    DrawScatterplot(p, g, Color.Green, g_histo);
            //}
            //if (cb_B.Checked) {
            //    DrawScatterplot(p, g, Color.Blue, b_histo);
            //}
            //if (cb_A.Checked) {
            //    DrawScatterplot(p, g, Color.Gray, a_histo);
            //}

            DrawScatterplot(p, g, Color.Red, r_histo);
            DrawScatterplot(p, g, Color.Green, g_histo);
            DrawScatterplot(p, g, Color.Blue, b_histo);
            DrawScatterplot(p, g, Color.Gray, a_histo);

            Bitmap image = pb_TransferFunc.Image as Bitmap;
            for (int x = 0; x < res; ++x) {
                Color c = Color.FromArgb((int)(255 * a_histo[x]), (int)(255 * r_histo[x]), (int)(255 * g_histo[x]), (int)(255 * b_histo[x]));
                image.SetPixel(x, 0, c);
                image.SetPixel(x, 1, c);
            }

            pb_TransferFunc.Invalidate();
        }

        private void PanelCanvas_Click(object sender, EventArgs e) {
            var p = sender as Panel;
            var me = e as MouseEventArgs;

            float el_width = (float)p.Width / res;
            int start_idx = (int)Math.Floor(me.Location.X / el_width);
            float val = 1.0f - ((float)me.Location.Y / p.Height);

            int idx = start_idx;
            if ((idx < res && idx >= 0) && (val <= 1.0f && val >= 0.0f)) {
                if (b_R.Checked) {
                    r_histo[idx] = val;
                }
                if (b_G.Checked) {
                    g_histo[idx] = val;
                }
                if (b_B.Checked) {
                    b_histo[idx] = val;
                }
                if (b_A.Checked) {
                    a_histo[idx] = val;
                }
            }
            panel_Canvas.Invalidate();
        }

        private void NUDRes_ValChanged(object sender, EventArgs e) {
            var t = sender as NumericUpDown;

            res = (uint)t.Value;

            pb_TransferFunc.Image = new Bitmap((int)res, 2);

            r_histo = new float[res];
            g_histo = new float[res];
            b_histo = new float[res];
            a_histo = new float[res];

            InitHistoWithRamp(ref r_histo);
            InitHistoWithRamp(ref g_histo);
            InitHistoWithRamp(ref b_histo);
            InitHistoWithRamp(ref a_histo);

            panel_Canvas.Refresh();
        }

        private void PanelCanvas_Resize(object sender, EventArgs e) {
            panel_Canvas.Refresh();
        }

        private void PanelCanvas_MouseDown(object sender, MouseEventArgs e) {
            drawing = true;

            var p = sender as Panel;

            float el_width = (float)p.Width / res;
            start_idx = (int)Math.Floor(e.Location.X / el_width);
            start_val = 1.0f - ((float)e.Location.Y / p.Height);
        }

        private void PanelCanvas_MouseUp(object sender, MouseEventArgs e) {
            drawing = false;
            start_idx = -1;
            start_val = 0.0f;
        }

        private void PanelCanvas_MouseMove(object sender, MouseEventArgs e) {
            if (drawing) {
                var p = sender as Panel;
                var me = e as MouseEventArgs;

                float el_width = (float)p.Width / res;
                int end_idx = (int)Math.Floor(me.Location.X / el_width);
                float end_val = 1.0f - ((float)me.Location.Y / p.Height);

                int step = 1;

                float dy = (end_val - start_val) / Math.Abs(end_idx - start_idx);
                if (start_idx == end_idx) {
                    PanelCanvas_Click(sender, e);
                    return;
                    //dy = 0.0f;
                }

                if (start_idx > end_idx) {
                    step = -1;
                }


                int idx = start_idx;
                float val = start_val;
                do {
                    if ((idx < res && idx >= 0) && (val <= 1.0f && val >= 0.0f)) {
                        if (b_R.Checked) {
                            r_histo[idx] = val;
                        }
                        if (b_G.Checked) {
                            g_histo[idx] = val;
                        }
                        if (b_B.Checked) {
                            b_histo[idx] = val;
                        }
                        if (b_A.Checked) {
                            a_histo[idx] = val;
                        }
                    }
                    idx += step;
                    val += dy;
                } while (idx != end_idx);
                panel_Canvas.Invalidate();
                start_idx = end_idx;
                start_val = end_val;
            }
        }

        private void btn_Zero_Click(object sender, EventArgs e) {
            if (b_R.Checked) {
                Array.Clear(r_histo, 0, r_histo.Length);
            }
            if (b_G.Checked) {
                Array.Clear(g_histo, 0, g_histo.Length);
            }
            if (b_B.Checked) {
                Array.Clear(b_histo, 0, b_histo.Length);
            }
            if (b_A.Checked) {
                Array.Clear(a_histo, 0, b_histo.Length);
            }
            panel_Canvas.Invalidate();
        }

        private void btn_Ramp_Click(object sender, EventArgs e) {
            if (b_R.Checked) {
                InitHistoWithRamp(ref r_histo);
            }
            if (b_G.Checked) {
                InitHistoWithRamp(ref g_histo);
            }
            if (b_B.Checked) {
                InitHistoWithRamp(ref b_histo);
            }
            if (b_A.Checked) {
                InitHistoWithRamp(ref a_histo);
            }
            panel_Canvas.Invalidate();
        }

        private void color_Clicked(object sender, EventArgs e) {
            if ((ModifierKeys & Keys.Control) == 0) {
                zero_Clicked(sender, e);
                b_R.Checked = (b_R == sender);
                b_G.Checked = (b_G == sender);
                b_B.Checked = (b_B == sender);
                b_A.Checked = (b_A == sender);
            } else {
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

        private void zero_Clicked(object sender, EventArgs e) {
            b_R.Checked = b_G.Checked = b_B.Checked = b_A.Checked = false;
        }

        private void all_Clicked(object sender, EventArgs e) {
            b_R.Checked = b_G.Checked = b_B.Checked = b_A.Checked = true;
        }
    }
}
