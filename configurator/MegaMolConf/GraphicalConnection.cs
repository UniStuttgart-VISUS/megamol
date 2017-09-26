using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;

namespace MegaMolConf {
    class GraphicalConnection {
        public GraphicalModule src { get; protected set; }
        public GraphicalModule dest { get; protected set; }
        public Data.CallerSlot srcSlot { get; protected set; }
        public Data.CalleeSlot destSlot { get; protected set; }

        private Data.Call theCall;
        private bool boundsCalculated = false;
        private Size bounds = new Size();
        static StringFormat callNameFormat = new StringFormat() { Alignment = StringAlignment.Center, LineAlignment = StringAlignment.Center, FormatFlags = StringFormatFlags.NoWrap };
        static Font callNameFont = new Font("Calibri", 10.0f);
        static Brush callNameBrush = Brushes.Black;
        static Brush callRectBrush = new SolidBrush(Color.FromArgb(192,192,192));
        static int callBorder = 4;

        static Pen selectedModulPen1 = makeMyPen1();
        static Pen selectedModulPen2 = makeMyPen2();

        static Pen makeMyPen1() {
            Pen p = new Pen(Color.FromArgb(0x60, 0x8f, 0xbf));
            p.DashStyle = System.Drawing.Drawing2D.DashStyle.Custom;
            p.DashPattern = new float[] { 2.0f, 2.0f };
            p.Width = 2.0f;
            p.DashOffset = 2.0f;
            return p;
        }

        static Pen makeMyPen2() {
            Pen p = new Pen(Color.FromArgb(0xbf, 0xdf, 0xff));
            p.DashStyle = System.Drawing.Drawing2D.DashStyle.Custom;
            p.DashPattern = new float[] { 2.0f, 2.0f };
            p.Width = 2.0f;
            p.DashOffset = 0.0f;
            return p;
        }

        public Size Bounds {
            get { return this.bounds; }
        }

        public Data.Call Call {
            get {
                return theCall;
            }
        }

        public GraphicalConnection(GraphicalModule source, GraphicalModule dest,
            Data.CallerSlot sourceSlot, Data.CalleeSlot destSlot, Data.Call call) {
                this.src = source;
                this.dest = dest;
                this.srcSlot = sourceSlot;
                this.destSlot = destSlot;
                this.theCall = call;
        }

        public void Draw(Graphics g) {
            if (!boundsCalculated) {
                SizeF fontSize = g.MeasureString(theCall.Name, callNameFont);
                bounds.Width = (int)(fontSize.Width + callBorder * 2);
                bounds.Height = (int)fontSize.Height + callBorder * 2;
                boundsCalculated = true;
            }
            Point p1 = src.GetTipLocation(srcSlot);
            Point p4 = dest.GetTipLocation(destSlot);
            Point p2 = new Point(p1.X + 32, p1.Y);
            Point p3 = new Point(p4.X - 32, p4.Y);
            g.DrawBezier(Pens.Black, p1, p2, p3, p4);
            Point p5 = new Point((p1.X + p4.X) / 2, (p1.Y + p4.Y) / 2);
            RectangleF rf = new RectangleF(p5.X - bounds.Width / 2, p5.Y - bounds.Height / 2, bounds.Width, bounds.Height);
            g.FillRectangle(callRectBrush, rf);
            g.DrawString(theCall.Name, callNameFont, callNameBrush, rf, callNameFormat);

            if (Form1.selectedConnection == this) {
                g.DrawRectangle(selectedModulPen2, rf.Left, rf.Top, rf.Width, rf.Height);
                g.DrawRectangle(selectedModulPen1, rf.Left, rf.Top, rf.Width, rf.Height);
            }

        }

        internal bool IsHit(Point point) {
            Point p1 = src.GetTipLocation(srcSlot);
            Point p4 = dest.GetTipLocation(destSlot);
            Point p5 = new Point((p1.X + p4.X) / 2, (p1.Y + p4.Y) / 2);
            RectangleF rf = new RectangleF(p5.X - bounds.Width / 2, p5.Y - bounds.Height / 2, bounds.Width, bounds.Height);
            if (rf.Contains(point)) {
                return true;
            }
            return false;
        }

    }
}
