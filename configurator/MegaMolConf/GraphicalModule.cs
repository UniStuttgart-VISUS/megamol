using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Linq;
using System.Windows.Forms;
using MegaMolConf.Data;

namespace MegaMolConf {
    public class GraphicalModule : INotifyPropertyChanged {
        static Font moduleClassFont = new Font("Calibri", 8.0f, FontStyle.Italic);
        static Font moduleNameFont = new Font("Calibri", 10.0f);
        static Brush moduleNameBrush = Brushes.Black;
        static Brush moduleRectBrush = new SolidBrush(Color.FromArgb(128, 192, 255));
        static Brush callerBrush = Brushes.Firebrick;
        static Brush calleeBrush = Brushes.Green;
        static Brush compatibleCalleeSlotBrush = Brushes.LightPink;
        static Brush compatibleCallerSlotBrush = Brushes.Chartreuse;
        static Brush selectedSlotBrush = Brushes.Yellow;
        static Pen compatibleSlotBorder = new Pen(selectedSlotBrush, 2.0f);
        static StringFormat moduleClassFormat = new StringFormat() { Alignment = StringAlignment.Center, LineAlignment = StringAlignment.Near, FormatFlags = StringFormatFlags.NoWrap };
        static StringFormat moduleNameFormat = new StringFormat() { Alignment = StringAlignment.Center, LineAlignment = StringAlignment.Far, FormatFlags = StringFormatFlags.NoWrap };
        private static int slotWidth => (int)Math.Round((float)slotWidth_ * DpiFactor);
        private static int slotHeight => (int)Math.Round((float)slotHeight_ * DpiFactor);
        private static int slotSpacing => (int)Math.Round((float)slotSpacing_ * DpiFactor);
        private const int slotWidth_ = 8;
        static int slotHeight_ = 10;
        static int slotSpacing_ = 16;
        static int slotBorder = (slotSpacing - slotHeight) / 2;
        //static float slotScale = 1.8f;
        static int moduleBorder = 10;
        static Pen selectedModulPen1 = makeMyPen1();
        static Pen selectedModulPen2 = makeMyPen2();
        private static float _dpiFactor = 1.0f;

        public static float DpiFactor
        {
            get { return _dpiFactor; }
            set { _dpiFactor = value; }
        }

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

        private Data.CalleeSlot selectedCallee;
        private Data.CallerSlot selectedCaller;

        private string name = string.Empty;
        public String Name {
            get { return name; }
            set {
                name = value;
                boundsCalculated = false;
            }
        }

        public GraphicalModule(Data.Module m, IEnumerable<GraphicalModule> mods) {
            myModule = m;

            int num = 0;
            bool found = true;
            while (found) {
                num++;
                Name = string.Format("{0}{1}", myModule.Name, num);
                found = false;
                foreach (GraphicalModule mm in mods) {
                    if (mm.Name.Equals(Name)) {
                        found = true;
                        break;
                    }
                }
            }

            if (m.ParamSlots != null) {
                foreach (Data.ParamSlot p in m.ParamSlots) {
                    if (p.Type is Data.ParamTypeValueBase) {
                        paramValues[p] = ((Data.ParamTypeValueBase)p.Type).DefaultValueString();
                    }
                    paramCmdLineNess[p] = false;
                }
            }
        }

        public Data.Module Module {
            get {
                return myModule;
            }
        }

        private Dictionary<Data.ParamSlot, string> paramValues = new Dictionary<Data.ParamSlot, string>();
        public Dictionary<Data.ParamSlot, string> ParameterValues {
            get {
                return paramValues;
            }
        }

        private Dictionary<Data.ParamSlot, bool> paramCmdLineNess = new Dictionary<Data.ParamSlot, bool>();
        public Dictionary<Data.ParamSlot, bool> ParameterCmdLineness {
            get {
                return paramCmdLineNess;
            }
        }

        private SizeF textBounds;
        private Size bounds;
        public Size Bounds {
            get {
                return bounds;
            }
        }

        private Point position;
        public Point Position {
            get { return position; }
            set {
                position = value;
                Pos.X = value.X;
                Pos.Y = value.Y;
            }
        }

        public Rectangle DrawBounds {
            get {
                Rectangle r = new Rectangle(position, bounds);
                r.Inflate(slotWidth, 0);
                return r;
            }
        }

        public void Repos() {
            position.X = (int)Pos.X;
            position.Y = (int)Pos.Y;
        }

        public void Draw(Graphics g) {
            string name = myModule.Name + "\n" + Name;

            if (!boundsCalculated) {
                textBounds = g.MeasureString(name, moduleNameFont);
                bounds.Width = (int)(textBounds.Width + moduleBorder * 2);
                int c1 = myModule.CalleeSlots == null ? 0 : myModule.CalleeSlots.Count();
                int c2 = myModule.CallerSlots == null ? 0 : myModule.CallerSlots.Count();
                bounds.Height = Math.Max(Math.Max(c1, c2) * slotSpacing, (int)textBounds.Height) + moduleBorder * 2;
                boundsCalculated = true;
            }
            g.FillRectangle(moduleRectBrush, new Rectangle(Position, Bounds));

            RectangleF textRect = new RectangleF(Position.X + (bounds.Width - textBounds.Width) / 2, 
                Position.Y + (bounds.Height - textBounds.Height) / 2,
                textBounds.Width, textBounds.Height);

            g.TextRenderingHint = System.Drawing.Text.TextRenderingHint.AntiAliasGridFit;
            g.DrawString(myModule.Name, moduleClassFont, moduleNameBrush, textRect, moduleClassFormat);
            g.TextRenderingHint = System.Drawing.Text.TextRenderingHint.SystemDefault;
            g.DrawString(Name, moduleNameFont, moduleNameBrush, textRect, moduleNameFormat);

            if (Form1.isMainView(this)) {
                g.DrawRectangle(Pens.Black, Position.X + moduleBorder / 2, Position.Y + moduleBorder / 2, Bounds.Width - moduleBorder, Bounds.Height - moduleBorder);
            }

            if (Form1.selectedModule == this) {
                g.DrawRectangle(selectedModulPen2, Position.X, Position.Y, Bounds.Width, Bounds.Height);
                g.DrawRectangle(selectedModulPen1, Position.X, Position.Y, Bounds.Width, Bounds.Height);
            }

            if (myModule.CallerSlots != null) {
                for (int x = 0; x < myModule.CallerSlots.Count(); x++) {
                    Data.CallerSlot cr = myModule.CallerSlots[x];
               
                    Point[] ps = new Point[3] {
                        new Point(bounds.Width + Position.X - slotWidth,
                            Position.Y + moduleBorder + x * slotSpacing + slotBorder),
                        new Point(bounds.Width + Position.X + slotWidth,
                            Position.Y + moduleBorder + x * slotSpacing + slotBorder + slotHeight / 2),
                        new Point(bounds.Width + Position.X - slotWidth,
                            Position.Y + moduleBorder + x * slotSpacing + slotBorder + slotHeight)
                    };

                    if (Form1.selectedCaller == cr && Form1.selectedModule != null  && Name == Form1.selectedModule.Name) {
                        g.FillPolygon(selectedSlotBrush, ps);
                        DrawSlotName(g, x, true, cr.Name);
                    } else if (Form1.selectedCallee != null && Form1.selectedCallee.CompatibleCalls.Intersect(cr.CompatibleCalls).Count() > 0) {
                        g.FillPolygon(callerBrush, ps);
                        g.DrawPolygon(compatibleSlotBorder, ps);
                        DrawSlotName(g, x, true, cr.Name);
                    }
                    else {
                        g.FillPolygon(callerBrush, ps);
                    }
                }
            }

            if (myModule.CalleeSlots != null) {
                for (int x = 0; x < myModule.CalleeSlots.Count(); x++) {
                    Data.CalleeSlot ce = myModule.CalleeSlots[x];
                    Point[] ps = new Point[3] {
                        new Point(Position.X - slotWidth, Position.Y + moduleBorder + x * slotSpacing + slotBorder),
                        new Point(Position.X + slotWidth, Position.Y + moduleBorder + x * slotSpacing + slotBorder + slotHeight / 2),
                        new Point(Position.X - slotWidth, Position.Y + moduleBorder + x * slotSpacing + slotBorder + slotHeight)
                    };
                    if (Form1.selectedCallee == ce && Form1.selectedModule != null && Name == Form1.selectedModule.Name) {
                        g.FillPolygon(selectedSlotBrush, ps);
                        DrawSlotName(g, x, false, ce.Name);
                    }
                    else if (Form1.selectedCaller != null && Form1.selectedCaller.CompatibleCalls.Intersect(ce.CompatibleCalls).Count() > 0) {
                        g.FillPolygon(calleeBrush, ps);
                        g.DrawPolygon(compatibleSlotBorder, ps);
                        DrawSlotName(g, x, false, ce.Name);
                    } else {
                        g.FillPolygon(calleeBrush, ps);
                    }
                }
            }
        }

        private void DrawSlotName(Graphics g, int x, bool addBounds, string displayName) {
            if (Form1.drawConnection && Form1.showSlotTips) {
            //if (Form1.drawConnection) {
                int xbase = Position.X + slotWidth;
                if (addBounds) xbase += bounds.Width;
                int ybase = Position.Y + moduleBorder + x * slotSpacing + slotBorder;
                SizeF slotbounds = g.MeasureString(displayName, moduleNameFont);
                ybase -= (int) slotbounds.Height / 4;
                g.FillRectangle(selectedSlotBrush, xbase, ybase, slotbounds.Width, slotbounds.Height);
                g.DrawString(displayName, moduleNameFont, moduleNameBrush, xbase, ybase);
            }
        }

        private Data.Module myModule;
        private bool boundsCalculated;

        internal bool IsHit(Point point) {
            Rectangle r = new Rectangle(Position, Bounds);
            if (r.Contains(point)) {
                return true;
            }
            return false;
        }

        internal bool IsSlotHit(Point point, out Data.CalleeSlot outCe, out Data.CallerSlot outCr, out Point tipLocation) {
            outCe = null;
            outCr = null;
            tipLocation = new Point();
            if (myModule.CallerSlots != null) {
                for (int x = 0; x < myModule.CallerSlots.Count(); x++) {
                    Data.CallerSlot cr = myModule.CallerSlots[x];
                    Rectangle r = new Rectangle(bounds.Width + Position.X - slotWidth, Position.Y + moduleBorder + x * slotSpacing + slotBorder,
                        slotWidth * 2, slotHeight);
                    if (r.Contains(point)) {
                        selectedCaller = cr;
                        outCr = cr;
                        tipLocation = GetCallerTipLocation(x);
                        //Form1.FireSlotSelected(this, cr.Name, cr.CompatibleCalls);
                        return true;
                    }
                }
            }

            if (myModule.CalleeSlots != null) {
                for (int x = 0; x < myModule.CalleeSlots.Count(); x++) {
                    Data.CalleeSlot ce = myModule.CalleeSlots[x];
                    Rectangle r = new Rectangle(Position.X - slotWidth, Position.Y + moduleBorder + x * slotSpacing + slotBorder,
                        slotWidth * 2, slotHeight);
                    if (r.Contains(point)) {
                        selectedCallee = ce;
                        //Form1.FireSlotSelected(this, ce.Name, ce.CompatibleCalls);
                        tipLocation = GetCalleeTipLocation(x);
                        outCe = ce;
                        return true;
                    }
                }
            }
            return false;
        }

        //internal void ClearSlots() {
        //    this.selectedCaller = null;
        //    this.selectedCallee = null;
        //}

        internal Point GetTipLocation(Data.CallerSlot sourceSlot) {
            if (myModule.CallerSlots != null) {
                for (int x = 0; x < myModule.CallerSlots.Count(); x++) {
                    if (myModule.CallerSlots[x] == sourceSlot) {
                        return GetCallerTipLocation(x);
                    }
                }
            }
            return new Point();
        }

        internal Point GetTipLocation(Data.CalleeSlot destSlot) {
            if (myModule.CalleeSlots != null) {
                for (int x = 0; x < myModule.CalleeSlots.Count(); x++) {
                    if (myModule.CalleeSlots[x] == destSlot) {
                        return GetCalleeTipLocation(x);
                    }
                }
            }
            return new Point();
        }

        private Point GetCallerTipLocation(int x) {
            Point ret = new Point();
            ret.X = bounds.Width + Position.X + slotWidth;
            ret.Y = Position.Y + moduleBorder + x * slotSpacing + slotBorder + slotHeight / 2;
            return ret;
        }

        private Point GetCalleeTipLocation(int x) {
            Point ret = new Point();
            ret.X = Position.X - slotWidth;
            ret.Y = Position.Y + moduleBorder + x * slotSpacing + slotBorder + slotHeight / 2;
            return ret;
        }

        #region Layouting temporary variables

        public PointF Force;
        public PointF Speed;
        public PointF Pos;

        #endregion

        public event PropertyChangedEventHandler PropertyChanged;

        public void FirePropertyChanged(string name) {
            if (PropertyChanged != null) {
                PropertyChanged(this, new PropertyChangedEventArgs(name));
            }
        }

    }
}
