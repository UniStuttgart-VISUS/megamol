namespace MegaMolConf {
    class NoflickerPanel : System.Windows.Forms.Panel {
        public NoflickerPanel() {
            SetStyle(
                System.Windows.Forms.ControlStyles.UserPaint |
                System.Windows.Forms.ControlStyles.AllPaintingInWmPaint |
                System.Windows.Forms.ControlStyles.OptimizedDoubleBuffer |
                System.Windows.Forms.ControlStyles.Selectable |
                System.Windows.Forms.ControlStyles.StandardClick,
                true);
        }

        protected override void OnMouseDown(System.Windows.Forms.MouseEventArgs e) {
            base.OnMouseDown(e);
            if (!Focused) {
                Focus();
            }
        }
    }
}
