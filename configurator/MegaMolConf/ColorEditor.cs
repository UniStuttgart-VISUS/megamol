using System.ComponentModel;
using System.Drawing;
using System.Drawing.Design;

namespace MegaMolConf
{

    internal class ColorEditor : UITypeEditor
    {
        public override UITypeEditorEditStyle GetEditStyle(ITypeDescriptorContext context)
        {
            return UITypeEditorEditStyle.None;
        }

        public override bool GetPaintValueSupported(
            ITypeDescriptorContext context)
        {
            return true;
        }

        public override void PaintValue(PaintValueEventArgs e)
        {
            Color color = Color.Transparent;
            try
            {
                color = Data.ParamType.Color.FromString((string)e.Value);
            }
            finally
            {
                using (Brush brush = new SolidBrush(color))
                {
                    e.Graphics.FillRectangle(brush, e.Bounds);
                }
            }
        }
    }

}