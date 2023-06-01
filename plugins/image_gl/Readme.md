## Imageviewer2
Imageviewer2 displays `jpg` or `png` pictures in MegaMol.

## Modules
The plugin provides a single module called `ImageViewer`.
The module connects via a `CallRender3D`.

List of Parameters:

| Parameter    | Default Value | Description                                                            |
|--------------|---------------|------------------------------------------------------------------------|
| blankMachine | `""`          | Semicolon-separated list of machines that do not load the image        |
| current      | `0`           | Current slideshow image index                                          |
| defaultEye   | `0` (left)    | Where the image goes if the slideshow only has one image per line      |
| leftImg      | `""`          | File name for the left image                                           |
| pasteFiles   | `""`          | Slot to paste both file names at once (semicolon-separated)            |
| pasteShow    | `""`          | Slot to paste filename pairs (semicolon-separated) on individual lines |
| rightImg     | `""`          | File name for the right image                                          |
