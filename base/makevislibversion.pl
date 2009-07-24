push @INC, "rev2ver";
require "rev2ver.inc";

my $path = shift;
my $inFile = shift;
my $outFile = shift;

my $vislib = getRevisionInfo($path);
my $vislib_base = getRevisionInfo($path . '/base');
my $vislib_cluster = getRevisionInfo($path . '/cluster');
my $vislib_clusterd3d = getRevisionInfo($path . '/clusterd3d');
my $vislib_clustergl = getRevisionInfo($path . '/clustergl');
my $vislib_d3d = getRevisionInfo($path . '/d3d');
my $vislib_gl = getRevisionInfo($path . '/gl');
my $vislib_graphics = getRevisionInfo($path . '/graphics');
my $vislib_math = getRevisionInfo($path . '/math');
my $vislib_net = getRevisionInfo($path . '/net');
my $vislib_sys = getRevisionInfo($path . '/sys');

my %hash;

$hash{'$VISLIB_VERSION$'} = $vislib->rev;
$hash{'$VISLIB_BASE_REVISION$'} = $vislib_base->rev;
$hash{'$VISLIB_CLUSTER_REVISION$'} = $vislib_cluster->rev;
$hash{'$VISLIB_CLUSTERD3D_REVISION$'} = $vislib_clusterd3d->rev;
$hash{'$VISLIB_CLUSTERGL_REVISION$'} = $vislib_clustergl->rev;
$hash{'$VISLIB_D3D_REVISION$'} = $vislib_d3d->rev;
$hash{'$VISLIB_GL_REVISION$'} = $vislib_gl->rev;
$hash{'$VISLIB_GRAPHICS_REVISION$'} = $vislib_graphics->rev;
$hash{'$VISLIB_MATH_REVISION$'} = $vislib_math->rev;
$hash{'$VISLIB_NET_REVISION$'} = $vislib_net->rev;
$hash{'$VISLIB_SYS_REVISION$'} = $vislib_sys->rev;

processFile($outFile, $inFile, \%hash);
