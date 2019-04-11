# convert new imd MPIIO binaries to MMPLD
# this only works for very specific files since the observable description is textual and not regulated (it seems)
# you need to look at the header and then decide how to parse it (line 43 onwards)

use strict;
use warnings qw(FATAL);
use File::stat;

use MMPLD;

$#ARGV == 1 or die "usage: $0 <inputfile> <outputfile>";

my $inname = $ARGV[0];
my $outname = $ARGV[1];

print "in: $inname\tout: $outname\n";

open my $infile, '<:raw', $inname or die "could not open input file $!";
my $stat = stat($infile);

my $bytes_read;
my $bytes;
my $preamble = 1024;

$bytes_read = read $infile, $bytes, $preamble;
$bytes_read == $preamble or die "could not read preamble of length $preamble $!";

my ($magic, $disp, $readatoms, $readobservables, $boxxx, $boxxy, $boxxz, $boxyx, $boxyy, $boxyz, $boxzx, $boxzy, $boxzz, $comment) = unpack 'a3 s q s d d d d d d d d d a936', $bytes;
$comment =~ s/\x00+$//;
$disp == $preamble or die "preamble ($disp) does not fit assumption ($preamble)";
print "input file metadata:\n";
print "$magic : $disp, $readatoms atoms, $readobservables observables\n";
print "boxx: $boxxx, $boxxy, $boxxz\n";
print "boxy: $boxyx, $boxyy, $boxyz\n";
print "boxz: $boxzx, $boxzy, $boxzz\n";
print "comment: $comment\n";
#$comment =~ /^X,Y,Z/ or die "observables do not start with X,Y,Z";
my $filesize = 8 * $readobservables * $readatoms + $preamble;
$filesize == $stat->size or die "file does not meet size expectations ($filesize)";

my $m = MMPLD->new({filename=>$outname, numframes=>1});
$m->StartFrame({frametime=>0.0, numlists=>1});
$m->StartList({
            vertextype=>$VERTEX_XYZ_FLOAT, colortype=>$COLOR_INTENSITY_FLOAT, globalradius=>1.0,
            globalcolor=>"255 0 0 255",
            minintensity=>0.0, maxintensity=>255, particlecount=>$readatoms
            });
for (my $x = 0; $x < $readatoms; $x++) {
    $bytes_read = read $infile, $bytes, $readobservables * 8;
    my ($mass, $x, $y, $z, $vx, $vy, $vz) = unpack 'd d d d d d d', $bytes;
    $m->AddParticle({
        x=>$x, y=>$y, z=>$z, i=>$mass
    });
}
$m->Close();