# convert cpe dumps to MMPLD
# these files are generated like so:
# cpe stat i:\grottel\FARO\ss4fsppc01_005.cpe > y:\ss4fsppc01_005.txt
# cpe dec i:\grottel\FARO\ss4fsppc01_005.cpe >> y:\ss4fsppc01_005.txt

use strict;
use warnings qw(FATAL);

use MMPLD;

$#ARGV == 2 or die "usage: $0 <inputfile> <outputfile>.chunknum.mmpld <numchunks>";

my $outfile = $ARGV[1];
my $numChunks = $ARGV[2];

my $temp = <>;
$temp = <>;
$temp =~ /\t(\d+)$/;
my $numPoints = $1;
$temp =~ /^.*?\t(.*?)\t/;
my $rad = $1;
$temp = <>;

print "$numPoints Points, Radius $rad\n";

my $numPerChunk = int($numPoints / $numChunks);
my $lastChunkNum = (($numChunks == 1) ? $numPoints : ($numPoints - ($numPerChunk * ($numChunks - 1))));
print "lastChunkNum: $lastChunkNum\n";

my $currPoint = 0;
my $oldperc = 0;
my ($minx, $miny, $minz, $maxx, $maxy, $maxz);

my @files = ();

sub adjustBounds {
    my ($x, $y, $z) = @_;
    
    if ($currPoint == 0) {
        $minx = $maxx = $x;
        $miny = $maxy = $y;
        $minz = $maxz = $z;
    } else {
        if ($x < $minx) {
            $minx = $x;
        }
        if ($y < $miny) {
            $miny = $y;
        }
        if ($z < $minz) {
            $minz = $z;
        }
        if ($maxx < $x) {
            $maxx = $x;
        }
        if ($maxy < $y) {
            $maxy = $y;
        }
        if ($maxz < $z) {
            $maxz = $z;
        }
    }
}

for (my $chunkIdx = 0; $chunkIdx < $numChunks; $chunkIdx++) {
    my $parts = $numPerChunk;
    if ($chunkIdx == $numChunks - 1) {
        $parts = $lastChunkNum;
    }

    my $fileName = "$outfile.$chunkIdx.mmpld";
    print "writing particle $currPoint to " . ($currPoint + $parts - 1) . " into $fileName\n";
    my $m = MMPLD->new({filename=>$fileName, numframes=>1});
    push @files, $m;
    $m->StartFrame({frametime=>1.23, numlists=>1});

    $m->StartList({
                vertextype=>$VERTEX_XYZ_FLOAT, colortype=>$COLOR_RGB_FLOAT, globalradius=>$rad,
                minintensity=>0.0, maxintensity=>255, particlecount=>$parts
                });

    for (my $i = 0; $i < $parts; $i++) {
        $_ = <>;
        my @vals = split /\s/;
        if ($#vals != 6 && $#vals != 10) {
            print "skipping line with $#vals values: $_\n";
            next;
        }
    
        #print "found particle: " . (join ",", @vals) . "\n";
        $m->AddParticle({
            x=>$vals[0], y=>$vals[1], z=>$vals[2], r=>($vals[4] / 255.0), g=>($vals[5] / 255.0), b=>($vals[6] / 255.0)
        });
        adjustBounds($vals[0], $vals[1], $vals[2]);
        
        $currPoint++;
        my $newperc = int(($currPoint * 100) / $numPoints);
        if ($newperc != $oldperc) {
            print "$newperc%\n";
            $oldperc = $newperc;
        }
        
    }
}

print "got $currPoint points, should be $numPoints: " . (($currPoint==$numPoints)?'Ok':'Problem') . ".\n";
print "rewriting bounding boxes...\n";

foreach my $f (@files) {
    $f->OverrideBBox($minx,$miny,$minz,$maxx,$maxy,$maxz);
    $f->Close();
}
