# convert TUKL xyz to MMPLD

use strict;
use warnings qw(FATAL);
use File::stat;

use MMPLD;

my %elemrads = (
    "Ar" => 0.5, # 188 pm vdW
    "Xe" => 0.5
);
my %elemcolors = (
    "Ar" => "0 128 192 255",
    "Xe" => "0 192 224 255"
);

$#ARGV == 1 or die "usage: $0 <inputfile> <outputfile>";

my $inname = $ARGV[0];
my $outname = $ARGV[1];

my @files = sort glob($inname);
my $numfiles = $#files + 1;

print "in: $inname ($numfiles file(s))\tout: $outname\n";

my $m = MMPLD->new({filename=>$outname, numframes=>$numfiles});

my $frametime = 0.0;
my $fileidx = 0;

foreach my $file (@files) {
    $fileidx++;
    print "file $fileidx of $numfiles\n";
    open my $infile, '<', $file or die "could not open input file $!";

    my $numparts = <$infile>;
    my $comment = <$infile>;
    my $foundparts = 0;
    my %parts;

    while(<$infile>) {
        # Ar    86.77    45.6743    115.934
        /^(\w+)\s+(\S+)\s+(\S+)\s+(\S+)$/;
        my $type = $1;
        if (defined $parts{$type}) {
            #print "adding to $type\n";
            push @{$parts{$type}}, $2;
            push @{$parts{$type}}, $3;
            push @{$parts{$type}}, $4;
        } else {
            print "new type: $type\n";
            $parts{$type} = [$2, $3, $4];
        }
        $foundparts++;
    }

    my $numlists = (scalar keys %parts);
    if ($foundparts == $numparts) {
        print "$file: found $foundparts atoms, Ok.\n";
    } else {
        die "$file: found $foundparts (instead of $numparts) atoms, broken.";
    }
    print "$file: $numlists types: " . (join ",", keys %parts) . "\n";
    foreach my $k (keys %parts) {
        if (!defined $elemcolors{$k}) {
            die "define color for $k!";
        }
        if (!defined $elemrads{$k}) {
            die "define radius for $k!"
        }
    }

    $m->StartFrame({frametime=>$frametime, numlists=>$numlists});
    foreach my $k (keys %parts) {
        my $cnt = scalar @{$parts{$k}} / 3;
        print "list: $k, count $cnt\n";
        $m->StartList({
                    vertextype=>$VERTEX_XYZ_FLOAT, colortype=>$COLOR_NONE, globalradius=>$elemrads{$k},
                    globalcolor=>$elemcolors{$k},
                    minintensity=>0.0, maxintensity=>255, particlecount=>$cnt
                    });
        for (my $idx = 0; $idx < $cnt; $idx++) {
            my $x = $parts{$k}[$idx * 3];
            my $y = $parts{$k}[$idx * 3 + 1];
            my $z = $parts{$k}[$idx * 3 + 2];
            $m->AddParticle({
                x=>$x, y=>$y, z=>$z
            });
        }
    }
}
$m->Close();
