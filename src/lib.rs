use std::fs::File;
use std::fs;
use std::os::unix::fs::FileExt;
use rand::{Rng, thread_rng};

use rand::prelude::*;
use rand::distributions::WeightedIndex;
use indicatif::ProgressBar;



//Rayon is Rust equivalent of openMP
use rayon::prelude::ParallelString;
//use rayon::prelude::ParallelSliceMut;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use rayon;

// Library for auto-magically generating python bindings.
use pyo3::prelude::*;



// NOTE: The arrays below are not used at run time ever, and should be optimized out.
//
// These arrays are used to at compile time to binary lookuptables from 2 bit encodings of DNA Bases to their representitive index i.e. the index shared between a kmer and it's
// reverse compliment. They are here, rather than generated in a systemic  manner because I had a pretrained model that worked well before implementing k-mer counting in Rust,
// and I needed to preserve the  correct indexing scheme from python to continue using it. (e.g. "AAAAA" (or 0b0000000000) and it's reverse comp ('TTTT' or 0b1111111111) maps to index 0). 
//
// This was the easiest way to do that. In essense it works like this:
//
// To count kmers, I take K-mer substring  - > find 2*k bit encoding of substring ->  kmer_dist_vector[lookup_index_table[two_bit_encoding]] += 1;
//
// where lookup_index_table is generated at compile time according to these *MER_RAY const arrays, and kmer_dist_vector is generalized TNF like vector
//
const FIVEMER_RAY: [(&'static str, usize);1024] =[("AAAAA", 0),("AAAAT", 3),("AAAAC", 1),("AAAAG", 2),("AAATA", 12),("AAATT", 15),("AAATC", 13),("AAATG", 14),("AAACA", 4),("AAACT", 7),("AAACC", 5),("AAACG", 6),("AAAGA", 8),("AAAGT", 11),("AAAGC", 9),("AAAGG", 10),("AATAA", 47),("AATAT", 50),("AATAC", 48),("AATAG", 49),("AATTA", 59),("AATTT", 15),("AATTC", 60),("AATTG", 61),("AATCA", 51),("AATCT", 54),("AATCC", 52),("AATCG", 53),("AATGA", 55),("AATGT", 58),("AATGC", 56),("AATGG", 57),("AACAA", 16),("AACAT", 19),("AACAC", 17),("AACAG", 18),("AACTA", 28),("AACTT", 31),("AACTC", 29),("AACTG", 30),("AACCA", 20),("AACCT", 23),("AACCC", 21),("AACCG", 22),("AACGA", 24),("AACGT", 27),("AACGC", 25),("AACGG", 26),("AAGAA", 32),("AAGAT", 35),("AAGAC", 33),("AAGAG", 34),("AAGTA", 44),("AAGTT", 31),("AAGTC", 45),("AAGTG", 46),("AAGCA", 36),("AAGCT", 39),("AAGCC", 37),("AAGCG", 38),("AAGGA", 40),("AAGGT", 43),("AAGGC", 41),("AAGGG", 42),("ATAAA", 174),("ATAAT", 177),("ATAAC", 175),("ATAAG", 176),("ATATA", 184),("ATATT", 50),("ATATC", 185),("ATATG", 186),("ATACA", 178),("ATACT", 164),("ATACC", 179),("ATACG", 180),("ATAGA", 181),("ATAGT", 109),("ATAGC", 182),("ATAGG", 183),("ATTAA", 212),("ATTAT", 177),("ATTAC", 213),("ATTAG", 214),("ATTTA", 221),("ATTTT", 3),("ATTTC", 222),("ATTTG", 223),("ATTCA", 215),("ATTCT", 123),("ATTCC", 216),("ATTCG", 217),("ATTGA", 218),("ATTGT", 65),("ATTGC", 219),("ATTGG", 220),("ATCAA", 187),("ATCAT", 190),("ATCAC", 188),("ATCAG", 189),("ATCTA", 197),("ATCTT", 35),("ATCTC", 198),("ATCTG", 199),("ATCCA", 191),("ATCCT", 151),("ATCCC", 192),("ATCCG", 193),("ATCGA", 194),("ATCGT", 95),("ATCGC", 195),("ATCGG", 196),("ATGAA", 200),("ATGAT", 190),("ATGAC", 201),("ATGAG", 202),("ATGTA", 209),("ATGTT", 19),("ATGTC", 210),("ATGTG", 211),("ATGCA", 203),("ATGCT", 137),("ATGCC", 204),("ATGCG", 205),("ATGGA", 206),("ATGGT", 80),("ATGGC", 207),("ATGGG", 208),("ACAAA", 62),("ACAAT", 65),("ACAAC", 63),("ACAAG", 64),("ACATA", 74),("ACATT", 58),("ACATC", 75),("ACATG", 76),("ACACA", 66),("ACACT", 69),("ACACC", 67),("ACACG", 68),("ACAGA", 70),("ACAGT", 73),("ACAGC", 71),("ACAGG", 72),("ACTAA", 106),("ACTAT", 109),("ACTAC", 107),("ACTAG", 108),("ACTTA", 117),("ACTTT", 11),("ACTTC", 118),("ACTTG", 119),("ACTCA", 110),("ACTCT", 113),("ACTCC", 111),("ACTCG", 112),("ACTGA", 114),("ACTGT", 73),("ACTGC", 115),("ACTGG", 116),("ACCAA", 77),("ACCAT", 80),("ACCAC", 78),("ACCAG", 79),("ACCTA", 89),("ACCTT", 43),("ACCTC", 90),("ACCTG", 91),("ACCCA", 81),("ACCCT", 84),("ACCCC", 82),("ACCCG", 83),("ACCGA", 85),("ACCGT", 88),("ACCGC", 86),("ACCGG", 87),("ACGAA", 92),("ACGAT", 95),("ACGAC", 93),("ACGAG", 94),("ACGTA", 103),("ACGTT", 27),("ACGTC", 104),("ACGTG", 105),("ACGCA", 96),("ACGCT", 99),("ACGCC", 97),("ACGCG", 98),("ACGGA", 100),("ACGGT", 88),("ACGGC", 101),("ACGGG", 102),("AGAAA", 120),("AGAAT", 123),("AGAAC", 121),("AGAAG", 122),("AGATA", 131),("AGATT", 54),("AGATC", 132),("AGATG", 133),("AGACA", 124),("AGACT", 127),("AGACC", 125),("AGACG", 126),("AGAGA", 128),("AGAGT", 113),("AGAGC", 129),("AGAGG", 130),("AGTAA", 161),("AGTAT", 164),("AGTAC", 162),("AGTAG", 163),("AGTTA", 171),("AGTTT", 7),("AGTTC", 172),("AGTTG", 173),("AGTCA", 165),("AGTCT", 127),("AGTCC", 166),("AGTCG", 167),("AGTGA", 168),("AGTGT", 69),("AGTGC", 169),("AGTGG", 170),("AGCAA", 134),("AGCAT", 137),("AGCAC", 135),("AGCAG", 136),("AGCTA", 145),("AGCTT", 39),("AGCTC", 146),("AGCTG", 147),("AGCCA", 138),("AGCCT", 141),("AGCCC", 139),("AGCCG", 140),("AGCGA", 142),("AGCGT", 99),("AGCGC", 143),("AGCGG", 144),("AGGAA", 148),("AGGAT", 151),("AGGAC", 149),("AGGAG", 150),("AGGTA", 158),("AGGTT", 23),("AGGTC", 159),("AGGTG", 160),("AGGCA", 152),("AGGCT", 141),("AGGCC", 153),("AGGCG", 154),("AGGGA", 155),("AGGGT", 84),("AGGGC", 156),("AGGGG", 157),("TAAAA", 480),("TAAAT", 221),("TAAAC", 479),("TAAAG", 382),("TAATA", 483),("TAATT", 59),("TAATC", 413),("TAATG", 268),("TAACA", 481),("TAACT", 171),("TAACC", 461),("TAACG", 348),("TAAGA", 482),("TAAGT", 117),("TAAGC", 439),("TAAGG", 310),("TATAA", 491),("TATAT", 184),("TATAC", 466),("TATAG", 357),("TATTA", 483),("TATTT", 12),("TATTC", 390),("TATTG", 233),("TATCA", 492),("TATCT", 131),("TATCC", 445),("TATCG", 320),("TATGA", 493),("TATGT", 74),("TATGC", 420),("TATGG", 279),("TACAA", 484),("TACAT", 209),("TACAC", 475),("TACAG", 374),("TACTA", 487),("TACTT", 44),("TACTC", 406),("TACTG", 257),("TACCA", 485),("TACCT", 158),("TACCC", 456),("TACCG", 339),("TACGA", 486),("TACGT", 103),("TACGC", 433),("TACGG", 300),("TAGAA", 488),("TAGAT", 197),("TAGAC", 471),("TAGAG", 366),("TAGTA", 487),("TAGTT", 28),("TAGTC", 398),("TAGTG", 245),("TAGCA", 489),("TAGCT", 145),("TAGCC", 451),("TAGCG", 330),("TAGGA", 490),("TAGGT", 89),("TAGGC", 427),("TAGGG", 290),("TTAAA", 510),("TTAAT", 212),("TTAAC", 476),("TTAAG", 376),("TTATA", 491),("TTATT", 47),("TTATC", 407),("TTATG", 259),("TTACA", 509),("TTACT", 161),("TTACC", 457),("TTACG", 341),("TTAGA", 502),("TTAGT", 106),("TTAGC", 434),("TTAGG", 302),("TTTAA", 510),("TTTAT", 174),("TTTAC", 462),("TTTAG", 350),("TTTTA", 480),("TTTTT", 0),("TTTTC", 384),("TTTTG", 224),("TTTCA", 504),("TTTCT", 120),("TTTCC", 440),("TTTCG", 312),("TTTGA", 494),("TTTGT", 62),("TTTGC", 414),("TTTGG", 270),("TTCAA", 511),("TTCAT", 200),("TTCAC", 472),("TTCAG", 368),("TTCTA", 488),("TTCTT", 32),("TTCTC", 400),("TTCTG", 248),("TTCCA", 508),("TTCCT", 148),("TTCCC", 452),("TTCCG", 332),("TTCGA", 500),("TTCGT", 92),("TTCGC", 428),("TTCGG", 292),("TTGAA", 511),("TTGAT", 187),("TTGAC", 467),("TTGAG", 359),("TTGTA", 484),("TTGTT", 16),("TTGTC", 392),("TTGTG", 236),("TTGCA", 506),("TTGCT", 134),("TTGCC", 446),("TTGCG", 322),("TTGGA", 497),("TTGGT", 77),("TTGGC", 421),("TTGGG", 281),("TCAAA", 494),("TCAAT", 218),("TCAAC", 478),("TCAAG", 380),("TCATA", 493),("TCATT", 55),("TCATC", 411),("TCATG", 265),("TCACA", 495),("TCACT", 168),("TCACC", 460),("TCACG", 346),("TCAGA", 496),("TCAGT", 114),("TCAGC", 438),("TCAGG", 308),("TCTAA", 502),("TCTAT", 181),("TCTAC", 465),("TCTAG", 355),("TCTTA", 482),("TCTTT", 8),("TCTTC", 388),("TCTTG", 230),("TCTCA", 503),("TCTCT", 128),("TCTCC", 444),("TCTCG", 318),("TCTGA", 496),("TCTGT", 70),("TCTGC", 418),("TCTGG", 276),("TCCAA", 497),("TCCAT", 206),("TCCAC", 474),("TCCAG", 372),("TCCTA", 490),("TCCTT", 40),("TCCTC", 404),("TCCTG", 254),("TCCCA", 498),("TCCCT", 155),("TCCCC", 455),("TCCCG", 337),("TCCGA", 499),("TCCGT", 100),("TCCGC", 432),("TCCGG", 298),("TCGAA", 500),("TCGAT", 194),("TCGAC", 470),("TCGAG", 364),("TCGTA", 486),("TCGTT", 24),("TCGTC", 396),("TCGTG", 242),("TCGCA", 501),("TCGCT", 142),("TCGCC", 450),("TCGCG", 328),("TCGGA", 499),("TCGGT", 85),("TCGGC", 425),("TCGGG", 287),("TGAAA", 504),("TGAAT", 215),("TGAAC", 477),("TGAAG", 378),("TGATA", 492),("TGATT", 51),("TGATC", 409),("TGATG", 262),("TGACA", 505),("TGACT", 165),("TGACC", 459),("TGACG", 344),("TGAGA", 503),("TGAGT", 110),("TGAGC", 436),("TGAGG", 305),("TGTAA", 509),("TGTAT", 178),("TGTAC", 464),("TGTAG", 353),("TGTTA", 481),("TGTTT", 4),("TGTTC", 386),("TGTTG", 227),("TGTCA", 505),("TGTCT", 124),("TGTCC", 442),("TGTCG", 315),("TGTGA", 495),("TGTGT", 66),("TGTGC", 416),("TGTGG", 273),("TGCAA", 506),("TGCAT", 203),("TGCAC", 473),("TGCAG", 370),("TGCTA", 489),("TGCTT", 36),("TGCTC", 402),("TGCTG", 251),("TGCCA", 507),("TGCCT", 152),("TGCCC", 454),("TGCCG", 335),("TGCGA", 501),("TGCGT", 96),("TGCGC", 430),("TGCGG", 295),("TGGAA", 508),("TGGAT", 191),("TGGAC", 469),("TGGAG", 362),("TGGTA", 485),("TGGTT", 20),("TGGTC", 394),("TGGTG", 239),("TGGCA", 507),("TGGCT", 138),("TGGCC", 448),("TGGCG", 325),("TGGGA", 498),("TGGGT", 81),("TGGGC", 423),("TGGGG", 284),("CAAAA", 224),("CAAAT", 223),("CAAAC", 225),("CAAAG", 226),("CAATA", 233),("CAATT", 61),("CAATC", 234),("CAATG", 235),("CAACA", 227),("CAACT", 173),("CAACC", 228),("CAACG", 229),("CAAGA", 230),("CAAGT", 119),("CAAGC", 231),("CAAGG", 232),("CATAA", 259),("CATAT", 186),("CATAC", 260),("CATAG", 261),("CATTA", 268),("CATTT", 14),("CATTC", 269),("CATTG", 235),("CATCA", 262),("CATCT", 133),("CATCC", 263),("CATCG", 264),("CATGA", 265),("CATGT", 76),("CATGC", 266),("CATGG", 267),("CACAA", 236),("CACAT", 211),("CACAC", 237),("CACAG", 238),("CACTA", 245),("CACTT", 46),("CACTC", 246),("CACTG", 247),("CACCA", 239),("CACCT", 160),("CACCC", 240),("CACCG", 241),("CACGA", 242),("CACGT", 105),("CACGC", 243),("CACGG", 244),("CAGAA", 248),("CAGAT", 199),("CAGAC", 249),("CAGAG", 250),("CAGTA", 257),("CAGTT", 30),("CAGTC", 258),("CAGTG", 247),("CAGCA", 251),("CAGCT", 147),("CAGCC", 252),("CAGCG", 253),("CAGGA", 254),("CAGGT", 91),("CAGGC", 255),("CAGGG", 256),("CTAAA", 350),("CTAAT", 214),("CTAAC", 351),("CTAAG", 352),("CTATA", 357),("CTATT", 49),("CTATC", 358),("CTATG", 261),("CTACA", 353),("CTACT", 163),("CTACC", 354),("CTACG", 343),("CTAGA", 355),("CTAGT", 108),("CTAGC", 356),("CTAGG", 304),("CTTAA", 376),("CTTAT", 176),("CTTAC", 377),("CTTAG", 352),("CTTTA", 382),("CTTTT", 2),("CTTTC", 383),("CTTTG", 226),("CTTCA", 378),("CTTCT", 122),("CTTCC", 379),("CTTCG", 314),("CTTGA", 380),("CTTGT", 64),("CTTGC", 381),("CTTGG", 272),("CTCAA", 359),("CTCAT", 202),("CTCAC", 360),("CTCAG", 361),("CTCTA", 366),("CTCTT", 34),("CTCTC", 367),("CTCTG", 250),("CTCCA", 362),("CTCCT", 150),("CTCCC", 363),("CTCCG", 334),("CTCGA", 364),("CTCGT", 94),("CTCGC", 365),("CTCGG", 294),("CTGAA", 368),("CTGAT", 189),("CTGAC", 369),("CTGAG", 361),("CTGTA", 374),("CTGTT", 18),("CTGTC", 375),("CTGTG", 238),("CTGCA", 370),("CTGCT", 136),("CTGCC", 371),("CTGCG", 324),("CTGGA", 372),("CTGGT", 79),("CTGGC", 373),("CTGGG", 283),("CCAAA", 270),("CCAAT", 220),("CCAAC", 271),("CCAAG", 272),("CCATA", 279),("CCATT", 57),("CCATC", 280),("CCATG", 267),("CCACA", 273),("CCACT", 170),("CCACC", 274),("CCACG", 275),("CCAGA", 276),("CCAGT", 116),("CCAGC", 277),("CCAGG", 278),("CCTAA", 302),("CCTAT", 183),("CCTAC", 303),("CCTAG", 304),("CCTTA", 310),("CCTTT", 10),("CCTTC", 311),("CCTTG", 232),("CCTCA", 305),("CCTCT", 130),("CCTCC", 306),("CCTCG", 307),("CCTGA", 308),("CCTGT", 72),("CCTGC", 309),("CCTGG", 278),("CCCAA", 281),("CCCAT", 208),("CCCAC", 282),("CCCAG", 283),("CCCTA", 290),("CCCTT", 42),
("CCCTC", 291),("CCCTG", 256),("CCCCA", 284),("CCCCT", 157),("CCCCC", 285),("CCCCG", 286),("CCCGA", 287),("CCCGT", 102),("CCCGC", 288),("CCCGG", 289),("CCGAA", 292),("CCGAT", 196),
("CCGAC", 293),("CCGAG", 294),("CCGTA", 300),("CCGTT", 26),("CCGTC", 301),("CCGTG", 244),("CCGCA", 295),("CCGCT", 144),("CCGCC", 296),("CCGCG", 297),("CCGGA", 298),("CCGGT", 87),("CCGGC", 299),("CCGGG", 289),
("CGAAA", 312),("CGAAT", 217),("CGAAC", 313),("CGAAG", 314),("CGATA", 320),("CGATT", 53),("CGATC", 321),("CGATG", 264),("CGACA", 315),("CGACT", 167),("CGACC", 316),("CGACG", 317),("CGAGA", 318),("CGAGT", 112),
("CGAGC", 319),("CGAGG", 307),("CGTAA", 341),("CGTAT", 180),("CGTAC", 342),("CGTAG", 343),("CGTTA", 348),("CGTTT", 6),("CGTTC", 349),("CGTTG", 229),("CGTCA", 344),("CGTCT", 126),("CGTCC", 345),("CGTCG", 317),
("CGTGA", 346),("CGTGT", 68),("CGTGC", 347),("CGTGG", 275),("CGCAA", 322),("CGCAT", 205),("CGCAC", 323),("CGCAG", 324),("CGCTA", 330),("CGCTT", 38),("CGCTC", 331),("CGCTG", 253),("CGCCA", 325),("CGCCT", 154),
("CGCCC", 326),("CGCCG", 327),("CGCGA", 328),("CGCGT", 98),("CGCGC", 329),("CGCGG", 297),("CGGAA", 332),("CGGAT", 193),("CGGAC", 333),("CGGAG", 334),("CGGTA", 339),("CGGTT", 22),("CGGTC", 340),("CGGTG", 241),
("CGGCA", 335),("CGGCT", 140),("CGGCC", 336),("CGGCG", 327),("CGGGA", 337),("CGGGT", 83),("CGGGC", 338),("CGGGG", 286),("GAAAA", 384),("GAAAT", 222),("GAAAC", 385),("GAAAG", 383),("GAATA", 390),("GAATT", 60),
("GAATC", 391),("GAATG", 269),("GAACA", 386),("GAACT", 172),("GAACC", 387),("GAACG", 349),("GAAGA", 388),("GAAGT", 118),("GAAGC", 389),("GAAGG", 311),("GATAA", 407),("GATAT", 185),("GATAC", 408),("GATAG", 358),
("GATTA", 413),("GATTT", 13),("GATTC", 391),("GATTG", 234),("GATCA", 409),("GATCT", 132),("GATCC", 410),("GATCG", 321),("GATGA", 411),("GATGT", 75),("GATGC", 412),("GATGG", 280),("GACAA", 392),("GACAT", 210),
("GACAC", 393),("GACAG", 375),("GACTA", 398),("GACTT", 45),("GACTC", 399),("GACTG", 258),("GACCA", 394),("GACCT", 159),("GACCC", 395),("GACCG", 340),("GACGA", 396),("GACGT", 104),("GACGC", 397),("GACGG", 301),
("GAGAA", 400),("GAGAT", 198),("GAGAC", 401),("GAGAG", 367),("GAGTA", 406),("GAGTT", 29),("GAGTC", 399),("GAGTG", 246),("GAGCA", 402),("GAGCT", 146),("GAGCC", 403),("GAGCG", 331),("GAGGA", 404),("GAGGT", 90),
("GAGGC", 405),("GAGGG", 291),("GTAAA", 462),("GTAAT", 213),("GTAAC", 463),("GTAAG", 377),("GTATA", 466),("GTATT", 48),("GTATC", 408),("GTATG", 260),("GTACA", 464),("GTACT", 162),("GTACC", 458),("GTACG", 342),
("GTAGA", 465),("GTAGT", 107),("GTAGC", 435),("GTAGG", 303),("GTTAA", 476),("GTTAT", 175),("GTTAC", 463),("GTTAG", 351),("GTTTA", 479),("GTTTT", 1),("GTTTC", 385),("GTTTG", 225),("GTTCA", 477),("GTTCT", 121),
("GTTCC", 441),("GTTCG", 313),("GTTGA", 478),("GTTGT", 63),("GTTGC", 415),("GTTGG", 271),("GTCAA", 467),("GTCAT", 201),("GTCAC", 468),("GTCAG", 369),("GTCTA", 471),("GTCTT", 33),("GTCTC", 401),("GTCTG", 249),
("GTCCA", 469),("GTCCT", 149),("GTCCC", 453),("GTCCG", 333),("GTCGA", 470),("GTCGT", 93),("GTCGC", 429),("GTCGG", 293),("GTGAA", 472),("GTGAT", 188),("GTGAC", 468),("GTGAG", 360),("GTGTA", 475),("GTGTT", 17),
("GTGTC", 393),("GTGTG", 237),("GTGCA", 473),("GTGCT", 135),("GTGCC", 447),("GTGCG", 323),("GTGGA", 474),("GTGGT", 78),("GTGGC", 422),("GTGGG", 282),("GCAAA", 414),("GCAAT", 219),("GCAAC", 415),("GCAAG", 381),
("GCATA", 420),("GCATT", 56),("GCATC", 412),("GCATG", 266),("GCACA", 416),("GCACT", 169),("GCACC", 417),("GCACG", 347),("GCAGA", 418),("GCAGT", 115),("GCAGC", 419),("GCAGG", 309),("GCTAA", 434),("GCTAT", 182),
("GCTAC", 435),("GCTAG", 356),("GCTTA", 439),("GCTTT", 9),("GCTTC", 389),("GCTTG", 231),("GCTCA", 436),("GCTCT", 129),("GCTCC", 437),("GCTCG", 319),("GCTGA", 438),("GCTGT", 71),("GCTGC", 419),("GCTGG", 277),
("GCCAA", 421),("GCCAT", 207),("GCCAC", 422),("GCCAG", 373),("GCCTA", 427),("GCCTT", 41),("GCCTC", 405),("GCCTG", 255),("GCCCA", 423),("GCCCT", 156),("GCCCC", 424),("GCCCG", 338),("GCCGA", 425),("GCCGT", 101),
("GCCGC", 426),("GCCGG", 299),("GCGAA", 428),("GCGAT", 195),("GCGAC", 429),("GCGAG", 365),("GCGTA", 433),("GCGTT", 25),("GCGTC", 397),("GCGTG", 243),("GCGCA", 430),("GCGCT", 143),("GCGCC", 431),("GCGCG", 329),
("GCGGA", 432),("GCGGT", 86),("GCGGC", 426),("GCGGG", 288),("GGAAA", 440),("GGAAT", 216),("GGAAC", 441),("GGAAG", 379),("GGATA", 445),("GGATT", 52),("GGATC", 410),("GGATG", 263),("GGACA", 442),("GGACT", 166),
("GGACC", 443),("GGACG", 345),("GGAGA", 444),("GGAGT", 111),("GGAGC", 437),("GGAGG", 306),("GGTAA", 457),("GGTAT", 179),("GGTAC", 458),("GGTAG", 354),("GGTTA", 461),("GGTTT", 5),("GGTTC", 387),("GGTTG", 228),
("GGTCA", 459),("GGTCT", 125),("GGTCC", 443),("GGTCG", 316),("GGTGA", 460),("GGTGT", 67),("GGTGC", 417),("GGTGG", 274),("GGCAA", 446),("GGCAT", 204),("GGCAC", 447),("GGCAG", 371),("GGCTA", 451),("GGCTT", 37),
("GGCTC", 403),("GGCTG", 252),("GGCCA", 448),("GGCCT", 153),("GGCCC", 449),("GGCCG", 336),("GGCGA", 450),("GGCGT", 97),("GGCGC", 431),("GGCGG", 296),("GGGAA", 452),("GGGAT", 192),("GGGAC", 453),("GGGAG", 363),
("GGGTA", 456),("GGGTT", 21),("GGGTC", 395),("GGGTG", 240),("GGGCA", 454),("GGGCT", 139),("GGGCC", 449),("GGGCG", 326),("GGGGA", 455),("GGGGT", 82),("GGGGC", 424),("GGGGG", 285),];



const FOURMER_RAY: [(&'static str, usize);256] =[("AAAA", 0),("AAAT", 3),("AAAC", 1),("AAAG", 2),("AATA", 12),("AATT", 15),("AATC", 13),("AATG", 14),("AACA", 4),("AACT", 7),("AACC", 5),("AACG", 6),("AAGA", 8),("AAGT", 11),("AAGC", 9),("AAGG", 10),("ATAA", 45),("ATAT", 48),("ATAC", 46),("ATAG", 47),("ATTA", 55),("ATTT", 3),("ATTC", 56),("ATTG", 57),("ATCA", 49),("ATCT", 34),("ATCC", 50),("ATCG", 51),("ATGA", 52),("ATGT", 19),("ATGC", 53),("ATGG", 54),("ACAA", 16),("ACAT", 19),("ACAC", 17),("ACAG", 18),("ACTA", 28),("ACTT", 11),("ACTC", 29),("ACTG", 30),("ACCA", 20),("ACCT", 23),("ACCC", 21),("ACCG", 22),("ACGA", 24),("ACGT", 27),("ACGC", 25),("ACGG", 26),("AGAA", 31),("AGAT", 34),("AGAC", 32),("AGAG", 33),("AGTA", 42),("AGTT", 7),("AGTC", 43),("AGTG", 44),("AGCA", 35),("AGCT", 38),("AGCC", 36),("AGCG", 37),("AGGA", 39),("AGGT", 23),("AGGC", 40),("AGGG", 41),("TAAA", 126),("TAAT", 55),("TAAC", 125),("TAAG", 98),("TATA", 129),("TATT", 12),("TATC", 106),("TATG", 67),("TACA", 127),("TACT", 42),("TACC", 120),("TACG", 89),("TAGA", 128),("TAGT", 28),("TAGC", 114),("TAGG", 79),("TTAA", 135),("TTAT", 45),("TTAC", 121),("TTAG", 91),("TTTA", 126),("TTTT", 0),("TTTC", 100),("TTTG", 58),("TTCA", 133),("TTCT", 31),("TTCC", 115),("TTCG", 81),("TTGA", 130),("TTGT", 16),("TTGC", 108),("TTGG", 70),("TCAA", 130),("TCAT", 52),("TCAC", 124),("TCAG", 96),("TCTA", 128),("TCTT", 8),("TCTC", 104),("TCTG", 64),("TCCA", 131),("TCCT", 39),("TCCC", 119),("TCCG", 87),("TCGA", 132),("TCGT", 24),("TCGC", 112),("TCGG", 76),("TGAA", 133),("TGAT", 49),("TGAC", 123),("TGAG", 94),("TGTA", 127),("TGTT", 4),("TGTC", 102),("TGTG", 61),("TGCA", 134),("TGCT", 35),("TGCC", 117),("TGCG", 84),("TGGA", 131),("TGGT", 20),("TGGC", 110),("TGGG", 73),("CAAA", 58),("CAAT", 57),("CAAC", 59),("CAAG", 60),("CATA", 67),("CATT", 14),("CATC", 68),("CATG", 69),("CACA", 61),("CACT", 44),("CACC", 62),("CACG", 63),("CAGA", 64),("CAGT", 30),("CAGC", 65),("CAGG", 66),("CTAA", 91),("CTAT", 47),("CTAC", 92),("CTAG", 93),("CTTA", 98),("CTTT", 2),("CTTC", 99),("CTTG", 60),("CTCA", 94),("CTCT", 33),("CTCC", 95),("CTCG", 83),("CTGA", 96),("CTGT", 18),("CTGC", 97),("CTGG", 72),("CCAA", 70),("CCAT", 54),("CCAC", 71),("CCAG", 72),("CCTA", 79),("CCTT", 10),("CCTC", 80),("CCTG", 66),("CCCA", 73),("CCCT", 41),("CCCC", 74),("CCCG", 75),("CCGA", 76),("CCGT", 26),("CCGC", 77),("CCGG", 78),("CGAA", 81),("CGAT", 51),("CGAC", 82),("CGAG", 83),("CGTA", 89),("CGTT", 6),("CGTC", 90),("CGTG", 63),("CGCA", 84),("CGCT", 37),("CGCC", 85),("CGCG", 86),("CGGA", 87),("CGGT", 22),("CGGC", 88),("CGGG", 75),("GAAA", 100),("GAAT", 56),("GAAC", 101),("GAAG", 99),("GATA", 106),("GATT", 13),("GATC", 107),("GATG", 68),("GACA", 102),("GACT", 43),("GACC", 103),("GACG", 90),("GAGA", 104),("GAGT", 29),("GAGC", 105),("GAGG", 80),("GTAA", 121),("GTAT", 46),("GTAC", 122),("GTAG", 92),("GTTA", 125),("GTTT", 1),("GTTC", 101),("GTTG", 59),("GTCA", 123),("GTCT", 32),("GTCC", 116),("GTCG", 82),("GTGA", 124),("GTGT", 17),("GTGC", 109),("GTGG", 71),("GCAA", 108),("GCAT", 53),("GCAC", 109),("GCAG", 97),("GCTA", 114),("GCTT", 9),("GCTC", 105),("GCTG", 65),("GCCA", 110),("GCCT", 40),("GCCC", 111),("GCCG", 88),("GCGA", 112),("GCGT", 25),("GCGC", 113),("GCGG", 77),("GGAA", 115),("GGAT", 50),("GGAC", 116),("GGAG", 95),("GGTA", 120),("GGTT", 5),("GGTC", 103),("GGTG", 62),("GGCA", 117),("GGCT", 36),("GGCC", 118),("GGCG", 85),("GGGA", 119),("GGGT", 21),("GGGC", 111),("GGGG", 74),];





const THREEMER_RAY: [(&'static str, usize);64] = [("AAA", 0),("AAT", 3),("AAC", 1),("AAG", 2),("ATA", 11),("ATT", 3),("ATC", 12),("ATG", 13),("ACA", 4),
 ("ACT", 7),("ACC", 5),("ACG", 6),("AGA", 8),("AGT", 7),("AGC", 9),("AGG", 10),("TAA", 30),("TAT", 11),
 ("TAC", 29),("TAG", 22),("TTA", 30),("TTT", 0),("TTC", 24),("TTG", 14),("TCA", 31),("TCT", 8),("TCC", 28),
 ("TCG", 20),("TGA", 31),("TGT", 4),("TGC", 26),("TGG", 17),("CAA", 14),("CAT", 13),("CAC", 15),("CAG", 16),
 ("CTA", 22),("CTT", 2),("CTC", 23),("CTG", 16),("CCA", 17),("CCT", 10),("CCC", 18),("CCG", 19),("CGA", 20),
 ("CGT", 6),("CGC", 21),("CGG", 19),("GAA", 24),("GAT", 12),("GAC", 25),("GAG", 23),("GTA", 29),("GTT", 1),("GTC", 25),("GTG", 15),("GCA", 26),("GCT", 9),("GCC", 27),("GCG", 21),
 ("GGA", 28),("GGT", 5),("GGC", 27),("GGG", 18),];

const TWOMER_RAY:[(&'static str, usize);16] =  [("AA", 0),("AT", 3),("AC", 1),("AG", 2),("TA", 9),("TT", 0),("TC", 7),("TG", 4),("CA", 4),("CT", 2),("CC", 5),("CG", 6),("GA", 7),("GT", 1),("GC", 8),("GG", 5),];

const ONEMER_RAY:[(&'static str, usize);4]  = [("A", 0),("T", 0),("C", 1),("G", 1),];

const fn reverse_bits_byte(b: u64) -> u64  {
    return (((b * 0x0802u64 & 0x22110u64) | (b * 0x8020u64 & 0x88440u64)) * 0x10101u64 >> 16) & 0xFF;
}

// works on K <= 16
const fn ry_rev_comp<const K: usize>(rep: u32) -> u32 {
    // divide and conquer reverseak if bits.
    let rep = rep as u64;
    let x = reverse_bits_byte(((0xFF00 & rep) >> 8) as u64) |  ( reverse_bits_byte(0xFF & rep) << 8) ;
    // right align so we are "left padded" / using least significant bits, take complement, mask to len K (& ((1<<K) -1) ).
    return (((x >> (16-K)) ^ 0xFFFF) & ((1<<K) -1)) as u32 ;

}

const fn gen_ry_table<const K: usize, const N:usize>() -> [u32; N] {
    let mut tbl_idx = 0;
    let mut tbl = [u32::MAX; N];
    let mut map_index = 0;
    loop {
        if tbl_idx >= N { break; }
        
        if tbl[tbl_idx] == u32::MAX {
            tbl[tbl_idx] = map_index;
            tbl[ry_rev_comp::<K>(tbl_idx as u32) as usize]= map_index;
            map_index += 1;

        }
        tbl_idx += 1;
    }
    return tbl;
}

const RY10TBL: [u32;1024]  = gen_ry_table::<10,1024>();
const RY9TBL: [u32;512]  = gen_ry_table::<9,512>();
const RY8TBL: [u32;256]  = gen_ry_table::<8,256>();
const RY7TBL: [u32;128]  = gen_ry_table::<7,128>();
const RY6TBL: [u32;64]  = gen_ry_table::<6,64>();

// /// Maps individual DNA bases from utf-8 to 1 bit representation
const fn base2number_ry(c: u8) -> Option<u8> {
    match c {
        b'A'| b'a' |  b'G'| b'g' => Some(0),
        b'T'| b't' | b'C'| b'c'  => Some(1),
        _ => None,
    }
}

/// Maps 4mer from utf-8 to 8 bit representation
const fn kmer_to_16_bit_ray_ry<const N: usize>( mer:  & [u8]) -> Option<u32> {
    let mut result: u32 = 0;
    let mut i = 0;
    loop {
        if i >= N {
            break;
        }
        let c = mer[i];
        result = result << 1;
        result |= match base2number_ry(c) {
            Some(b) => b as u32,
            None => {
                return None;
            }
        };
        i += 1;
    }

    Some(result)
}


fn subm_rymers(bin_rep: u32, N: usize) -> (u32, u32) {
    let mask = ((1u32<<(N-1))-1);
    return ((mask & bin_rep) ,  (((mask << 1) & bin_rep) >> 1));
}

fn quotient_ry_mers<const A: usize, const B: usize, const C: usize, const D: usize >(mut a:[f32;A],b:[f32;B] , a_tbl: [u32;C] , b_tbl: [u32;D], log2c: usize ) -> [f32;A] {
    let mut aleady_quotiented = [false; A];

    for kmer in 0..C {
        let canon_idx = a_tbl[kmer];

        if aleady_quotiented[canon_idx as usize] { continue; }

        aleady_quotiented[canon_idx as usize] = true;

        let (submer_1, submer_2) = subm_rymers(kmer as u32,log2c);

        let (count1, count2) = (b[b_tbl[submer_1 as usize] as usize], b[b_tbl[submer_2 as usize] as usize]);

        let denom = count1*count2;

        if denom > 0.000001 {
            a[canon_idx as usize]/= denom
        }
    }
    return a;
}


fn find_rymers_bytes(contig: &[u8]) -> ([f32; 528],[f32; 256], [f32; 136],  [f32; 64],  [f32; 36], usize) {
    let mut sixmer_counts = [0u32; 36];
    let mut sevenmer_counts = [0u32; 64];
    let mut eightmer_counts = [0u32; 136];
    let mut ninemer_counts = [0u32; 256];
    let mut tenmer_counts = [0u32; 528];

    let mut invalid_count = 0;

    let mut fmer = [0u8;10];

    for i in 0..9 {
        fmer[i+1] = contig[i];
    }

   for b in  contig[9..].iter() {
       for i in 0..9 {
           fmer[i] = fmer[i+1];
       }
       fmer[9] = *b;
       match(kmer_to_16_bit_ray_ry::<10>(&fmer)) {
            Some(f) => {
                let class = RY10TBL[f as usize];
                tenmer_counts[class as usize] += 1;
            },
            None => {
                invalid_count += 1;
            },
       }
       match(kmer_to_16_bit_ray_ry::<9>(&fmer)) {
            Some(f) => {
                let class = RY9TBL[f as usize];
                ninemer_counts[class as usize] += 1;
            },
            None => (),
        }
        match(kmer_to_16_bit_ray_ry::<8>(&fmer)) {
            Some(f) => {
                let class = RY8TBL[f as usize];
                eightmer_counts[class as usize] += 1;
            },
            None => (),
        }
        match(kmer_to_16_bit_ray_ry::<7>(&fmer)) {
            Some(f) => {
                let class = RY7TBL[f as usize];
                sevenmer_counts[class as usize] += 1;
            }
            None => (),
        }
        match(kmer_to_16_bit_ray_ry::<6>(&fmer)) {
            Some(f) => {
                let class = RY6TBL[f as usize];
                sixmer_counts[class as usize] += 1;
            }
            None => (),
        }
   }



   let out_10mer  = norm_vector(tenmer_counts);
   let out_9mer  = norm_vector(ninemer_counts);
   let out_8mer  = norm_vector(eightmer_counts);
   let out_7mer = norm_vector(sevenmer_counts);
   let out_6mer = norm_vector(sixmer_counts);

   let q10mer = norm_vector_f(quotient_ry_mers(out_10mer, out_9mer, RY10TBL, RY9TBL,10));
   let q9mer = norm_vector_f(quotient_ry_mers(out_9mer, out_8mer, RY9TBL, RY8TBL,9));

   let q8mer = norm_vector_f(quotient_ry_mers(out_8mer, out_7mer, RY8TBL, RY7TBL,8));
   let q7mer = norm_vector_f(quotient_ry_mers(out_7mer, out_6mer, RY7TBL, RY6TBL,7));

   (q10mer,q9mer,q8mer,q7mer,out_6mer, invalid_count)
}

fn find_rymers(contig: &str) -> ([f32; 528],[f32; 256], [f32; 136],  [f32; 64],  [f32; 36], usize) {
    let mut sixmer_counts = [0u32; 36];
    let mut sevenmer_counts = [0u32; 64];
    let mut eightmer_counts = [0u32; 136];
    let mut ninemer_counts = [0u32; 256];
    let mut tenmer_counts = [0u32; 528];

    let mut invalid_count = 0;

    let mut fmer = [0u8;10];

    for i in 0..9 {
        fmer[i+1] = contig.as_bytes()[i];
    }

   for b in  contig.as_bytes()[9..].iter().filter(|p| **p != b'\n') {
       for i in 0..9 {
           fmer[i] = fmer[i+1];
       }
       fmer[9] = *b;
       match(kmer_to_16_bit_ray_ry::<10>(&fmer)) {
            Some(f) => {
                let class = RY10TBL[f as usize];
                tenmer_counts[class as usize] += 1;
            },
            None => {
                invalid_count += 1;
            },
       }
       match(kmer_to_16_bit_ray_ry::<9>(&fmer)) {
            Some(f) => {
                let class = RY9TBL[f as usize];
                ninemer_counts[class as usize] += 1;
            },
            None => (),
        }
        match(kmer_to_16_bit_ray_ry::<8>(&fmer)) {
            Some(f) => {
                let class = RY8TBL[f as usize];
                eightmer_counts[class as usize] += 1;
            },
            None => (),
        }
        match(kmer_to_16_bit_ray_ry::<7>(&fmer)) {
            Some(f) => {
                let class = RY7TBL[f as usize];
                sevenmer_counts[class as usize] += 1;
            }
            None => (),
        }
        match(kmer_to_16_bit_ray_ry::<6>(&fmer)) {
            Some(f) => {
                let class = RY6TBL[f as usize];
                sixmer_counts[class as usize] += 1;
            }
            None => (),
        }
   }



   let out_10mer  = norm_vector(tenmer_counts);
   let out_9mer  = norm_vector(ninemer_counts);
   let out_8mer  = norm_vector(eightmer_counts);
   let out_7mer = norm_vector(sevenmer_counts);
   let out_6mer = norm_vector(sixmer_counts);

   let q10mer = norm_vector_f(quotient_ry_mers(out_10mer, out_9mer, RY10TBL, RY9TBL,10));
   let q9mer = norm_vector_f(quotient_ry_mers(out_9mer, out_8mer, RY9TBL, RY8TBL,9));

   let q8mer = norm_vector_f(quotient_ry_mers(out_8mer, out_7mer, RY8TBL, RY7TBL,8));
   let q7mer = norm_vector_f(quotient_ry_mers(out_7mer, out_6mer, RY7TBL, RY6TBL,7));

   (q10mer,q9mer,q8mer,q7mer,out_6mer, invalid_count)


} 

// Parallel read.
fn par_read(
    file: &mut File,
    read_buffer: &mut Vec<u8>,
    file_size: u64,
) {
    let chunk_size:u64 = 1024 * 1024 *1024;
    //dbg!("preresize");
    //read_buffer.resize(file_size as usize, 0);
    //dbg!("resized");
    // read_buffer
    //     .chunks_mut(chunk_size as usize)
    //     .enumerate()
    //     .for_each(|(idx, buffer)| {
    //         // reads  buffer length (max chunk_size), offset second parameter
    //         // gives stack trace if fails (e.g. bad math, can't acess file)
    //         file.read_exact_at(
    //             buffer,
    //             chunk_size * (idx as u64),
    //         )
    //         .unwrap();
    //     });
    file.read_to_end(read_buffer).unwrap();
   // dbg!(&read_buffer[0..150]);
}




const FIVEMERTABLE: [u32;1024] = gen_Nmer_table::<1024,5>(FIVEMER_RAY);
const FOURMERTABLE: [u32;256] = gen_Nmer_table::<256,4>(FOURMER_RAY);
const THREEMERTABLE: [u32;64] = gen_Nmer_table::<64,3>(THREEMER_RAY);
const TWOMERTABLE: [u32;16] = gen_Nmer_table::<16,2>(TWOMER_RAY);
const ONEMERTABLE: [u32;4] = gen_Nmer_table::<4,1>(ONEMER_RAY);


const fn gen_multipliers() -> [(u8,u8);136] {
    let mut result = [(0,0); 136];
    let mut i = 0;
    loop {
        if i >= 256 {
            break;
        }
        let  (mut count_at, mut count_cg) = (0u8,0u8);
        let four_mer = FOURMER_RAY[i].0.as_bytes();

        match four_mer[0] {
           b'A'|b'T'=> {
                count_at += 1;
            },
            _ => {
                count_cg += 1;
            },
        }
        match four_mer[1] {
            b'A'|b'T' => {
                count_at += 1;
            },
            _ => {
                count_cg += 1;
            },
        }
        match four_mer[2] {
            b'A'|b'T' => {
                count_at += 1;
            },
            _ => {
                count_cg += 1;
            },
        }
        match four_mer[3] {
            b'A'|b'T' => {
                count_at += 1;
            },
            _ => {
                count_cg += 1;
            },
        }

        let loc = FOURMER_RAY[i].1;

        result[loc] = (count_at, count_cg);
        i += 1;
    }
    result
}
const MULTIPLIERS:[(u8,u8);136] = gen_multipliers();

// /// Maps individual DNA bases from utf-8 to 2 bit representation
const fn base2number(c: u8) -> Option<u8> {
    match c {
        b'A'| b'a' => Some(0),
        b'T'| b't' => Some(1),
        b'C'| b'c' => Some(2),
        b'G'| b'g' => Some(3),
        _ => None,
    }
}

/// Maps 4mer from utf-8 to 8 bit representation
const fn kmer_to_16_bit<const N: usize>( mer:  & str) -> Option<u32> {
    let mut result: u32 = 0;

    let b = mer.as_bytes();
    let mut i = 0;
    loop {
        if i >= N {
            break;
        }
        let c = b[i];
        result = result << 2;
        result |= match base2number(c) {
            Some(b) => b as u32,
            None => {
                return None;
            }
        };
        i += 1;
    }

    Some(result)
}


/// Maps 4mer from utf-8 to 8 bit representation
const fn kmer_to_16_bit_ray<const N: usize>( mer:  & [u8]) -> Option<u32> {
    let mut result: u32 = 0;
    let mut i = 0;
    loop {
        if i >= N {
            break;
        }
        let c = mer[i];
        result = result << 2;
        result |= match base2number(c) {
            Some(b) => b as u32,
            None => {
                return None;
            }
        };
        i += 1;
    }

    Some(result)
}


const fn gen_Nmer_table<const N: usize, const K: usize>(mapping: [(&'static str, usize);N] ) -> [u32; N] {
    let mut result = [0u32; N];

    let mut i: usize = 0;
    loop {
        if i >= N {
            break;
        }
        match kmer_to_16_bit::<K>(mapping[i].0) {
            Some(x) => {
                result[x as usize] = mapping[i].1 as u32;
            } 
            None => unreachable!(),
        }
        i += 1;
    }

    result
}

fn norm_vector<const N: usize>(in_vec: [u32;N]) -> [f32; N] {
    let sum = in_vec.iter()
                    .map(|&x| x as u64)
                    .sum::<u64>();
    
    let mut result =  [0.0; N];

    for i in 0..N {
        result[i] = ((in_vec[i] as f64)/(sum as f64)) as f32;
    }

    result
}


fn norm_vector_f<const N: usize>(in_vec: [f32;N]) -> [f32; N] {
    let sum = in_vec.iter()
                    .map(|&x| (x as f64)*(x as f64))
                    .sum::<f64>()
                    .sqrt();
    
    let mut result =  [0.0; N];

    if sum > 0. {
        for i in 0..N {
            result[i] = ((in_vec[i] as f64)/(sum as f64)) as f32;
        }
    }

    result
}



fn contig_2_nmer_distrs_bytes(contig: &[u8]) -> ([f32; 512], [f32; 136], [f32; 136], [f32; 32], [f32; 10], [f32; 2], usize) {
    let mut onemer_counts = [0u32; 2];
    let mut twomer_counts = [0u32; 10];
    let mut threemer_counts = [0u32; 32];
    let mut fourmer_counts = [0u32; 136];
    let mut fivemer_counts = [0u32; 512];

    let mut invalid_count = 0;

    let mut fmer = [
        0,
        contig[0],
        contig[1],
        contig[2],
        contig[3],
    ];

    for b in contig[4..].iter() {
        fmer[0] = fmer[1];
        fmer[1] = fmer[2];
        fmer[2] = fmer[3];
        fmer[3] = fmer[4];
        fmer[4] = *b;
    

        match kmer_to_16_bit_ray::<5>(&fmer) {
            Some(f) => {
                let class = FIVEMERTABLE[f as usize];
                fivemer_counts[class as usize] += 1;
            }
            None => {
                invalid_count += 1;
            },
        }
    
        match kmer_to_16_bit_ray::<4>(&fmer[..4]) {
            Some(f) => {
                let class = FOURMERTABLE[f as usize];
                fourmer_counts[class as usize] += 1;
            }
            None => (),
        }
    
        match kmer_to_16_bit_ray::<3>(&fmer[..3]) {
            Some(f) => {
                let class = THREEMERTABLE[f as usize];
                threemer_counts[class as usize] += 1;
            }
            None => (),
        }

        match kmer_to_16_bit_ray::<2>(&fmer[..2]) {
            Some(f) => {
                let class = TWOMERTABLE[f as usize];
                twomer_counts[class as usize] += 1;
            }
            None => (),
        }

        match kmer_to_16_bit_ray::<1>(&fmer[..1]) {
            Some(f) => {
                let class = ONEMERTABLE[f as usize];
                onemer_counts[class as usize] += 1;
            }
            None => (),
        }

    }

    
    match kmer_to_16_bit_ray::<4>(&fmer[1..5]) {
        Some(f) => {
            let class = FOURMERTABLE[f as usize];
            fourmer_counts[class as usize] += 1;
        }
        None => (),
    }

    for i in 1..3 {
        match kmer_to_16_bit_ray::<3>(&fmer[i..3+i]) {
            Some(f) => {
                let class = THREEMERTABLE[f as usize];
                threemer_counts[class as usize] += 1;
            }
            None => (),
        }
    }
    for i in 1..4 {
        match kmer_to_16_bit_ray::<2>(&fmer[i..2+i]) {
            Some(f) => {
                let class = TWOMERTABLE[f as usize];
                twomer_counts[class as usize] += 1;
            }
            None => (),
        }
    }
    for i in 1..5 {
        match kmer_to_16_bit_ray::<1>(&fmer[i..1+i]) {
            Some(f) => {
                let class = ONEMERTABLE[f as usize];
                onemer_counts[class as usize] += 1;
            }
            None => (),
        }
    }



    let out_5mer  = norm_vector(fivemer_counts);
    let out_4mer  = norm_vector(fourmer_counts);
    let out_3mer  = norm_vector(threemer_counts);
    let out_2mer = norm_vector(twomer_counts);
    let out_1mer = norm_vector(onemer_counts);

    let mut fourmer_multipliers =  [0f64;136];
    for i in 0..136 {
        let mut denom: f64 = 1.0;
        for _ in 0..MULTIPLIERS[i].0 {
            denom *= out_1mer[0] as f64;
           // assert!(onemer_counts[0]  != 0);
        }
        for _ in 0..MULTIPLIERS[i].1 {
            denom *= out_1mer[1] as f64;
          //  assert!(onemer_counts[1]  != 0);

        }
        fourmer_multipliers[i] = denom as f64;
    }

    let mut out_l4n1 = [0.0f32; 136];

    for i in 0..136 {
        let denom = fourmer_multipliers[i];
        // if denom == 0 and fourmer_counts[i] != 0 {
        //     panic!();
        // }
        out_l4n1[i] = if denom > 0.0 {
            ((out_4mer[i] as f64)/denom) as f32
        } else {
            if out_4mer[i]  > 0.0 {
                dbg!(&fourmer_counts[i]);
                dbg!(&fourmer_multipliers[i]);
                panic!();
            }
            0.0
        };
    }

    (out_5mer, out_4mer, out_4mer, out_3mer, out_2mer, out_1mer, invalid_count)
}


// /// Given a contig of DNA bases, calculates L2 normed
// /// "Distribution" over contigs 4mer count, where 4mers
// /// are quotiented by reverse complement.
fn contig_2_nmer_distrs(contig: &str) -> ([f32; 512], [f32; 136], [f32; 136], [f32; 32], [f32; 10], [f32; 2], usize) {
    let mut onemer_counts = [0u32; 2];
    let mut twomer_counts = [0u32; 10];
    let mut threemer_counts = [0u32; 32];
    let mut fourmer_counts = [0u32; 136];
    let mut fivemer_counts = [0u32; 512];

    let mut invalid_count = 0;

    let mut fmer = [
        0,
        contig.as_bytes()[0],
        contig.as_bytes()[1],
        contig.as_bytes()[2],
        contig.as_bytes()[3],
    ];

    for b in contig.as_bytes()[4..].iter().filter(|p| **p != b'\n') {
        fmer[0] = fmer[1];
        fmer[1] = fmer[2];
        fmer[2] = fmer[3];
        fmer[3] = fmer[4];
        fmer[4] = *b;
    

        match kmer_to_16_bit_ray::<5>(&fmer) {
            Some(f) => {
                let class = FIVEMERTABLE[f as usize];
                fivemer_counts[class as usize] += 1;
            }
            None => {
                invalid_count += 1;
            },
        }
    
        match kmer_to_16_bit_ray::<4>(&fmer[..4]) {
            Some(f) => {
                let class = FOURMERTABLE[f as usize];
                fourmer_counts[class as usize] += 1;
            }
            None => (),
        }
    
        match kmer_to_16_bit_ray::<3>(&fmer[..3]) {
            Some(f) => {
                let class = THREEMERTABLE[f as usize];
                threemer_counts[class as usize] += 1;
            }
            None => (),
        }

        match kmer_to_16_bit_ray::<2>(&fmer[..2]) {
            Some(f) => {
                let class = TWOMERTABLE[f as usize];
                twomer_counts[class as usize] += 1;
            }
            None => (),
        }

        match kmer_to_16_bit_ray::<1>(&fmer[..1]) {
            Some(f) => {
                let class = ONEMERTABLE[f as usize];
                onemer_counts[class as usize] += 1;
            }
            None => (),
        }

    }

    
    match kmer_to_16_bit_ray::<4>(&fmer[1..5]) {
        Some(f) => {
            let class = FOURMERTABLE[f as usize];
            fourmer_counts[class as usize] += 1;
        }
        None => (),
    }

    for i in 1..3 {
        match kmer_to_16_bit_ray::<3>(&fmer[i..3+i]) {
            Some(f) => {
                let class = THREEMERTABLE[f as usize];
                threemer_counts[class as usize] += 1;
            }
            None => (),
        }
    }
    for i in 1..4 {
        match kmer_to_16_bit_ray::<2>(&fmer[i..2+i]) {
            Some(f) => {
                let class = TWOMERTABLE[f as usize];
                twomer_counts[class as usize] += 1;
            }
            None => (),
        }
    }
    for i in 1..5 {
        match kmer_to_16_bit_ray::<1>(&fmer[i..1+i]) {
            Some(f) => {
                let class = ONEMERTABLE[f as usize];
                onemer_counts[class as usize] += 1;
            }
            None => (),
        }
    }



    let out_5mer  = norm_vector(fivemer_counts);
    let out_4mer  = norm_vector(fourmer_counts);
    let out_3mer  = norm_vector(threemer_counts);
    let out_2mer = norm_vector(twomer_counts);
    let out_1mer = norm_vector(onemer_counts);


    let mut fourmer_multipliers =  [0f64;136];
    for i in 0..136 {
        let mut denom: f64 = 1.0;
        for _ in 0..MULTIPLIERS[i].0 {
            denom *= out_1mer[0] as f64;
           // assert!(onemer_counts[0]  != 0);
        }
        for _ in 0..MULTIPLIERS[i].1 {
            denom *= out_1mer[1] as f64;
          //  assert!(onemer_counts[1]  != 0);

        }
        fourmer_multipliers[i] = denom as f64;
    }

    let mut out_l4n1 = [0.0f32; 136];

    for i in 0..136 {
        let denom = fourmer_multipliers[i];
        // if denom == 0 and fourmer_counts[i] != 0 {
        //     panic!();
        // }
        out_l4n1[i] = if denom > 0.0 {
            ((out_4mer[i] as f64)/denom) as f32
        } else {
            if out_4mer[i]  > 0.0 {
                dbg!(&fourmer_counts[i]);
                dbg!(&fourmer_multipliers[i]);
                panic!();
            }
            0.0
        };
    }


    (out_5mer, out_4mer, out_4mer, out_3mer, out_2mer, out_1mer, invalid_count)
}

use numpy::IntoPyArray;
use std::io::Write;
use pyo3::Python;
use pyo3::types::PyString;
use flate2::read::GzDecoder;
use std::io::Read;
//#use std::str::pattern::Pattern;
use numpy::array::PyArray1;
use hashbrown::hash_map::HashMap;
use std::path::Path;

use numpy::array::PyArray;
use rayon::iter::ParallelDrainRange;
use pyo3::types::PySequence;



// #[pymodule]
// fn DataBase(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
//     m.add_class::<FastaDataBase>()?;
//     Ok(())
// }

#[pymodule]
fn kmer_counter(_py: Python<'_>, m: &PyModule) -> PyResult<()> {

    #[pyclass]
    #[pyo3(name = "FastaDataBase")]
    struct FastaDataBase {
        file_contents: Vec<Vec<u8>>,
        // list of all contigs; first tuple element describes start position in some file_contents[i] for unknown  i;
        // third gives len
        contigs: Vec<(usize, usize)>,
        // contigs_start[i]: first `contigs` index to loo for genome[i]
        // contigs_start[i+1] - 1 : wlog, last `contigs` index for genome[i]
        // to access first contig for genome i
        //
        // first_idx = contigs_start[i]
        // lst contig idx =  contigs_start[i+1] -1
        //
        // file_contents[i][contigs[first_idx].0 .. first_idx].1] <-- to tsring
        contigs_start: Vec<usize>,

        // weight distribution for drawing contigs  from this genome (proportional to length of contig)
        weight_index : Vec<WeightedIndex<usize>>,
    }

    m.add_class::<FastaDataBase>()?;

impl FastaDataBase  {
    fn get_contig_slice(&self, file_idx: usize, contig_idx :usize , pos: usize, len: usize) -> &[u8] {
        &(&self.file_contents[file_idx][self.contigs[self.contigs_start[file_idx] + contig_idx].0..self.contigs[self.contigs_start[file_idx] + contig_idx].0+self.contigs[self.contigs_start[file_idx] + contig_idx].1])[pos..pos+len]
    }
}


#[pymethods]
impl FastaDataBase {
    #[new]
    fn new(contig_file_paths: Vec<String>, min_len: usize) -> Self {

        let mut result = Self {
            file_contents: Vec::new(),
            contigs: Vec::new(),
            contigs_start: Vec::new(),
            weight_index: Vec::new(),
        };

        let bar = ProgressBar::new(contig_file_paths.len() as u64);

        for contig_file in contig_file_paths.iter() {

            let file_info = fs::metadata(contig_file).unwrap();
            let file_size = file_info.len();

            let mut file = File::open(contig_file).unwrap();

            //read using multiple file handles for faster spead
            let mut read_buffer: Vec<u8> = Vec::new();
            par_read(&mut file, &mut read_buffer, file_size);

            // if needs to be decompressed,  replace dataa buffer with decompressed one, otherwise leave as is;
            // 'z' -> check for gz

            let read_buffer = if contig_file.len() >3 && b'z'==contig_file.as_bytes()[contig_file.len()-1] {
                let mut decom_buffer: Vec<u8> = Vec::new();
                let mut gz = GzDecoder::new(&read_buffer[..]);
                gz.read_to_end(&mut  decom_buffer).unwrap();
                decom_buffer
            } else {
                read_buffer
            };


            result.contigs_start.push(result.contigs.len());

            let mut current_pos_in_buffer = 0;

            let mut contig_buffer = Vec::new();

            // a vector of contig lengths, this is how we will weight draws
            let mut to_weights = Vec::new();

            read_buffer.split_inclusive(|x| *x == b'>')
                                   .for_each(|name_contig| {
                                    

                                        if let Some(end_of_name_pos) = name_contig.iter().position(|&r| r == b'\n') {
                                            // leaves '\n' and '>', but we handle those when counting kmers

                                                               // leaves '\n' and '>', but we handle those when counting kmers

                                            //check if meets min size
                                            let size = name_contig[end_of_name_pos+1..].iter().filter(|&&x| x != b'\n' && x != b'>').count();


                                            if size >= min_len {
                                                //dbg!("included");
                                                contig_buffer.extend(name_contig[end_of_name_pos+1..].iter().filter(|&&x| x != b'\n' && x != b'>'));
                                                result.contigs.push((current_pos_in_buffer, size));
                                                current_pos_in_buffer += size;
                                                to_weights.push(size);
                                            } 
                                            //dbg!("hi", size);
                                            // result.contigs.push((current_pos_in_buffer+end_of_name_pos+1, name_contig[end_of_name_pos+1..].len() ));
                                            
                                        }
                                        
                                   });
            if let Ok(a) = WeightedIndex::new(&to_weights) {
                result.weight_index.push(a);
                result.file_contents.push(contig_buffer);
            } else {
                result.contigs_start.pop();
            }
            bar.inc(1);
            
        }
        bar.finish();

        return result;
    }
    fn get_num_contig(&self, file_idx: usize) -> isize {
        if file_idx >= self.contigs_start.len() {
            return -1;
        }
        if file_idx == self.contigs_start.len()  -1 {
            return (self.contigs.len() - self.contigs_start[file_idx]) as isize;
        }
        return (self.contigs_start[file_idx+1] - self.contigs_start[file_idx]) as isize;
    }

    fn get_num_contig_unch(&self, file_idx: usize) -> usize {
        if file_idx == self.contigs_start.len()  -1 {
            return (self.contigs.len() - self.contigs_start[file_idx]);
        }
        return (self.contigs_start[file_idx+1] - self.contigs_start[file_idx]);
    }

    fn get_contig_size(&self, file_idx: usize, contig_idx: usize) -> isize {
        if file_idx >= self.contigs_start.len() {
            return -1;
        }
        let num_contigs = if file_idx == self.contigs_start.len()  -1 {
             (self.contigs.len() - self.contigs_start[file_idx])
        } else  { 
            (self.contigs_start[file_idx+1] - self.contigs_start[file_idx])
        };

        if contig_idx >= num_contigs {
            return -1;
        } else {
            return self.contigs[self.contigs_start[file_idx] + contig_idx].1 as isize;
        }
    }

    fn get_contig_size_unch(&self, file_idx: usize, contig_idx: usize) -> usize {
        return self.contigs[self.contigs_start[file_idx] + contig_idx].1;
    }
    fn get_contig(&self, file_idx: usize, contig_idx: usize) -> String {
        if file_idx >= self.contigs_start.len() {
            return String::new();
        }
        let num_contigs = if file_idx == self.contigs_start.len()  -1 {
             (self.contigs.len() - self.contigs_start[file_idx])
        } else  { 
            (self.contigs_start[file_idx+1] - self.contigs_start[file_idx])
        };

        if contig_idx >= num_contigs {
            return  String::new();
        } else {
            return std::str::from_utf8(&self.file_contents[file_idx][self.contigs[self.contigs_start[file_idx] + contig_idx].0..self.contigs[self.contigs_start[file_idx] + contig_idx].0+self.contigs[self.contigs_start[file_idx] + contig_idx].1]).unwrap().to_string();
        }
    }

    fn sample<'py>(&'py self, py: Python<'py>, n: usize, contig_sample_size: usize) -> (&'py PyArray1<f32>,&'py PyArray1<f32>,&'py PyArray1<f32>,&'py PyArray1<f32>,&'py PyArray1<f32>,&'py PyArray1<f32>, &'py PyArray1<f32>,&'py PyArray1<f32>,&'py PyArray1<f32>,&'py PyArray1<f32>,&'py PyArray1<usize> ) {
        let qq=  py.allow_threads(move || {
        let mut rng = rand::thread_rng();


        let to_sample = (0..n)
                .map(|_| {
                    let file_idx =rng.gen_range(0..self.file_contents.len());
                    let contig_idx = self.weight_index[file_idx].sample(&mut rng);
                    let start_pos =  rng.gen_range(0..self.get_contig_size_unch(file_idx, contig_idx)- contig_sample_size);
                    (file_idx,contig_idx,start_pos)
                })
                .collect::<Vec<(usize,usize,usize)>>();

        let pre_tens = to_sample.par_iter()
                .map(| (file_idx,contig_idx,start_pos)|  (self.get_contig_slice(*file_idx,*contig_idx,*start_pos, contig_sample_size), *file_idx))
                .map(|(ctg, file_idx)| (contig_2_nmer_distrs_bytes(ctg), find_rymers_bytes(ctg), file_idx) )
                .map( |mut x| { 
                    while   x.1.5 >= 100 {
                        let mut rng = rand::thread_rng();
                        let file_idx = x.2;
                        let contig_idx = self.weight_index[file_idx].sample(&mut rng);// rng.gen_range(0..self.get_num_contig_unch(file_idx));
                        let start_pos =  rng.gen_range(0..self.get_contig_size_unch(file_idx, contig_idx)- contig_sample_size);
                        let ctg = self.get_contig_slice(file_idx,contig_idx,start_pos, contig_sample_size);
                        x = (contig_2_nmer_distrs_bytes(ctg), find_rymers_bytes(ctg), file_idx) ;
                    }
                    x
                })
                .collect::<Vec<_>>();

        let pre_5mers = pre_tens
                                .par_iter()
                                .flat_map_iter(|i| i.0.0)
                                .collect::<Vec<_>>();
        
        let pre_l4n1mers = pre_tens
                                .par_iter()
                                .flat_map_iter(|i| i.0.1)
                                .collect::<Vec<_>>();

            
        let pre_4mers = pre_tens
                                .par_iter()
                                .flat_map_iter(|i| i.0.2)
                                .collect::<Vec<_>>();
        
        let pre_3mers = pre_tens
                                .par_iter()
                                .flat_map_iter(|i| i.0.3)
                                .collect::<Vec<_>>();
        
        let pre_2mers = pre_tens
                                .par_iter()
                                .flat_map_iter(|i| i.0.4)
                                .collect::<Vec<_>>();
            
        let pre_1mers = pre_tens
                                .par_iter()
                                .flat_map_iter(|i| i.0.5)
                                .collect::<Vec<_>>();
        
                                let pre_10mers = pre_tens
                                .par_iter()
                                .flat_map_iter(|i| i.0.0)
                                .collect::<Vec<_>>();

        let pre_9mers = pre_tens
                                .par_iter()
                                .flat_map_iter(|i| i.1.1)
                                .collect::<Vec<_>>();

        let pre_8mers = pre_tens
                                .par_iter()
                                .flat_map_iter(|i| i.1.2)
                                .collect::<Vec<_>>();
        let pre_7mers = pre_tens
                                .par_iter()
                                .flat_map_iter(|i| i.1.3)
                                .collect::<Vec<_>>();
        let pre_6mers = pre_tens
                                .par_iter()
                                .flat_map_iter(|i| i.1.4)
                                .collect::<Vec<_>>();
        let pre_10mers = pre_tens
                                .par_iter()
                                .flat_map_iter(|i| i.1.0)
                                .collect::<Vec<_>>();
        let label = pre_tens
                                .par_iter()
                                .map(|i| i.2)
                                .collect::<Vec<_>>();

  
        let valids = pre_tens
                                .par_iter()
                                .map(|i| i.0.6)
                                .collect::<Vec<_>>();

                        
                                // (
                                //     pre_5mers.into_pyarray(py),
                                //     pre_4mers.into_pyarray(py),
                                //     pre_3mers.into_pyarray(py),
                                //     pre_2mers.into_pyarray(py),
                                //     pre_1mers.into_pyarray(py),
                                //     pre_10mers.into_pyarray(py),
                                //     pre_9mers.into_pyarray(py),
                                //     pre_8mers.into_pyarray(py),
                                //     pre_7mers.into_pyarray(py),
                                //     pre_6mers.into_pyarray(py),
                                //     label.into_pyarray(py)
                                // )

                                (
                                    pre_5mers,
                                    pre_l4n1mers,
                                    pre_3mers,
                                    pre_2mers,
                                    pre_1mers,
                                    pre_10mers,
                                    pre_9mers,
                                    pre_8mers,
                                    pre_7mers,
                                    pre_6mers,
                                    label,
                                ) });

        (qq.0.into_pyarray(py),qq.1.into_pyarray(py),qq.2.into_pyarray(py),qq.3.into_pyarray(py),qq.4.into_pyarray(py),qq.5.into_pyarray(py),qq.6.into_pyarray(py),qq.7.into_pyarray(py),qq.8.into_pyarray(py),qq.9.into_pyarray(py),qq.10.into_pyarray(py))
        //unimplemented!();
        
        // PyArray::from_slice(py, &pre_5mers)
        
    }

        
    
}


    #[pyfn(m)]
    #[pyo3(name = "find_single")]
    pub fn find_single<'py>(py: Python<'py>, contig: &str) -> (&'py PyArray1<f32>,&'py PyArray1<f32>,&'py PyArray1<f32>,&'py PyArray1<f32>,&'py PyArray1<f32>,&'py PyArray1<f32>, &'py PyArray1<f32>,&'py PyArray1<f32>,&'py PyArray1<f32>,&'py PyArray1<f32>,&'py PyArray1<f32>,usize ){
        let (res,res2) = py.allow_threads(||( contig_2_nmer_distrs(contig),  find_rymers(contig)   ));
        return (PyArray::from_slice(py, &res.1), PyArray::from_slice(py, &res.0), PyArray::from_slice(py, &res.2), PyArray::from_slice(py, &res.3),PyArray::from_slice(py, &res.4),PyArray::from_slice(py, &res.5),
                PyArray::from_slice(py, &res2.0) ,PyArray::from_slice(py, &res2.1) ,PyArray::from_slice(py, &res2.2), PyArray::from_slice(py, &res2.3),PyArray::from_slice(py, &res2.4),   res2.5);
    }


    #[pyfn(m)]
    #[pyo3(name = "sampling")]
    pub fn sampling<'py>(py: Python<'py>, contigs: Vec<&str>, sample_size: usize) -> (&'py PyArray1<f32>,&'py PyArray1<f32>,&'py PyArray1<f32>,&'py PyArray1<f32>,&'py PyArray1<f32>,&'py PyArray1<f32>, &'py PyArray1<usize>) {
        let qqq=  py.allow_threads(||{
       
        let pre_tens = contigs
                .par_iter()
                .map(|ctg| {
                    let mut rng = rand::thread_rng();
                    let start = rng.gen_range(0..ctg.len()-sample_size);
                    return &ctg[start..start+sample_size]
                })
                .collect::<Vec<_>>();

        let pre_tens = contigs
                .par_iter()
                .enumerate()
                .map(|(a,&ctg)| contig_2_nmer_distrs(ctg))
                .collect::<Vec<_>>();
                //dbg!("here6");

        let pre_5mers = pre_tens
                                .par_iter()
                                .flat_map_iter(|i| i.0)
                                .collect::<Vec<_>>();
        
        let pre_l4n1mers = pre_tens
                                .par_iter()
                                .flat_map_iter(|i| i.1)
                                .collect::<Vec<_>>();

            
        let pre_4mers = pre_tens
                                .par_iter()
                                .flat_map_iter(|i| i.2)
                                .collect::<Vec<_>>();
        
        let pre_3mers = pre_tens
                                .par_iter()
                                .flat_map_iter(|i| i.3)
                                .collect::<Vec<_>>();
        
        let pre_2mers = pre_tens
                                .par_iter()
                                .flat_map_iter(|i| i.4)
                                .collect::<Vec<_>>();
            
        let pre_1mers = pre_tens
                                .par_iter()
                                .flat_map_iter(|i| i.5)
                                .collect::<Vec<_>>();

        let valids = pre_tens
                                .par_iter()
                                .map(|i| i.6)
                                .collect::<Vec<_>>();

        (
            pre_l4n1mers,
            pre_5mers,
            pre_4mers,
            pre_3mers,
            pre_2mers,
            pre_1mers,
            valids
        )
    });
    return    (qqq.0.into_pyarray(py),qqq.1.into_pyarray(py),qqq.2.into_pyarray(py),qqq.3.into_pyarray(py),qqq.4.into_pyarray(py),qqq.5.into_pyarray(py), qqq.6.into_pyarray(py));
    }

    #[pyfn(m)]
    #[pyo3(name = "find_nMer_distributions")]
    pub fn find_nMer_distributions<'py>(py: Python<'py>, contig_file: &str, mini_size: usize) -> (Vec<usize>,&'py PyArray1<f32>,&'py PyArray1<f32>,&'py PyArray1<f32>,&'py PyArray1<f32>,&'py PyArray1<f32>, &'py PyArray1<f32>, &'py PyArray1<f32>, &'py PyArray1<f32>, &'py PyArray1<f32>,&'py PyArray1<f32>, Vec<String>) {
        //rayon::ThreadPoolBuilder::new().num_threads(32).build_global().unwrap();
        //rayon::ThreadPoolBuilder::new().num_threads(64).build_global().unwrap();
        let file_info = fs::metadata(contig_file).unwrap();
        let file_size = file_info.len();

        let mut file = File::open(contig_file).unwrap();

        //read using multiple file handles for faster spead
        let mut read_buffer: Vec<u8> = Vec::new();
        par_read(&mut file, &mut read_buffer, file_size);
        //dbg!("read done");
        let mut decom_buffer: Vec<u8> = Vec::new();

        //covert to string representation for easy tooling (slightly slower, but easy)
        let string_slice_rep: &str = if contig_file.len() >3 && b'z'==contig_file.as_bytes()[contig_file.len()-1] {
            let mut gz = GzDecoder::new(&read_buffer[..]);
            gz.read_to_end(&mut  decom_buffer).unwrap();
            std::str::from_utf8(&decom_buffer[..]).unwrap() 
        } else {
            std::str::from_utf8(&read_buffer[..]).unwrap() 
        };

        //dbg!("here1");


       { };


        // split on new fasta name delimitor, skip first trivial
        let lines = string_slice_rep.par_split('>').filter(|x| x.len() >= mini_size).collect::<Vec<_>>();
        //dbg!("here2");]
        

        let num_contigs = lines.len();
        //dbg!("here3");

        // init contig name, contig vec containers
        let mut contig_names: Vec<String> = vec![String::new(); 0];
        let mut contigs: Vec<&str> = vec!["Bad Contig Name"; 0];
        //dbg!("here4");

        // we divide into contig name and actual contigs
        lines
            .par_iter()
            .enumerate()
            .map(|(q,line)| {
                let end_of_name_pos = line.find('\n').unwrap_or_else(|| {
                    dbg!("hh");
                    // dbg!("hh",lines.last().unwrap());
                    // dbg!("hh",lines[lines.len()-2]);
                    dbg!(q);
                    dbg!(line.len());
                    dbg!(&line[0..100]);

                    dbg!( lines.len());
                    panic!();
                });
                let end_of_name_pos2 = line[..end_of_name_pos].find(' ').unwrap_or(end_of_name_pos);
                let contig_len = line[end_of_name_pos + 1..line.len() - 1].chars().filter(|&x| x != '\n').count();
                // returns tuple of (contig name string slice, actual contig string slice ) for each line
            
                 (
                        line[..end_of_name_pos2].to_string(),
                        &line[end_of_name_pos + 1..line.len() - 1])
                // (
                //     line[..end_of_name_pos2].to_string(),
                //     &line[end_of_name_pos + 1..line.len() - 1])
            })
            .unzip_into_vecs(&mut contig_names, &mut contigs);
            //dbg!("here5");



        let contig_lens = contigs
                            .par_iter()
                            .map(|&x| x.chars().filter(|&x| x != '\n').count())
                            .collect::<Vec<usize>>();
                            //dbg!("here7");

        let contig_names = contig_names
            .par_drain(..)
            .enumerate()
            .filter(|(a,b)| contig_lens[*a] >= mini_size  )
            .map(|(a,ctg)|ctg)
            .collect::<Vec<_>>();
            //dbg!("here6");
                // count 1..5 (and l4n1) mers.
        let pre_tens = contigs
                .par_iter()
                .enumerate()
                .filter(|(a,b)| contig_lens[*a] >= mini_size  )
                .map(|(a,&ctg)| contig_2_nmer_distrs(ctg))
                .collect::<Vec<_>>();
                //dbg!("here6");

        let pretens_ry= contigs
                        .par_iter()
                         .enumerate()
                      .filter(|(a,b)| contig_lens[*a] >= mini_size )
                        .map(|(a,&ctg)| find_rymers(ctg))
                     .collect::<Vec<_>>();

        let pre_5mers = pre_tens
                                .par_iter()
                                .flat_map_iter(|i| i.0)
                                .collect::<Vec<_>>();
        
        let pre_l4n1mers = pre_tens
                                .par_iter()
                                .flat_map_iter(|i| i.1)
                                .collect::<Vec<_>>();

            
        let pre_4mers = pre_tens
                                .par_iter()
                                .flat_map_iter(|i| i.2)
                                .collect::<Vec<_>>();
        
        let pre_3mers = pre_tens
                                .par_iter()
                                .flat_map_iter(|i| i.3)
                                .collect::<Vec<_>>();
        
        let pre_2mers = pre_tens
                                .par_iter()
                                .flat_map_iter(|i| i.4)
                                .collect::<Vec<_>>();
            
        let pre_1mers = pre_tens
                                .par_iter()
                                .flat_map_iter(|i| i.5)
                                .collect::<Vec<_>>();
        let pre_10mers = pretens_ry
                                .par_iter()
                                .flat_map_iter(|i| i.0)
                                .collect::<Vec<_>>();

        let pre_9mers = pretens_ry
                                .par_iter()
                                .flat_map_iter(|i| i.1)
                                .collect::<Vec<_>>();

        let pre_8mers = pretens_ry
                                .par_iter()
                                .flat_map_iter(|i| i.2)
                                .collect::<Vec<_>>();
        let pre_7mers = pretens_ry
                                .par_iter()
                                .flat_map_iter(|i| i.3)
                                .collect::<Vec<_>>();
        let pre_6mers = pretens_ry
                                .par_iter()
                                .flat_map_iter(|i| i.4)
                                .collect::<Vec<_>>();

        let contig_lens = contig_lens
                                .par_iter()
                                .filter(|x| *x >= &0)
                                .map(|x| *x)
                                .collect::<Vec<usize>>();
                                //dbg!("here7");
        (
            contig_lens,
            pre_5mers.into_pyarray(py),
            pre_4mers.into_pyarray(py),
            pre_3mers.into_pyarray(py),
            pre_2mers.into_pyarray(py),
            pre_1mers.into_pyarray(py),
            pre_10mers.into_pyarray(py),
            pre_9mers.into_pyarray(py),
            pre_8mers.into_pyarray(py),
            pre_7mers.into_pyarray(py),
            pre_6mers.into_pyarray(py),
            contig_names
        )
    }
    #[pyfn(m)]
    #[pyo3(name = "write_fasta_bins")]
    pub fn write_fasta_bins<'py>(py: Python<'py>, contig_name: Vec<String>, bin_number: Vec<usize> ,src_contig_file: &str, outfolder: &str) -> usize {

        let file_info = fs::metadata(src_contig_file).unwrap();
        let file_size = file_info.len();

        let mut file = File::open(src_contig_file).unwrap();

        //read using multiple file handles for faster spead
        let mut read_buffer: Vec<u8> = Vec::new();
        par_read(&mut file, &mut read_buffer, file_size);

        let mut decom_buffer: Vec<u8> = Vec::new();

        //covert to string representation for easy tooling (slightly slower, but easy)
         let string_slice_rep: &str = if src_contig_file.len() >3 && b'z'==src_contig_file.as_bytes()[src_contig_file.len()-1] {    
            let mut gz = GzDecoder::new(&read_buffer[..]);
            gz.read_to_end(&mut  decom_buffer).unwrap();
            std::str::from_utf8(&decom_buffer[..]).unwrap() 
        } else {
            std::str::from_utf8(&read_buffer[..]).unwrap() 
        };


        // split on new fasta name delimitor, skip first trivial
        let lines = string_slice_rep.par_split('>').filter(|x| x.len() != 0).collect::<Vec<_>>();

        let num_contigs = lines.len();

  
        let contig_name_map = lines
            .par_iter()
            .map(|line| {
                let end_of_name_pos = line.find('\n').unwrap_or_else(|| {
                    dbg!(line);
                    dbg!(lines.last().unwrap());
                    dbg!(lines[lines.len()-2]);

                    dbg!(lines.len());
                    panic!();
                });
                let end_of_name_pos2 = line[..end_of_name_pos].find(' ').unwrap_or(end_of_name_pos);

                // returns tuple of (contig name string slice, actual contig string slice ) for each line
                (
                    &line[..end_of_name_pos2],
                    &line[end_of_name_pos + 1..line.len() - 1],
                )
            }) // collects tuples into these vecs 
            .collect::<HashMap<&str, &str>>();
        

        // maps  bin -> vector of (contig name, contig) pairs in bin
        let mut h: HashMap<usize, Vec<(&str,&str)>> = HashMap::new();

        for (ctg_nm, &bin_n) in contig_name.iter().zip(bin_number.iter()) {
            let mut v = h.entry(bin_n).or_insert_with(|| Vec::new());
            v.push((ctg_nm,*contig_name_map.get(&ctg_nm[..]).unwrap()));
        }
        fs::create_dir(outfolder).unwrap();
        let bases_binned = h.drain()
               .filter_map(|(bin_num, contents_vec)| {
                   let bases = contents_vec.iter()
                               .flat_map(|(a,b)| b.chars())
                               .filter(|&x| x != '\n')
                               .count();
                    if bases >= 200000 {
                        return Some((bin_num, contents_vec,bases));
                    }
                    None
               })
               .map(|(bin_num, contents_vec,bases)|{
                    let dest= Path::join(Path::new(outfolder), bin_num.to_string());
                    let mut file = File::create(dest).unwrap();
                    for (name,ctg) in contents_vec.iter() {
                        file.write_all(&[b'>']);
                        file.write_all(name.as_bytes());
                        file.write_all(&[b'\n']);
                        file.write_all(ctg.as_bytes());
                        file.write_all(&[b'\n']);
                    }
                    file.write_all(&[b'\n']);
                    bases
               }).sum();
        return bases_binned;



    }
    Ok(())
}
