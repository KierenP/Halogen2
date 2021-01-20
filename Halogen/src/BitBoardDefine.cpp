#include "BitBoardDefine.h"

constexpr std::array<int, N_SQUARES> index64 = {
	0, 47,  1, 56, 48, 27,  2, 60,
   57, 49, 41, 37, 28, 16,  3, 61,
   54, 58, 35, 52, 50, 42, 21, 44,
   38, 32, 29, 23, 17, 11,  4, 62,
   46, 55, 26, 59, 40, 36, 15, 53,
   34, 51, 20, 43, 31, 22, 10, 45,
   25, 39, 14, 33, 19, 30,  9, 24,
   13, 18,  8, 12,  7,  6,  5, 63
};

Players operator!(const Players& val)
{
	return val == WHITE ? BLACK : WHITE;
}

//--------------------------------------------------------------------------
//Below code adapted with permission from Terje, author of Weiss.

uint64_t BishopAttacksMagic[0x1480];
uint64_t RookAttacksMagic[0x19000];
void InitSliderAttacks(Magic m[64], uint64_t* table, const int steps[4]);

//--------------------------------------------------------------------------

void BBInit()
{
	const int BSteps[4] = { 7, 9, -7, -9 };
	const int RSteps[4] = { 8, 1, -8, -1 };

	InitSliderAttacks(BishopTable, BishopAttacksMagic, BSteps);
	InitSliderAttacks(RookTable, RookAttacksMagic, RSteps);
}

char PieceToChar(unsigned int piece)
{
	assert(piece <= N_PIECES);

	char PieceChar[13] = { 'p', 'n', 'b', 'r', 'q', 'k', 'P', 'N', 'B', 'R', 'Q', 'K', ' ' };
	return PieceChar[piece];
}

unsigned int Piece(unsigned int piecetype, unsigned int colour)
{
	assert(piecetype < N_PIECE_TYPES);
	assert(colour < N_PLAYERS);

	return piecetype + N_PIECE_TYPES * colour;
}

Pieces Piece(PieceTypes type, Players colour)
{
	assert(type < N_PIECE_TYPES);
	assert(colour < N_PLAYERS);

	return static_cast<Pieces>(type + N_PIECE_TYPES * colour);
}

Square GetPosition(File file, Rank rank)
{
	assert(file < N_FILES);
	assert(rank < N_RANKS);

	return static_cast<Square>(rank * 8 + file);
}

unsigned int GetPosition(unsigned int file, unsigned int rank)
{
	assert(file < N_FILES);
	assert(file < N_RANKS);

	return rank * 8 + file;
}

unsigned int GetBitCount(uint64_t bb)
{
#if defined(_MSC_VER) && defined(USE_POPCNT) && defined(_WIN64)
	return __popcnt64(bb);
#elif defined(__GNUG__) && defined(USE_POPCNT)
	return __builtin_popcountll(bb);
#else
	return std::bitset<std::numeric_limits<uint64_t>::digits>(bb).count();
#endif
}

unsigned int AlgebraicToPos(std::string str)
{
	if (str == "-")
		return N_SQUARES;

	assert(str.length() >= 2);

	return (str[0] - 97) + (str[1] - 49) * 8;		
}

unsigned int ColourOfPiece(unsigned int piece)
{
	assert(piece < N_PIECES);

	return piece / N_PIECE_TYPES;
}

int LSBpop(uint64_t &bb)
{
	assert(bb != 0);

	int index = LSB(bb);

	bb &= bb - 1;
	return index;
}

int LSB(uint64_t bb)
{
#if defined(_MSC_VER) && defined(USE_POPCNT) && defined(_WIN64)
	unsigned long index;
	_BitScanForward64(&index, bb);
	return index;
#elif defined(__GNUG__) && defined(USE_POPCNT)
	return __builtin_ctzll(bb);
#else
	/**
	 * bitScanForward
	 * @author Kim Walisch (2012)
	 * @param bb bitboard to scan
	 * @precondition bb != 0
	 * @return index (0..63) of least significant one bit
	 */
	const uint64_t debruijn64 = uint64_t(0x03f79d71b4cb0a89);
	return index64[((bb ^ (bb - 1)) * debruijn64) >> 58];
#endif
}

//--------------------------------------------------------------------------
//Below code adapted with permission from Terje, author of Weiss.

#ifndef USE_PEXT
constexpr uint64_t RookMagics[64] = {
	0xA180022080400230ULL, 0x0040100040022000ULL, 0x0080088020001002ULL, 0x0080080280841000ULL,
	0x4200042010460008ULL, 0x04800A0003040080ULL, 0x0400110082041008ULL, 0x008000A041000880ULL,
	0x10138001A080C010ULL, 0x0000804008200480ULL, 0x00010011012000C0ULL, 0x0022004128102200ULL,
	0x000200081201200CULL, 0x202A001048460004ULL, 0x0081000100420004ULL, 0x4000800380004500ULL,
	0x0000208002904001ULL, 0x0090004040026008ULL, 0x0208808010002001ULL, 0x2002020020704940ULL,
	0x8048010008110005ULL, 0x6820808004002200ULL, 0x0A80040008023011ULL, 0x00B1460000811044ULL,
	0x4204400080008EA0ULL, 0xB002400180200184ULL, 0x2020200080100380ULL, 0x0010080080100080ULL,
	0x2204080080800400ULL, 0x0000A40080360080ULL, 0x02040604002810B1ULL, 0x008C218600004104ULL,
	0x8180004000402000ULL, 0x488C402000401001ULL, 0x4018A00080801004ULL, 0x1230002105001008ULL,
	0x8904800800800400ULL, 0x0042000C42003810ULL, 0x008408110400B012ULL, 0x0018086182000401ULL,
	0x2240088020C28000ULL, 0x001001201040C004ULL, 0x0A02008010420020ULL, 0x0010003009010060ULL,
	0x0004008008008014ULL, 0x0080020004008080ULL, 0x0282020001008080ULL, 0x50000181204A0004ULL,
	0x48FFFE99FECFAA00ULL, 0x48FFFE99FECFAA00ULL, 0x497FFFADFF9C2E00ULL, 0x613FFFDDFFCE9200ULL,
	0xFFFFFFE9FFE7CE00ULL, 0xFFFFFFF5FFF3E600ULL, 0x0010301802830400ULL, 0x510FFFF5F63C96A0ULL,
	0xEBFFFFB9FF9FC526ULL, 0x61FFFEDDFEEDAEAEULL, 0x53BFFFEDFFDEB1A2ULL, 0x127FFFB9FFDFB5F6ULL,
	0x411FFFDDFFDBF4D6ULL, 0x0801000804000603ULL, 0x0003FFEF27EEBE74ULL, 0x7645FFFECBFEA79EULL,
};

constexpr uint64_t BishopMagics[64] = {
	0xFFEDF9FD7CFCFFFFULL, 0xFC0962854A77F576ULL, 0x5822022042000000ULL, 0x2CA804A100200020ULL,
	0x0204042200000900ULL, 0x2002121024000002ULL, 0xFC0A66C64A7EF576ULL, 0x7FFDFDFCBD79FFFFULL,
	0xFC0846A64A34FFF6ULL, 0xFC087A874A3CF7F6ULL, 0x1001080204002100ULL, 0x1810080489021800ULL,
	0x0062040420010A00ULL, 0x5028043004300020ULL, 0xFC0864AE59B4FF76ULL, 0x3C0860AF4B35FF76ULL,
	0x73C01AF56CF4CFFBULL, 0x41A01CFAD64AAFFCULL, 0x040C0422080A0598ULL, 0x4228020082004050ULL,
	0x0200800400E00100ULL, 0x020B001230021040ULL, 0x7C0C028F5B34FF76ULL, 0xFC0A028E5AB4DF76ULL,
	0x0020208050A42180ULL, 0x001004804B280200ULL, 0x2048020024040010ULL, 0x0102C04004010200ULL,
	0x020408204C002010ULL, 0x02411100020080C1ULL, 0x102A008084042100ULL, 0x0941030000A09846ULL,
	0x0244100800400200ULL, 0x4000901010080696ULL, 0x0000280404180020ULL, 0x0800042008240100ULL,
	0x0220008400088020ULL, 0x04020182000904C9ULL, 0x0023010400020600ULL, 0x0041040020110302ULL,
	0xDCEFD9B54BFCC09FULL, 0xF95FFA765AFD602BULL, 0x1401210240484800ULL, 0x0022244208010080ULL,
	0x1105040104000210ULL, 0x2040088800C40081ULL, 0x43FF9A5CF4CA0C01ULL, 0x4BFFCD8E7C587601ULL,
	0xFC0FF2865334F576ULL, 0xFC0BF6CE5924F576ULL, 0x80000B0401040402ULL, 0x0020004821880A00ULL,
	0x8200002022440100ULL, 0x0009431801010068ULL, 0xC3FFB7DC36CA8C89ULL, 0xC3FF8A54F4CA2C89ULL,
	0xFFFFFCFCFD79EDFFULL, 0xFC0863FCCB147576ULL, 0x040C000022013020ULL, 0x2000104000420600ULL,
	0x0400000260142410ULL, 0x0800633408100500ULL, 0xFC087E8E4BB2F736ULL, 0x43FF9E4EF4CA2C89ULL,
};
#endif

Magic BishopTable[64];
Magic RookTable[64];

// Helper function that returns a bitboard with the landing square of
// the step, or an empty bitboard if the step would go outside the board
uint64_t LandingSquareBB(const int sq, const int step) {
	const unsigned int to = sq + step;
	return (uint64_t)(to <= SQ_H8 && std::max(AbsFileDiff(sq, to), AbsRankDiff(sq, to)) <= 2) << (to & SQ_H8);
}

// Helper function that makes a slider attack bitboard
uint64_t MakeSliderAttackBB(const int sq, const uint64_t occupied, const int steps[4]) {

	uint64_t attacks = 0;

	for (int dir = 0; dir < 4; ++dir) {

		int s = sq;
		while (!(occupied & SquareBB[s]) && LandingSquareBB(s, steps[dir]))
			attacks |= SquareBB[(s += steps[dir])];
	}

	return attacks;
}

// Initializes slider attack lookups
void InitSliderAttacks(Magic m[64], uint64_t *table, const int steps[4]) {

#ifndef USE_PEXT
	const uint64_t* magics = steps[0] == 8 ? RookMagics : BishopMagics;
#endif

	for (uint32_t sq = 0; sq < N_SQUARES; ++sq) {

		m[sq].attacks = table;

		// Construct the mask
		uint64_t edges = ((RankBB[RANK_1] | RankBB[RANK_8]) & ~RankBB[GetRank(sq)])
			| ((FileBB[FILE_A] | FileBB[FILE_H]) & ~FileBB[GetFile(sq)]);

		m[sq].mask = MakeSliderAttackBB(sq, 0, steps) & ~edges;

#ifndef USE_PEXT
		m[sq].magic = magics[sq];
		m[sq].shift = 64 - GetBitCount(m[sq].mask);
#endif

		uint64_t occupied = 0;
		do {
			m[sq].attacks[AttackIndex(sq, occupied, m)] = MakeSliderAttackBB(sq, occupied, steps);
			occupied = (occupied - m[sq].mask) & m[sq].mask; // Carry rippler
			table++;
		} while (occupied);
	}
}

//--------------------------------------------------------------------------