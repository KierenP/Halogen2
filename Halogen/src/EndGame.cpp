#include "EndGame.h"

enum EndGames
{
    KXK,
    KBNK,
};

constexpr std::array<int, N_SQUARES> KingCenterDistance =
{
    6, 5, 4, 3, 3, 4, 5, 6,
    5, 4, 3, 2, 2, 3, 4, 5,
    4, 3, 2, 1, 1, 2, 3, 4,
    3, 2, 1, 0, 0, 1, 2, 3,
    3, 2, 1, 0, 0, 1, 2, 3,
    4, 3, 2, 1, 1, 2, 3, 4,
    5, 4, 3, 2, 2, 3, 4, 5,
    6, 5, 4, 3, 3, 4, 5, 6,
};

constexpr std::array<std::array<int, N_SQUARES>, N_PLAYERS> KingDistanceColorCorner =
{
    7, 6, 5, 4, 3, 2, 1, 0,
    6, 5, 4, 3, 2, 1, 0, 1,
    5, 4, 3, 2, 1, 0, 1, 2,
    4, 3, 2, 1, 0, 1, 2, 3,
    3, 2, 1, 0, 1, 2, 3, 4,
    2, 1, 0, 1, 2, 3, 4, 5,
    1, 0, 1, 2, 3, 4, 5, 6,
    0, 1, 2, 3, 4, 5, 6, 7,

    0, 1, 2, 3, 4, 5, 6, 7,
    1, 0, 1, 2, 3, 4, 5, 6,
    2, 1, 0, 1, 2, 3, 4, 5,
    3, 2, 1, 0, 1, 2, 3, 4,
    4, 3, 2, 1, 0, 1, 2, 3,
    5, 4, 3, 2, 1, 0, 1, 2,
    6, 5, 4, 3, 2, 1, 0, 1,
    7, 6, 5, 4, 3, 2, 1, 0,
};

template<EndGames type>
int EndGame(const Position& position, Players side);

//Rescale so that the highest scores become close to MATE_IN_MAX_PLY and the lowest scores are a little higher than TB_WIN_SCORE
int EndGameAdjustment(int score, int high, int low)
{
    int scale = (MATE_IN_MAX_PLY - TB_WIN_SCORE) / (high - low);
    return score * scale + TB_WIN_SCORE - (low * scale);
}

template<>
int EndGame<KXK>(const Position& position, Players side)
{
    int score = KingCenterDistance[position.GetKing(!side)];
    return EndGameAdjustment(score, 6, 0);
}

template<>
int EndGame<KBNK>(const Position& position, Players side)
{
    int score = KingDistanceColorCorner[SquareColour[LSB(position.GetPieceBB(BISHOP, side))]][position.GetKing(!side)];
    return EndGameAdjustment(score, 7, 0);
}

bool DeduceEndGame(const Position& position, int& eval)
{
    std::array<int, N_PIECES> count = {};
    for (int i = 0; i < N_PIECES; i++)
    {
        count[i] = GetBitCount(position.GetPieceBB(static_cast<Pieces>(i)));
    }

    if (count == std::array<int, N_PIECES>({ 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1 })) //KRK
    {
        eval = EndGame<KXK>(position, WHITE);
        return true;
    }

    if (count == std::array<int, N_PIECES>({ 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1 })) //KRK
    {
        eval = EndGame<KXK>(position, BLACK);
        return true;
    }

    if (count == std::array<int, N_PIECES>({ 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1 })) //KQK
    {
        eval = EndGame<KXK>(position, WHITE);
        return true;
    }

    if (count == std::array<int, N_PIECES>({ 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1 })) //KQK
    {
        eval = EndGame<KXK>(position, BLACK);
        return true;
    }

    if (count == std::array<int, N_PIECES>({ 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1 })) //KBNK
    {
        eval = EndGame<KBNK>(position, WHITE);
        return true;
    }

    if (count == std::array<int, N_PIECES>({ 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1 })) //KBNK
    {
        eval = EndGame<KBNK>(position, BLACK);
        return true;
    }

    if (count == std::array<int, N_PIECES>({ 0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 1 })) //KBBK
    {
        eval = EndGame<KXK>(position, WHITE);   //use same strategy as rook/queen: Push to edge/corner
        return true;
    }

    if (count == std::array<int, N_PIECES>({ 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 1 })) //KBBK
    {
        eval = EndGame<KXK>(position, BLACK);   //use same strategy as rook/queen: Push to edge/corner
        return true;
    }

    return false;
}