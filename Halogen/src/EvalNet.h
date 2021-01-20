#pragma once
#include "Position.h"
#include <functional>
#include <valarray>
#include <array>
#include <algorithm>

//needed for SEE
constexpr int pieceValueVector[N_PIECE_TYPES] = { 91, 532, 568, 715, 1279, 5000 };

bool DeadPosition(const Position& position);

int EvaluatePositionNet(const Position& position, EvalCacheTable& evalTable);

int PieceValues(unsigned int Piece);

