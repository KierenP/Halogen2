#include "EvalCache.h"

//1MB in size
constexpr size_t TableSize = 1024 * 1024 / sizeof(EvalCacheEntry);

//mask to get the lower 32 bits
constexpr size_t MASK = 0xffffffff;	

void EvalCacheTable::AddEntry(uint64_t key, int eval)
{
	uint64_t index = key % TableSize;
	table[index].key = key & MASK;
	table[index].eval = eval;
}

bool EvalCacheTable::GetEntry(uint64_t key, int& eval)
{
	uint64_t index = key % TableSize;
	if (table[index].key != (key & MASK))
	{
		return false;
	}
	else
	{
		eval = table[index].eval;
		return true;
	}
}

void EvalCacheTable::Reset()
{
	table.clear();
	table.resize(TableSize);
}
