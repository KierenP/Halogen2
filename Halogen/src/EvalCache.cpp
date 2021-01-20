#include "EvalCache.h"

void EvalCacheTable::AddEntry(uint64_t key, int eval)
{
	table[key % TableSize].key = key;
	table[key % TableSize].eval = eval;
}

bool EvalCacheTable::GetEntry(uint64_t key, int& eval)
{
	if (table[key % TableSize].key != key)
	{
		return false;
	}

	eval = table[key % TableSize].eval;
	return true;
}

void EvalCacheTable::Reset()
{
	table = std::vector<EvalCacheEntry>(TableSize);
}
