// Physical allocation admission lives beside MemoryLedger's lock and reservation entries.
// The check and new owner charge must be atomic even when demand and optional growth race.
