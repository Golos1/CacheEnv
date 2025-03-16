Custom Gymnasium Environment simulating a variable TTL cache. At each step, the agent knows the current items in the cache, their time to live, and which are dirty or not.
Additionally, the agent knows what row was requested last and whether it was written to (making it dirty if present in the cache) or read. Negative reward is given for cache misses
and for dirty rows being in the cache at the time they are requested; positive reward is given for cache hits.
