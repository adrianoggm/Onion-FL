# SEC1: secure aggregation stays out

- **Issue:** #227 (SEC1), which takes over what is still open of #39, #47 and #51.
- **Status:** decided, 2026-11-10. TLS and authentication on MQTT and gRPC are built in the same issue; secure aggregation is not.

## Decision

**No secure aggregation plugin for now.** "Secure aggregation" here means pairwise masking (Bonawitz et al., CCS 2017) or an equivalent, where the aggregator learns only the sum of its children's updates. TLS (#227) already keeps outsiders on the network from reading or injecting updates. Secure aggregation would also hide each update from the aggregator, and that conflicts with what the framework is for.

## Why

1. **It only computes sums.** The aggregator gets masked vectors and can only add them. The aggregators that make the framework useful need each update in the clear:
   - the robust ones: `median`, `trimmed_mean`, `krum`, `multi_krum`, `geometric_median`, `bulyan` and `norm_clip`;
   - the diagnostics, attacks and staleness weighting, which work on individual updates.

   Under masking, only `fedavg` and `mean` (and `dp_fedavg` with clipping moved to the edge) would still work.
2. **It changes the round protocol.** Masks cancel only when every masked child reports. With partial quorum, late updates or a crashed edge, the aggregator has to recover the missing masks from Shamir shares. That costs two or three extra message rounds per aggregation, at every level of the tree, with key agreement in `hello`. The simulator, the byte counts per link and the sim ≡ real equivalence tests would all have to model those extra rounds.
3. **The privacy goal is already covered another way.** `local_dp` protects each update from a curious aggregator, and `dp_fedavg` protects the output. Both are plugins, both are measured, and neither restricts the aggregator.
4. **No research question needs it.** The TFM measures hierarchical learning, its cost and its fidelity. Secure aggregation adds cost (fixed-point encoding, masks, recovery rounds) but would not change any result we report.

## If it is taken up again

- An aggregator plugin `secure_sum` (fedavg only) plus a masking step at the edge, with fixed-point vectors in Z_2^32 and pairwise seeds from an X25519 key agreement carried in `hello`.
- Start with full quorum and no dropouts, refusing anything else in the config. Add Shamir recovery after that.
- Its extra messages go through the same `Context`, so `sim` counts their bytes and time like any other message.
