# Market and strategies

## Synthetic market

Interarrival times use Hawkes self-excitation with thinning and an exponential decay. The configuration requires alpha < beta. Fundamental log price follows bounded jump-diffusion with common, sector and individual normal shocks. Noise traders, inventory-aware liquidity providers, momentum agents, fundamental agents and execution agents submit/cancel orders. Price emerges from book matching. Calm, bull, bear, sideways, high-volatility, flash-crash and news-shock regimes alter this process. This is a small synthetic agent model, not a calibrated exchange.

Every market match contributes to trade OHLC/volume and observed VWAP. Candles cover `sample_every` events, not fixed clock durations. Empty-trade candles use the mark. The detailed plotted price is in integer ticks; the price card converts using `tick_size`.

## Strategies

| ID / config name | Signal |
|---|---|
| 0 / sma | Fast versus slow rolling means |
| 1 / ema | Fast versus slow fixed-point EMA |
| 2 / mean_reversion | Rolling price z-score with enter/exit bands |
| 3 / breakout | Prior high/low plus top-depth confirmation |
| 4 / rsi | Smoothed gains/losses, 30/70 thresholds |
| 5 / macd | Fast-minus-slow EMA versus signal EMA |
| 6 / imbalance | Signed top-of-book quantity ratio |
| 7 / market_making | Inventory-skewed, variance-dependent bid/ask quotes |
| 8 / pairs | Rolling OLS residual z-score; paired symbols and approximate hedge quantity |
| 9 / execution | TWAP parent target over event horizon |
| 10 / mlp | Sell/hold/buy probabilities from a 32–64–32–3 MLP |
| 11 / qlearning | Greedy width/skew action from a trained 81×9 table |
| 12 / selector | Four k-means feature regimes and three UCB1 strategy arms |

The maker is inspired by inventory/reservation-price models; its bounded discrete quotes are not an exact continuous-time optimal control solution. Pair beta is rolling, constrained positive, and hedge size is clipped by order limits. Leg fills need not be simultaneous. Breakout uses top-depth confirmation rather than a calibrated exchange volume model.

Features include lagged returns, trend, variance, spread, depth imbalance, RSI, price deviations, inventory and observed trading activity. Online Welford normalization uses past/current observed values, never future labels or hidden fundamental value. Training generates 16-event-ahead price-change labels, uses a chronological 75/25 split and a 16-sample gap. No normalization fit uses the validation future.

MLP uses cross-entropy SGD, ReLU hidden layers, softmax output and weight decay. OpenMP uses private minibatch gradients and a fixed host reduction; OpenCL uses sample gradients and a weight-update kernel; MPI Allreduce averages minibatch contributions. Q-learning uses epsilon exploration, discounted reward and an inventory-square penalty, then averages independently trained episode tables. The selector's k-means centroids are fit during CPU MLP training; inference uses UCB1 rewards from actual realized marked P&L changes.

All strategies share quantity/position, equal-partition gross/loss, collar and token-bucket rate checks. Delayed orders are rechecked on arrival without charging the rate bucket twice. The delay queue is bounded at 32 per symbol; overflowing queued orders are dropped and counted. This is a deliberate capacity limit, not an unlimited exchange network.
