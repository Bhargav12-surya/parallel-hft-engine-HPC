# Viva questions and concise answers

1. **Why use a serial reference?**

   It supplies a simple behavioral baseline for orders, fills and accounting.

2. **Why integer price ticks?**

   They remove binary floating-point ambiguity from discrete price comparisons.

3. **What does Philox indexing accomplish?**

   Each event can obtain its own random stream regardless of worker scheduling.

4. **Does the same seed prove correctness?**

   No. Known-answer RNG tests and book/accounting checks are also needed.

5. **What is price-time priority?**

   Better prices execute first; orders at the same price execute FIFO.

6. **What is an IOC order?**

   It executes available matching quantity immediately and discards the remainder.

7. **How does the optimized best-price search work?**

   It scans occupancy words and returns the head of the best price FIFO.

8. **What happens at fixed capacity?**

   An invalid or occupied resting slot can reject the order; storage is not unlimited.

9. **Why is self-trading handled?**

   Cancelling the strategy maker before a self-cross avoids artificial self-generated fills.

10. **How is marked P&L computed?**

   Cash plus signed inventory times the current mark, in integer ticks.

11. **Why divide portfolio risk budgets by symbol?**

   It makes independent pair ownership deterministic at the cost of conservative allocation.

12. **Why might a loss halt leave inventory?**

   The book may not have enough liquidity for immediate liquidation.

13. **What is a Hawkes process?**

   A point process where arrivals temporarily raise the intensity of later arrivals.

14. **Why require alpha below beta?**

   It bounds branching intensity for the exponential self-excitation model.

15. **What adds correlation between symbols?**

   Common and sector random shocks shared by indexed streams.

16. **What is a data race?**

   Unsynchronized conflicting accesses to the same memory; in C it can be undefined behavior.

17. **Why use one SPSC queue per worker?**

   A single producer and consumer simplify ownership and acquire/release synchronization.

18. **What does a release store publish?**

   Earlier writes become visible to a consumer that acquires that publication.

19. **Why a triple buffer?**

   Producer, consumer and middle slots avoid overwriting the snapshot currently being read.

20. **Why own pairs together?**

   Both legs can make coupled decisions without a cross-thread book lock.

21. **Static versus dynamic OpenMP scheduling?**

   Static has lower dispatch overhead; dynamic may balance variable per-pair work.

22. **What is false sharing?**

   Independent writable data share a cache line and cause unnecessary coherence traffic.

23. **Why measure critical, atomic and reduction?**

   They expose different synchronization and privatization costs for correct aggregation.

24. **Why pin a worker?**

   It can reduce migration noise and improve cache locality; affinity is not a universal speedup.

25. **What does MPI Bcast do?**

   Distributes one rank’s configuration/model to all participating ranks.

26. **Why Scatterv and Gatherv?**

   They support unequal pair counts and corresponding variable amounts of state.

27. **What does Allreduce add to SGD?**

   It combines gradients so ranks apply the same model update.

28. **Why use derived MPI datatypes?**

   They describe record fields and padding explicitly rather than assuming packed layout.

29. **What must remain alive during Isend?**

   The send buffer until the matching completion wait succeeds.

30. **Strong versus weak scaling?**

   Strong keeps total work fixed; weak increases total work with the worker count.

31. **Why can a central topology bottleneck?**

   All workers compete for one rank’s communication/processing capacity.

32. **Why OpenCL C in this project?**

   It supplies custom device kernels while keeping application source in C.

33. **Why require fp64?**

   The shared deterministic market calculations use double-precision intermediates.

34. **What is local-memory tiling?**

   A workgroup reuses a small loaded block across many arithmetic operations.

35. **Why can a GPU lose at batch one?**

   Transfers, queueing and launch overhead can outweigh the small amount of work.

36. **Are kernel time and wall time interchangeable?**

   No. End-to-end time also includes transfers, copies, synchronization and other work.

37. **Why might queue-duration sums exceed wall time?**

   Independent queues can execute concurrently.

38. **What is cuBLAS used for?**

   Dense covariance matrix multiplication through its C API.

39. **Does cuRAND guarantee this project’s Philox sequence?**

   No. Generator family alone does not specify identical stream mapping or output transformation.

40. **Why hold out later observations?**

   It approximates predicting future data and avoids random-split temporal leakage.

41. **Why a label gap?**

   It prevents training labels from reaching into the start of the validation window.

42. **What does softmax provide?**

   Normalized probabilities over sell, hold and buy scores.

43. **What does backpropagation compute?**

   The loss gradient with respect to every trainable weight and bias.

44. **What is Q-learning’s update?**

   Move Q toward immediate reward plus discounted best next-state value.

45. **Why penalize inventory?**

   Unbounded inventory can dominate reward with large uncontrolled exposure.

46. **How does UCB1 balance choices?**

   It adds an exploration bonus to each arm’s estimated reward.

47. **Why is CG suitable for the example?**

   The regularized covariance system is symmetric positive definite.

48. **Why can Cholesky fail?**

   The supplied matrix may not be positive definite or may suffer numerical problems.

49. **VaR versus CVaR?**

   VaR is a loss quantile; CVaR averages losses beyond that threshold.

50. **Is the project Sharpe annualized?**

   No. It is an explicitly labeled per-sample statistic.

51. **What does zero device p99 mean here?**

   No per-event CPU-style latency histogram was collected for that backend.

52. **How is cloud cost estimated?**

   Measured hours multiplied by an entered illustrative hourly price.

53. **Why does latency-coupled P&L change?**

   Orders arrive later into a changed book and may fill differently or fail risk checks.

54. **Can this app prove a profitable strategy?**

   No. It is a synthetic teaching environment with simplified market and execution models.
