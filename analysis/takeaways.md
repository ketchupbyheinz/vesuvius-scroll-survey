* **Coverage.** __S_TRACED__ survey patches were traced in the 21 eligible scrolls; __S_OK__ pass the sheet check and
  __S_INKED__ were inked, in 20 scrolls. No PHerc. 175B trace passed the sheet check (__PHerc0175B__) and few PHerc. 268
  traces did (__PHerc0268__). Another __F_TRACED__ patches were traced to follow up leads, plus one PHerc. 1545 patch
  from the Windows end-to-end test.
* **What known text looks like, and its limits.** On PHerc. 343 the known-text segment and an overlapping trace a few
  voxels inward score high on both ink scores, with the signal on the inner face. An overlapping trace a few voxels
  outward, a patch a few windings away and a patch elsewhere score flat. A flat card means "nothing found here",
  not "nothing in this scroll", and a trace over real text can still score flat.
* **Most survey patches look like blank papyrus to these models.** Survey medians: face asymmetry __S_ASYM_MED__
  (known text 3.2 and 4.0), face consistency __S_CONS_MED__ (known text 0.52 and 0.77).
* **The strongest-looking lead was largely air gaps.** One survey patch, PHerc. 846A at (2765, 2952, 4205), scores
  like the known text on both ink scores, its follow-ups mostly agree (15 of 17 inner face), and its strongest
  follow-up (2601, 2784, 3851) beats the known text on face asymmetry (5.55), though with a high crack score (0.24).
  The 2.4 µm scan settles it: on both patches the agreed ink falls on gaps between layers 1.8 to 2.5 times as often
  as elsewhere, and the same sheet traced again in the sharp scan scores far lower (1.63 / 0.29). This lead is
  dropped.
* **Read the crack score next to the ink scores.** On the four sharp-scan patches it was consistent with the check
  (0.16 to 0.24 with a gap signal, 0.13 without; survey median 0.14). The other earlier leads with high ink scores
  also have raised crack scores: PHerc. 846B (its patches lean on dark CT) and the PHerc. 1545 Windows-test patch
  (0.19). PHerc. 211 had already been dropped: its neighbouring windings flip sides and seed43 does not confirm it.
* **The most writing-like low-crack patch is PHerc. 1545 at (3051, 3484, 14673)** (consistency 0.64, crack score
  0.13), but it does not hold up beyond its own 4 cm²: 4 of 6 model pairs agree, its neighbours score flat or show
  no clear side, and grown to 12 cm² it scores flat. **No letters can be read yet.**
* **What to do with this.** Run the card on any new trace (~6 minutes on a laptop GPU once traced) and read the crack
  score next to the two ink scores; where a sharper scan exists, run the sharp-scan check before spending time on a
  patch. The traces here are ~4 cm² auto-grown patches, and grown larger they drift across windings: clean, large
  single-sheet surfaces are what a First Letters attempt needs next.
